import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="PTA Dashboard", layout="wide")
st.title("Pressure Transient Analysis Dashboard")

st.sidebar.header('Reservoir Parameters')
porosity = st.sidebar.number_input('Enter Porosity(∅)', value=0.16)
viscosity = st.sidebar.number_input('Enter Fluid Viscosity(μ)', value=0.25)
FVF = st.sidebar.number_input('Enter Formation Volume Factor(B)', value=1.06)
ct = st.sidebar.number_input('Enter Total Compressibility(ct)', value=0.000012, format="%f")
h = st.sidebar.number_input('Enter Pay Zone Thickness(h)', value=144.35)
rw = st.sidebar.number_input('Enter Wellbore Radius(rw)', value=0.25)
q = st.sidebar.number_input('Enter Flow Rate(q)', value=2898.0)


st.sidebar.header('Upload Data')
uploaded_file = st.sidebar.file_uploader("Upload Gauge Data (.txt)", type=['txt'])

if uploaded_file is not None:
    
    def load_gauge_data(file):
        # Changed 'file_path' to 'file' to accept the uploaded object
        df = pd.read_csv(file, sep='\s+', usecols=[0, 1, 2], header=None) 
        df.columns = ['Date', 'Time', 'Pressure']
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d-%m-%y %H:%M:%S')
        time_zero = df['Datetime'].iloc[0]
        df['Elapsed_Hours'] = (df['Datetime'] - time_zero).dt.total_seconds() / 3600.0
        max_time = df['Elapsed_Hours'].max()
        clean_df = df[(df['Elapsed_Hours'] >= 1) & (df['Elapsed_Hours'] <= (max_time - 1))].copy()
        clean_df = clean_df[['Elapsed_Hours', 'Pressure']]
        clean_df.columns = ['Time_hrs', 'Pressure']
        return clean_df.reset_index(drop=True)

    df = load_gauge_data(uploaded_file)

    # Determining the Shut-in Time
    df['Pressure_Diff'] = df['Pressure'].diff().abs()
    shut_in_id = df[df['Pressure_Diff'] >= 10].index[0] - 1
    t_shutin = df.loc[shut_in_id, 'Time_hrs']
    p_shutin = df.loc[shut_in_id, 'Pressure']

    df = df.drop(columns=['Pressure_Diff'])

    print(f"Shut-in detected at {t_shutin} hours with a pressure of {p_shutin} psi")

    # Deleting all data from before shut-in
    df = df.loc[shut_in_id:].copy()

    # Resetting the time column to show shut-in time
    df['Time_hrs'] = df['Time_hrs'] - t_shutin
    df = df.reset_index(drop=True)

    # Thinning the data
    df = df.iloc[::60].copy()
    df = df.reset_index(drop=True)

    # Calculating dP
    p_wf0 = df['Pressure'].iloc[0]
    df['dP'] = (df['Pressure'] - p_wf0).abs()

    def bourdet_derivative(df):
        
        df['log10_dt'] = np.log10(df['Time_hrs'].replace(0, 1e-10))
        
        delta_dP = df['dP'].shift(-3) - df['dP'].shift(3)
        delta_log10_dt = df['log10_dt'].shift(-3) - df['log10_dt'].shift(3)
        
        df['Bourdet_Deriv'] = (delta_dP / delta_log10_dt).abs()
        
        df = df.drop(columns=['log10_dt'])
        
        return df

    df = bourdet_derivative(df)

    def iarf_region(df, window_size=5, tolerance=1.5, max_gap=2):

        # 1. Calculate the rolling maximum and minimum over a 5-row window
        rolling_max = df['Bourdet_Deriv'].rolling(window=window_size).max()
        rolling_min = df['Bourdet_Deriv'].rolling(window=window_size).min()
        
        # 2. Check where the difference between max and min is <= 2
        is_flat = (rolling_max - rolling_min) <= tolerance
        
        # 3. Find the indices where this condition is True
        flat_indices = df[is_flat].index
        
        if len(flat_indices) > 0:
            # Grab the very first time this 5-row flatness occurs
            end_idx = flat_indices[0]
            start_idx = end_idx - window_size + 1

            last_flat_idx = end_idx
            gap_count = 0

            for idx in df.index[df.index > end_idx]:
                if is_flat.loc[idx]:
                    last_flat_idx = idx
                    gap_count = 0
                else:
                    gap_count += 1
                    if gap_count > max_gap:
                        break

            end_idx = last_flat_idx
            
            # Extract the exact start and end times of this region
            t_start = df.loc[start_idx, 'Time_hrs']
            t_end = df.loc[end_idx, 'Time_hrs']
            
            # Calculating the average derivative value (m)
            slope_m = df.loc[start_idx:end_idx, 'Bourdet_Deriv'].mean()
            
            return t_start, t_end, slope_m
        else:
            return None, None, None

    t_start, t_end, slope_m= iarf_region(df)

    print(f"IARF detected from {t_start} to {t_end} hrs. Average Derivative: {slope_m}.")

    # --- P(1 hr) via semi-log straight-line extrapolation ---
    def calculate_p1hr(df, t_start, t_end):
        """
        Fits a straight line through the IARF window in Pressure vs log10(Delta t)
        space and extrapolates it to Delta t = 1 hr.

        Returns (p1hr, slope, intercept). slope is the semilog straight-line
        slope 'm' (psi per log cycle) used later in Horner analysis. Returns
        (None, None, None) if no IARF region was detected.
        """
        if t_start is None or t_end is None:
            return None, None, None

        iarf_df = df[(df['Time_hrs'] >= t_start) & (df['Time_hrs'] <= t_end)]
        iarf_df = iarf_df[iarf_df['Time_hrs'] > 0]  # log10(0) is undefined

        if len(iarf_df) < 2:
            return None, None, None

        log_t = np.log10(iarf_df['Time_hrs'])

        # Pressure = slope * log10(Delta t) + intercept
        slope, intercept = np.polyfit(log_t, iarf_df['Pressure'], 1)

        # At Delta t = 1 hr, log10(1) = 0, so P(1hr) is just the intercept
        p1hr = intercept

        return p1hr, slope, intercept

    p1hr, m_slope, _ = calculate_p1hr(df, t_start, t_end)
    if p1hr is not None:
        print(f"P(1hr) = {p1hr:.2f} psi | Semilog slope m = {m_slope:.2f} psi/cycle")
    else:
        print("P(1hr) not calculated (no IARF region detected).")

    # Generating the Diagnostic Plot
    def diagnostic_plot(df):
        
        fig = go.Figure()

        # 1. Plot the Pressure Change (ΔP) Curve
        fig.add_trace(go.Scatter(
            x=df['Time_hrs'], 
            y=df['dP'], 
            mode='lines+markers', 
            name='ΔP (Pressure Change)',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))

        # 2. Plot the Bourdet Derivative Curve
        fig.add_trace(go.Scatter(
            x=df['Time_hrs'], 
            y=df['Bourdet_Deriv'], 
            mode='lines+markers', 
            name="Bourdet Derivative (t*ΔP')",
            line=dict(color='red', width=2),
            marker=dict(size=4)
        ))

        # 3. Format the layout for Log-Log scale
        min_x = np.log10(df[df['Time_hrs'] > 0]['Time_hrs'].min())
        max_x = np.log10(df['Time_hrs'].max())

        max_y = np.log10(df['dP'].max())
        
        fig.update_layout(
            title="Diagnostic Plot",
            xaxis_title="Shut-in Time, Δt (hr)",
            yaxis_title="Pressure & Pressure Derivative (psi)",
            xaxis=dict(
                type="log",
                range=[min_x - 0.1, max_x + 0.1] # Locks X-axis perfectly around your data
            ),
            yaxis=dict(
                type="log",
                range=[1, max_y + 0.2 ] # Starts at 10^0 (1 psi) and ends just above your max dP
            ),
            hovermode="x unified",
            template="plotly_white",
            height=600
        )

        if t_start and t_end:
            fig.add_vrect(
            x0=t_start, x1=t_end, 
            fillcolor="green", opacity=0.2, 
            layer="below", line_width=0,
            annotation_text="Auto-Detected IARF", 
            annotation_position="top left"
        )

        return fig

    diag_fig = diagnostic_plot(df)
    st.plotly_chart(diag_fig, width='stretch')



    st.subheader("Test Detection Parameters")
    
    st.info(f"**Shut-in Event:** Detected at **{t_shutin:.2f} hours** with a pressure of **{p_shutin:.1f} psi**")
    
    if t_start and t_end:
        st.success(
            f"**IARF Regime:** Detected from **{t_start:.2f}** to **{t_end:.2f} hrs** | "
            f"Average Derivative: **{slope_m:.2f}** |"
            f" P @ 1 hr: **{p1hr:.2f} psi**"
        )
    else:
        st.warning("IARF Region could not be automatically detected with current tolerance.")

    st.subheader("Analysis & Interpretation")
    
    
    k = 162.6 * q * FVF * viscosity / (abs(m_slope) * h)
    s = 1.1513 * ((abs(p1hr - p_wf0) / abs(m_slope)) - np.log10(k / (porosity * viscosity * ct * rw * rw)) + 3.23)
    
    
    col1, col2= st.columns(2)
    
    col1.metric("Permeability (k)", f"{k:.2f} md")
    col2.metric("Skin Factor (s)", f"{s:.2f}")

else:
    st.info("Please upload a gauge data .txt file in the sidebar to begin analysis.")