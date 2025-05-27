import streamlit as st
import base64
import pandas as pd
import numpy as np
import plotly.express as px
import uuid

# ==== PAGE CONFIG ====
st.set_page_config(page_title="Water Quality Dashboard", page_icon="📊", layout="wide")

# ==== LOAD FONT ====
font_base64 = None
try:
    with open("Montserrat-Bold.ttf", "rb") as f:
        font_base64 = base64.b64encode(f.read()).decode()
except FileNotFoundError:
    st.warning("Montserrat-Bold.ttf not found. Using default font.")

# ==== LOAD BANNER IMAGE ====
banner_img_base64 = None
try:
    with open("images/header.png", "rb") as img_file:
        banner_img_base64 = base64.b64encode(img_file.read()).decode()
except FileNotFoundError:
    st.warning("header.png not found. Using solid background for tabs.")


# ==== LOAD DATA ====
@st.cache_data
def load_data():
    bfar_df = pd.DataFrame()
    philvolcs_df = pd.DataFrame()
    try:
        bfar_df = pd.read_parquet('datasets/cleaned_dataset.parquet', engine='pyarrow')
    except FileNotFoundError:
        st.error("cleaned_dataset.parquet not found.")
    except Exception as e:
        st.error(f"Error loading cleaned_dataset.parquet: {e}")

    try:
        philvolcs_df = pd.read_parquet('datasets/PHIVOLCS.parquet', engine='pyarrow')
    except FileNotFoundError:
        st.error("PHIVOLCS.parquet not found.")
    except Exception as e:
        st.error(f"Error loading PHIVOLCS.parquet: {e}")

    return bfar_df, philvolcs_df


bfar_df, philvolcs_df = load_data()

# ==== LOAD TAAL INFO ====
taal_info = ""
try:
    with open("taalinfo.txt", "r") as f:
        taal_info = f.read()
except FileNotFoundError:
    st.warning("taalinfo.txt not found.")

hide_streamlit_style = """
    <style>
        /* Hide the Streamlit header and menu */
        header {visibility: hidden;}
        /* Optionally, hide the footer */
        .streamlit-footer {display: none;}
        /* Hide your specific div class, replace class name with the one you identified */
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ==== CSS STYLING ====
font_style = f"""
@font-face {{
    font-family: 'Montserrat';
    src: url("data:font/ttf;base64,{font_base64}") format('truetype');
}}
""" if font_base64 else ""

tab_style = f"""
.stTabs [data-baseweb="tab-list"] {{
    background-image: linear-gradient(rgba(255,255,255,0), rgba(255,255,255,0)), 
    url("data:image/png;base64,{banner_img_base64}");
    background-size: cover;
    background-repeat: no-repeat;
    padding: 30px;
    border-radius: 8px 8px 0px 0px;
}}
""" if banner_img_base64 else """
.stTabs [data-baseweb="tab-list"] {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
    margin-bottom: 0px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
"""

st.markdown(f"""
<style>
    {font_style}
    .block-container {{
        max-width: 98% !important;
        padding: 0rem 0rem 0rem 0rem !important;  # Adjusted bottom padding for footer
        margin-top: 0px 0px 5px 5px !important;
    }}
    .stApp, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .section-header, .custom-label, .stTabs [data-baseweb="tab"] p,
    .stButton button, .stMultiSelect, .streamlit-expanderHeader p, .streamlit-expanderContent div,
    .custom-text-primary, .custom-text-secondary {{
        font-family: {'Montserrat' if font_base64 else 'sans-serif'} !important;
    }}

    /* Selected parameters in multiselect (background and text color) */
    .stMultiSelect [data-baseweb="select"] .st-ae {{
        background-color: #004A99 !important;
        color: #FFFFFF !important;
    }}
    /* Checkbox styling (including Select All Parameters) */
    .stCheckbox input {{
        accent-color: #004A99 !important;
    }}
    /* Checkbox label text color */
    .stCheckbox label {{
        color: #222831 !important;
    }}
    /* Outline of dropdown menus (selectbox and multiselect) */
    [data-baseweb="select"] > div {{
        border: 0px solid #004A99 !important;
        border-radius: 8px !important;
    }}
    {tab_style}
    .stTabs [data-baseweb="tab-panel"] {{
        padding: 0px 5px 5px 5px;
        margin: 0 0 0 0;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 25px; justify-content: right; padding-right: 3rem; }}

    .stTabs [data-baseweb="tab"] {{ 
        background-color: rgba(128, 150, 173, 0.5);
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, background-color 0.3s ease, box-shadow 0.3s ease;
        border: none !important;
        padding: 20px;
        backdrop-filter: blur(2px);
    }}
    .stTabs [data-baseweb="tab"] p {{ 
        color: #002244;
        font-weight: 600;
        margin: 0;
        font-size: 15px;
    }}
    .stTabs [data-baseweb="tab"]:hover {{ 
        background-color: rgba(88, 139, 206, 0.5) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(2px);
    }}
    .stTabs [data-baseweb="tab"]:hover p {{ color: #FFFFFF !important; }}
    .stTabs [data-baseweb="tab-highlight"] {{ display: none !important; }}
    .stTabs [aria-selected="true"] {{ 
        background-color: rgba(0, 74, 173, 0.95) !important;
        color: #ffffff !important;
        box-shadow: 0 5px 10px rgba(0, 0, 0, 0.15);
        transform: translateY(-3px);
    }}
    .stTabs [aria-selected="true"] p {{ color: #ffffff !important; }}
    .custom-divider {{ border-top: 1px solid #748DA6; margin-top: 10px; margin-bottom:15px; }}
    .custom-text-primary {{ color: #222831; font-size: 18px; padding-top: 0px; }}
    .custom-text-secondary {{ color: #393E46; font-size: 16px; }}

    .full-width-footer {{
        max-width: 100% !important;
        display: block;
        margin: 0 auto;
        border-radius: 0 0 8px 8px;
        width: 100%;
        position: relative;
        bottom: 0;
    }}
</style>
""", unsafe_allow_html=True)

# ==== TABS ====
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Homepage"


def set_active_tab(tab):
    st.session_state.active_tab = tab


tab1, tab3, tab4, tab_info = st.tabs(["🏠 Homepage", "📈 Visualizations", "🔮 Prediction", "ℹ️ About"])

# ==== Homepage  ====
with tab1:
    set_active_tab("Homepage")
    st.markdown("""
    <style>
    .full-width-gif {
        max-width: 100% !important;
        display: block;
        margin: 0 auto;
        margin-top: 0rem;
        border-radius: 0px;
    }
    </style>
    """, unsafe_allow_html=True)

    try:
        with open("images/homepage.gif", "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
        st.markdown(
            f'<img src="data:image/gif;base64,{img_base64}" class="full-width-gif" alt="Homepage GIF">',
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("homepage.gif not found in the images folder. Please ensure the file exists.")
    except Exception as e:
        st.error(f"Error loading homepage.gif: {e}")

    st.markdown("<div class='custom-divider' style='margin-bottom: 7rem;'></div>", unsafe_allow_html=True)

# ==== Visualization ====
# ==== Visualization ====
with tab3:
    set_active_tab("Visualizations")
    if 'visualization' not in st.session_state:
        st.session_state.visualization = "Correlation Matrix"

    # Function to calculate WQI for raw data
    def calculate_wqi(values_dict, params):
        try:
            normalized = []
            for param in params:
                values = values_dict[param]
                # Remove NaN values for normalization
                valid_values = values[~np.isnan(values)]
                if len(valid_values) == 0:
                    normalized.append(np.full_like(values, np.nan))
                    continue
                # Min-max normalization to [0, 1]
                min_val = valid_values.min()
                max_val = valid_values.max()
                if max_val == min_val:
                    normalized_values = np.zeros_like(values)
                else:
                    normalized_values = (values - min_val) / (max_val - min_val)
                # Replace NaN values with 0 for WQI calculation
                normalized_values = np.nan_to_num(normalized_values, nan=0.0)
                normalized.append(normalized_values)
            normalized = np.array(normalized)
            # Calculate WQI as mean of normalized values, scaled to 0-100
            wqi = np.mean(normalized, axis=0, where=~np.isnan(normalized)) * 100
            wqi = np.clip(wqi, 0, 100)  # Ensure [0, 100]
            remarks = []
            for val in wqi:
                if np.isnan(val):
                    remarks.append("N/A")
                elif val <= 20:
                    remarks.append("Poor")
                elif val <= 40:
                    remarks.append("Fair")
                elif val <= 60:
                    remarks.append("Good")
                elif val <= 80:
                    remarks.append("Very Good")
                else:
                    remarks.append("Excellent")
            return wqi, remarks
        except Exception as e:
            st.error(f"Error calculating WQI: {str(e)}")
            return np.zeros(len(values_dict[params[0]])), ["N/A"] * len(values_dict[params[0]])

    st.markdown(""" 
    <style>
    [data-testid="stButton"] button {
        transition: transform 0.3s ease, background-color 0.3s ease, box-shadow 0.3s ease;
        background-color: rgba(128, 150, 173, 0.15) !important;
        border-radius: 8px !important;
        padding: 0px 13px !important;
        width: 180px !important; 
        text-align: center !important;
        color: #002244 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        font-family: Montserrat, sans-serif !important;
        border: none !important;
        margin-bottom: 0px !important;
        cursor: pointer !important;
        display: block !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.);
    }
    [data-testid="stButton"] button:hover {
        background-color: rgba(88, 139, 206, 0.5) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.0);
        transform: translateX(5px);
    }
    [data-testid="stButton"] button[kind="primary"] {
        background-color: #004A99 !important; 
        color: #FFFFFF !important;
        box-shadow: 0 5px 10px rgba(0, 0, 0, 0);
        transform: translateX(5px);
    }
    [data-testid="stButton"] button:active {
        background-color: #003366 !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        transform: translateX(5px);
        transition: transform 0.3s ease, background-color 0.3s ease, box-shadow 0.3s ease, color 0.3s ease;
    }
    div.js-plotly-plot {
        border: 2px solid #004A99 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        box-shadow: 0 0px 5px rgba(0, 0, 0, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

    colA, colB = st.columns([1, 5])
    with colA:
        st.markdown(
            "<div class='custom-text-primary' style='margin-bottom: 10px; margin-top: 0px; "
            "font-size: 15px; text-align: justify;'>Select a Visualization</div>",
            unsafe_allow_html=True)
        visualization_options = [
            "Correlation Matrix",
            "Scatter Plots",
            "Distributions",
            "Histogram",
            "Box Plot",
            "Line Chart",
            "WQI Over Time",
            "Descriptive Analytics",
        ]
        for option in visualization_options:
            is_selected = st.session_state.visualization == option
            st.button(
                option,
                key=f"vis_button_{option.lower().replace(' ', '_')}",
                on_click=lambda opt=option: st.session_state.update(visualization=opt),
                type="primary" if is_selected else "secondary"
            )
        visualization = st.session_state.visualization

    with colB:
        if visualization == "Correlation Matrix":
            if not bfar_df.empty:
                available_sites = sorted(bfar_df['Site'].astype(str).unique())
                numeric_params = sorted([col for col in bfar_df.select_dtypes(include=np.number).columns
                                         if col not in ['Date', 'Site', 'Year', 'Month', 'Weather Condition',
                                                        'Wind Direction']
                                         and bfar_df[col].notna().any()])
                col1, col2 = st.columns([5, 2])
                with col2:
                    st.markdown(
                        "<div class='custom-text-primary' style='margin-bottom: 0px; margin-top: 0px; "
                        "font-size: 15px; text-align: justify;'>Correlation Matrix Configuration</div>",
                        unsafe_allow_html=True)
                    sites = ['All Sites'] + available_sites
                    selected_site = st.selectbox("Select Site:", sites, key="heatmap_site")
                    select_all_params = st.checkbox("Select All Parameters", key="heatmap_select_all_params")
                    selected_params = st.multiselect(
                        "Select Parameters (min 2):",
                        options=numeric_params,
                        default=numeric_params if select_all_params else numeric_params[:min(len(numeric_params), 5)],
                        key="heatmap_params"
                    )
                    min_date = bfar_df['Date'].min()
                    max_date = bfar_df['Date'].max()
                    start_date = st.date_input("Start Date (Optional):", value=None, min_value=min_date,
                                               max_value=max_date,
                                               key="heatmap_start_date")
                    end_date = st.date_input("End Date (Optional):", value=None, min_value=min_date, max_value=max_date,
                                             key="heatmap_end_date")
                with col1:
                    try:
                        filtered_df = bfar_df.copy()
                        if selected_site != 'All Sites':
                            filtered_df = filtered_df[filtered_df['Site'] == selected_site]
                        if start_date:
                            start_date = pd.to_datetime(start_date)
                            filtered_df = filtered_df[filtered_df['Date'] >= start_date]
                        if end_date:
                            end_date = pd.to_datetime(end_date)
                            filtered_df = filtered_df[filtered_df['Date'] <= end_date]
                        if start_date and end_date and start_date > end_date:
                            st.error("Error: Start date cannot be after end date.")
                        elif len(selected_params) < 2:
                            st.info("Please select at least two parameters for the correlation heatmap.")
                        else:
                            corr_df = filtered_df[selected_params].dropna()
                            if len(corr_df) < 2:
                                st.warning("Not enough data points after filtering to calculate correlation.")
                            else:
                                corr_matrix = corr_df.corr().round(2)
                                if not corr_matrix.empty:
                                    fig_heatmap = px.imshow(
                                        corr_matrix,
                                        text_auto=True,
                                        aspect="auto",
                                        color_continuous_scale='Blues',
                                        title=f"Correlation Matrix for {selected_site}"
                                    )
                                    fig_heatmap.update_traces(
                                        xgap=0,
                                        ygap=0
                                    )
                                    fig_heatmap.update_layout(
                                        xaxis_title="Parameters",
                                        yaxis_title="Parameters",
                                        height=550,
                                        plot_bgcolor='white',
                                        paper_bgcolor='white',
                                        title_font=dict(
                                            size=18,
                                            family='Montserrat' if font_base64 else 'sans-serif'
                                        ),
                                        title_x=0.03,
                                        margin=dict(l=20, r=20, t=60, b=20),
                                        font=dict(family='Montserrat' if font_base64 else 'sans-serif')
                                    )
                                    fig_heatmap.update_xaxes(tickangle=45)
                                    fig_heatmap.update_yaxes(tickangle=0)
                                    st.plotly_chart(fig_heatmap, use_container_width=True)
                                else:
                                    st.warning("No correlation data to display.")
                    except Exception as e:
                        st.error(f"Error generating heatmap: {e}")
            else:
                st.error("Water Quality data not loaded. Cannot display heatmap.")
        elif visualization == "Scatter Plots":
            if not bfar_df.empty:
                numeric_params = sorted([col for col in bfar_df.select_dtypes(include=np.number).columns
                                         if col not in ['Date', 'Site', 'Year', 'Month', 'Weather Condition',
                                                        'Wind Direction']
                                         and bfar_df[col].notna().any()])
                if len(numeric_params) < 2:
                    st.warning("At least two numeric parameters are required for scatter plots.")
                else:
                    col1, col2 = st.columns([5, 2])
                    with col2:
                        st.markdown(
                            "<div class='custom-text-primary' style='margin-bottom: 0px; margin-top: 0px; "
                            "font-size: 17px; text-align: justify;'>Scatter Plot Configuration</div>",
                            unsafe_allow_html=True)
                        x_axis = st.selectbox("Select X-axis Parameter:", numeric_params, key="scatter_x")
                        y_axis = st.selectbox("Select Y-axis Parameter:", numeric_params,
                                              index=1 if len(numeric_params) > 1 else 0, key="scatter_y")
                        sites = ['All Sites'] + sorted(bfar_df['Site'].astype(str).unique())
                        selected_site = st.selectbox("Filter by Site (Optional):", sites, key="scatter_site_filter")
                        show_best_fit = st.checkbox("Show Best-Fit Line", value=True, key="scatter_best_fit")
                        min_date = bfar_df['Date'].min()
                        max_date = bfar_df['Date'].max()
                        start_date = st.date_input("Start Date (Optional):", value=None, min_value=min_date,
                                                   max_value=max_date, key="scatter_start_date")
                        end_date = st.date_input("End Date (Optional):", value=None, min_value=min_date,
                                                 max_value=max_date, key="scatter_end_date")

                    with col1:
                        try:
                            filtered_df = bfar_df.copy()
                            if selected_site != 'All Sites':
                                filtered_df = filtered_df[filtered_df['Site'] == selected_site]
                            if start_date and end_date:
                                start_date = pd.to_datetime(start_date)
                                end_date = pd.to_datetime(end_date)
                                if start_date > end_date:
                                    st.error("Error: Start date cannot be after end date.")
                                    st.stop()
                                filtered_df = filtered_df[(filtered_df['Date'] >= start_date) &
                                                          (filtered_df['Date'] <= end_date)]
                            elif start_date:
                                filtered_df = filtered_df[filtered_df['Date'] >= pd.to_datetime(start_date)]
                            elif end_date:
                                filtered_df = filtered_df[filtered_df['Date'] <= pd.to_datetime(end_date)]
                            filtered_df = filtered_df.dropna(subset=[x_axis, y_axis])
                            if filtered_df.empty:
                                st.warning("No data available for the selected parameters and filters.")
                                st.stop()
                            if len(filtered_df) < 2:
                                st.warning("Not enough data points (minimum 2 required) to generate scatter plot.")
                                st.stop()
                            if not (filtered_df[x_axis].dtype in [np.float64, np.int64] and
                                    filtered_df[y_axis].dtype in [np.float64, np.int64]):
                                st.error("Selected parameters must be numeric for scatter plot.")
                                st.stop()
                            fig_scatter = px.scatter(filtered_df, x=x_axis, y=y_axis, color='Site',
                                                     title=f"{y_axis} vs. {x_axis}", hover_data=['Date'])
                            if show_best_fit:
                                try:
                                    x_data = filtered_df[x_axis].values
                                    y_data = filtered_df[y_axis].values
                                    coeffs = np.polyfit(x_data, y_data, 1)
                                    slope, intercept = coeffs
                                    x_range = np.array([x_data.min(), x_data.max()])
                                    y_fit = slope * x_range + intercept
                                    fig_scatter.add_scatter(
                                        x=x_range,
                                        y=y_fit,
                                        mode='lines',
                                        name='Best Fit',
                                        line=dict(color='red', width=2)
                                    )
                                except Exception as e:
                                    st.warning(f"Unable to compute best-fit line: {str(e)}")
                            fig_scatter.update_layout(
                                height=500,
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                title_font=dict(
                                    size=18,
                                    family='Montserrat' if font_base64 else 'sans-serif'
                                ),
                                title_x=0.03,
                                margin=dict(l=20, r=20, t=60, b=20),
                                font=dict(family='Montserrat' if font_base64 else 'sans-serif')
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating scatter plot: {str(e)}")
                            st.stop()
            else:
                st.error("Water Quality data not loaded. Cannot display scatter plots.")
                st.stop()

        elif visualization == "Distributions":
            if not bfar_df.empty or not philvolcs_df.empty:
                bfar_params = sorted([col for col in bfar_df.select_dtypes(include=np.number).columns
                                      if col not in ['Date', 'Site', 'Year', 'Month', 'Weather Condition',
                                                     'Wind Direction']
                                      and bfar_df[col].notna().any()])
                philvolcs_params = sorted([col for col in philvolcs_df.select_dtypes(include=np.number).columns
                                           if col not in ['Year', 'Month', 'Day', 'Latitude', 'Longitude']
                                           and philvolcs_df[col].notna().any()])
                param_options = ([f"{param} (Water Quality)" for param in bfar_params] +
                                 [f"{param} (PHIVOLCS)" for param in philvolcs_params])
                if not param_options:
                    st.warning("No numeric parameters available for distribution plots.")
                else:
                    col1, col2 = st.columns([5, 2])
                    with col2:
                        st.markdown(
                            "<div class='custom-text-primary' style='margin-bottom: 0px; margin-top: 0px; "
                            "font-size: 17px; text-align: justify;'>Distributions Configuration</div>",
                            unsafe_allow_html=True)
                        selected_param = st.selectbox("Select Parameter for Distribution:", param_options,
                                                      index=0, key="dist_param")
                        sites = ['All Sites'] + sorted(bfar_df['Site'].astype(str).unique()) if not bfar_df.empty else [
                            'All Sites']
                        selected_site = st.selectbox("Filter by Site (Optional, Water Quality only):", sites,
                                                     key="dist_site_filter")
                        min_date = bfar_df['Date'].min() if not bfar_df.empty else philvolcs_df['Date'].min()
                        max_date = bfar_df['Date'].max() if not bfar_df.empty else philvolcs_df['Date'].max()
                        show_trend_line = st.checkbox("Show Trend Line", value=False, key="dist_trend_line")
                        start_date = st.date_input("Start Date (Optional):", value=None, min_value=min_date,
                                                   max_value=max_date,
                                                   key="dist_start_date")
                        end_date = st.date_input("End Date (Optional):", value=None, min_value=min_date,
                                                 max_value=max_date,
                                                 key="dist_end_date")
                    with col1:
                        param_name = selected_param.split(" (")[0]
                        dataset = selected_param.split(" (")[1].rstrip(")")
                        if dataset == "Water Quality" and not bfar_df.empty:
                            filtered_df = bfar_df.copy()
                            if selected_site != 'All Sites':
                                filtered_df = filtered_df[filtered_df['Site'] == selected_site]
                        elif dataset == "PHIVOLCS" and not philvolcs_df.empty:
                            filtered_df = philvolcs_df.copy()
                        else:
                            filtered_df = pd.DataFrame()
                        if not filtered_df.empty:
                            if start_date and end_date:
                                start_date = pd.to_datetime(start_date)
                                end_date = pd.to_datetime(end_date)
                                if start_date > end_date:
                                    st.error("Error: Start date cannot be after end date.")
                                else:
                                    filtered_df = filtered_df[(filtered_df['Date'] >= start_date) &
                                                              (filtered_df['Date'] <= end_date)]
                            elif start_date:
                                filtered_df = filtered_df[filtered_df['Date'] >= pd.to_datetime(start_date)]
                            elif end_date:
                                filtered_df = filtered_df[filtered_df['Date'] <= pd.to_datetime(end_date)]
                            data = filtered_df[param_name].dropna()
                            title = f"Distribution of {param_name} ({dataset}) in {selected_site}"
                            if not data.empty:
                                fig_hist = px.histogram(data, x=param_name, nbins=30, title=title,
                                                        color_discrete_sequence=['#004A99'])
                                fig_hist.update_traces(opacity=0.75)
                                if show_trend_line:
                                    try:
                                        from scipy.stats import gaussian_kde
                                        import numpy as np

                                        if len(data) >= 2 and data.nunique() > 1:
                                            kde = gaussian_kde(data)
                                            x_range = np.linspace(data.min(), data.max(), 100)
                                            kde_vals = kde(x_range)
                                            hist_height = data.size
                                            kde_vals_scaled = kde_vals * hist_height * (data.max() - data.min()) / 30
                                            fig_hist.add_scatter(
                                                x=x_range,
                                                y=kde_vals_scaled,
                                                mode='lines',
                                                name='KDE Trend',
                                                showlegend=False,
                                                line=dict(color='red', width=2)
                                            )
                                        else:
                                            st.warning(
                                                "Not enough data or variability in the selected range to compute KDE trend line.")
                                    except Exception as e:
                                        st.warning(
                                            "Not enough data or variability in the selected range to compute KDE trend line.")
                                fig_hist.update_layout(
                                    showlegend=False,
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    height=500,
                                    xaxis_title=param_name,
                                    title_font=dict(
                                        size=18,
                                        family='Montserrat' if font_base64 else 'sans-serif'
                                    ),
                                    title_x=0.03,
                                    margin=dict(l=20, r=20, t=60, b=20),
                                    yaxis_title="Count",
                                    font=dict(family='Montserrat' if font_base64 else 'sans-serif')
                                )
                                st.plotly_chart(fig_hist, use_container_width=True)
                            else:
                                st.warning(f"No data available for {selected_param} after applying filters.")
                        else:
                            st.warning(f"No data available for {selected_param}.")
        elif visualization == "Histogram":
            if not bfar_df.empty or not philvolcs_df.empty:
                bfar_params = sorted([col for col in bfar_df.select_dtypes(include=np.number).columns
                                      if col not in ['Date', 'Site', 'Year', 'Month', 'Weather Condition',
                                                     'Wind Direction']
                                      and bfar_df[col].notna().any()])
                philvolcs_params = sorted([col for col in philvolcs_df.select_dtypes(include=np.number).columns
                                           if col not in ['Year', 'Month', 'Day', 'Latitude', 'Longitude']
                                           and philvolcs_df[col].notna().any()])
                param_options = ([f"{param} (Water Quality)" for param in bfar_params] +
                                 [f"{param} (PHIVOLCS)" for param in philvolcs_params])
                if not param_options:
                    st.warning("No numeric parameters available for histogram.")
                else:
                    col1, col2 = st.columns([5, 2])
                    with col2:
                        st.markdown(
                            "<div class='custom-text-primary' style='margin-bottom: 0px; margin-top: 0px; "
                            "font-size: 17px; text-align: justify;'>Histogram Configuration</div>",
                            unsafe_allow_html=True)
                        selected_param = st.selectbox("Select Parameter for Histogram:", param_options,
                                                      index=0, key="hist_param")
                        sites = ['All Sites'] + sorted(bfar_df['Site'].astype(str).unique()) if not bfar_df.empty else [
                            'All Sites']
                        selected_site = st.selectbox("Filter by Site (Optional, Water Quality only):", sites,
                                                     key="hist_site_filter")
                        min_date = bfar_df['Date'].min() if not bfar_df.empty else philvolcs_df['Date'].min()
                        max_date = bfar_df['Date'].max() if not bfar_df.empty else philvolcs_df['Date'].max()
                        start_date = st.date_input("Start Date (Optional):", value=None, min_value=min_date,
                                                   max_value=max_date,
                                                   key="hist_start_date")
                        end_date = st.date_input("End Date (Optional):", value=None, min_value=min_date,
                                                 max_value=max_date,
                                                 key="hist_end_date")
                    with col1:
                        param_name = selected_param.split(" (")[0]
                        dataset = selected_param.split(" (")[1].rstrip(")")
                        if dataset == "Water Quality" and not bfar_df.empty:
                            filtered_df = bfar_df.copy()
                            if selected_site != 'All Sites':
                                filtered_df = filtered_df[filtered_df['Site'] == selected_site]
                        elif dataset == "PHIVOLCS" and not philvolcs_df.empty:
                            filtered_df = philvolcs_df.copy()
                        else:
                            filtered_df = pd.DataFrame()
                        if not filtered_df.empty:
                            if start_date and end_date:
                                start_date = pd.to_datetime(start_date)
                                end_date = pd.to_datetime(end_date)
                                if start_date > end_date:
                                    st.error("Error: Start date cannot be after end date.")
                                else:
                                    filtered_df = filtered_df[(filtered_df['Date'] >= start_date) &
                                                              (filtered_df['Date'] <= end_date)]
                            elif start_date:
                                filtered_df = filtered_df[filtered_df['Date'] >= pd.to_datetime(start_date)]
                            elif end_date:
                                filtered_df = filtered_df[filtered_df['Date'] <= pd.to_datetime(end_date)]
                            data = filtered_df[param_name].dropna()
                            title = f"Histogram of {param_name} ({dataset}) in {selected_site}"
                            if not data.empty:
                                fig_hist = px.histogram(data, x=param_name, nbins=30, title=title,
                                                        color_discrete_sequence=['#004A99'])
                                fig_hist.update_traces(opacity=0.75)
                                fig_hist.update_layout(
                                    showlegend=False,
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    height=500,
                                    title_font=dict(
                                        size=18,
                                        family='Montserrat' if font_base64 else 'sans-serif'
                                    ),
                                    title_x=0.03,
                                    margin=dict(l=20, r=20, t=60, b=20),
                                    xaxis_title=param_name,
                                    yaxis_title="Count",
                                    font=dict(family='Montserrat' if font_base64 else 'sans-serif')
                                )
                                st.plotly_chart(fig_hist, use_container_width=True)
                            else:
                                st.warning(f"No data available for {selected_param} after applying filters.")
                        else:
                            st.warning(f"No data available for {selected_param}.")

        elif visualization == "Box Plot":
            if not bfar_df.empty or not philvolcs_df.empty:
                bfar_params = sorted([col for col in bfar_df.select_dtypes(include=np.number).columns
                                      if col not in ['Date', 'Site', 'Year', 'Month', 'Weather Condition',
                                                     'Wind Direction']
                                      and bfar_df[col].notna().any()])
                philvolcs_params = sorted([col for col in philvolcs_df.select_dtypes(include=np.number).columns
                                           if col not in ['Year', 'Month', 'Day', 'Latitude', 'Longitude']
                                           and philvolcs_df[col].notna().any()])
                param_options = ([f"{param} (Water Quality)" for param in bfar_params] +
                                 [f"{param} (PHIVOLCS)" for param in philvolcs_params])
                if not param_options:
                    st.warning("No numeric parameters available for box plot.")
                else:
                    col1, col2 = st.columns([5, 2])
                    with col2:
                        st.markdown(
                            "<div class='custom-text-primary' style='margin-bottom: 0px; margin-top: 0px; "
                            "font-size: 17px; text-align: justify;'>Box Plot Configuration</div>",
                            unsafe_allow_html=True)
                        selected_param = st.selectbox("Select Parameter for Box Plot:", param_options,
                                                      index=0, key="box_param")
                        sites = ['All Sites'] + sorted(bfar_df['Site'].astype(str).unique()) if not bfar_df.empty else [
                            'All Sites']
                        selected_site = st.selectbox("Filter by Site (Optional, Water Quality only):", sites,
                                                     key="box_site_filter")
                        min_date = bfar_df['Date'].min() if not bfar_df.empty else philvolcs_df['Date'].min()
                        max_date = bfar_df['Date'].max() if not bfar_df.empty else philvolcs_df['Date'].max()
                        start_date = st.date_input("Start Date (Optional):", value=None, min_value=min_date,
                                                   max_value=max_date,
                                                   key="box_start_date")
                        end_date = st.date_input("End Date (Optional):", value=None, min_value=min_date,
                                                 max_value=max_date,
                                                 key="box_end_date")
                    with col1:
                        param_name = selected_param.split(" (")[0]
                        dataset = selected_param.split(" (")[1].rstrip(")")
                        if dataset == "Water Quality" and not bfar_df.empty:
                            filtered_df = bfar_df.copy()
                            if selected_site != 'All Sites':
                                filtered_df = filtered_df[filtered_df['Site'] == selected_site]
                        elif dataset == "PHIVOLCS" and not philvolcs_df.empty:
                            filtered_df = philvolcs_df.copy()
                        else:
                            filtered_df = pd.DataFrame()
                        if not filtered_df.empty:
                            if start_date and end_date:
                                start_date = pd.to_datetime(start_date)
                                end_date = pd.to_datetime(end_date)
                                if start_date > end_date:
                                    st.error("Error: Start date cannot be after end date.")
                                else:
                                    filtered_df = filtered_df[(filtered_df['Date'] >= start_date) &
                                                              (filtered_df['Date'] <= end_date)]
                            elif start_date:
                                filtered_df = filtered_df[filtered_df['Date'] >= pd.to_datetime(start_date)]
                            elif end_date:
                                filtered_df = filtered_df[filtered_df['Date'] <= pd.to_datetime(end_date)]
                            data = filtered_df[[param_name]].dropna(subset=[param_name])
                            title = f"Box Plot of {param_name} ({dataset}) in {selected_site}"
                            if not data.empty:
                                fig_box = px.box(data, x=param_name, title=title,
                                                 color_discrete_sequence=['#004A99'])
                                fig_box.update_traces(marker=dict(size=5, opacity=0.75))
                                min_val = data[param_name].min()
                                max_val = data[param_name].max()
                                tick_vals = np.linspace(min_val, max_val, num=10).round(2)
                                fig_box.update_xaxes(
                                    tickvals=tick_vals,
                                    ticktext=[f"{val:.2f}" for val in tick_vals],
                                    gridcolor='rgba(200, 200, 200, 0.5)',
                                    showgrid=True,
                                    zeroline=False
                                )
                                fig_box.update_layout(
                                    showlegend=False,
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    height=500,
                                    title_font=dict(
                                        size=18,
                                        family='Montserrat' if font_base64 else 'sans-serif'
                                    ),
                                    title_x=0.03,
                                    margin=dict(l=20, r=20, t=60, b=20),
                                    xaxis_title=param_name,
                                    yaxis_title="",
                                    font=dict(family='Montserrat' if font_base64 else 'sans-serif')
                                )
                                st.plotly_chart(fig_box, use_container_width=True)
                            else:
                                st.warning(f"No data available for {selected_param} after applying filters.")
                        else:
                            st.warning(f"No data available for {selected_param}.")
        elif visualization == "Line Chart":
            if not bfar_df.empty or not philvolcs_df.empty:
                bfar_params = sorted([col for col in bfar_df.select_dtypes(include=np.number).columns
                                      if col not in ['Date', 'Site', 'Year', 'Month', 'Weather Condition',
                                                     'Wind Direction']
                                      and bfar_df[col].notna().any()])
                philvolcs_params = sorted([col for col in philvolcs_df.select_dtypes(include=np.number).columns
                                           if col not in ['Year', 'Month', 'Day', 'Latitude', 'Longitude']
                                           and philvolcs_df[col].notna().any()])
                param_options = ([f"{param} (Water Quality)" for param in bfar_params] +
                                 [f"{param} (PHIVOLCS)" for param in philvolcs_params])
                if not param_options:
                    st.warning("No numeric parameters available for line chart.")
                else:
                    col1, col2 = st.columns([5, 2])
                    with col2:
                        st.markdown(
                            "<div class='custom-text-primary' style='margin-bottom: 0px; margin-top: 0px; "
                            "font-size: 17px; text-align: justify;'>Line Chart Configuration</div>",
                            unsafe_allow_html=True)
                        compare_mode = st.radio("Compare By:", ["Parameters", "Sites"], index=0,
                                                key="line_compare_mode", horizontal=True)
                        if compare_mode == "Parameters":
                            selected_params = st.multiselect("Select Parameters for Comparison (at least 1):",
                                                             param_options,
                                                             default=[param_options[0]] if param_options else [],
                                                             key="line_params")
                            sites = ['All Sites'] + sorted(
                                bfar_df['Site'].astype(str).unique()) if not bfar_df.empty else [
                                'All Sites']
                            selected_site = st.selectbox("Filter by Site (Optional, Water Quality only):", sites,
                                                         key="line_site_filter")
                        else:
                            if not bfar_df.empty:
                                selected_param = st.selectbox("Select Parameter for Comparison:", param_options,
                                                              index=0, key="line_param")
                                sites = sorted(bfar_df['Site'].astype(str).unique())
                                selected_sites = st.multiselect("Select Sites for Comparison (at least 1):", sites,
                                                                default=[sites[0]] if sites else [], key="line_sites")
                            else:
                                st.warning("Site comparison is only available for Water Quality data.")
                                selected_param = None
                                selected_sites = []
                        min_date = bfar_df['Date'].min() if not bfar_df.empty else philvolcs_df['Date'].min()
                        max_date = bfar_df['Date'].max() if not bfar_df.empty else philvolcs_df['Date'].max()
                        start_date = st.date_input("Start Date (Optional):", value=None, min_value=min_date,
                                                   max_value=max_date,
                                                   key="line_start_date")
                        end_date = st.date_input("End Date (Optional):", value=None, min_value=min_date,
                                                 max_value=max_date,
                                                 key="line_end_date")
                    with col1:
                        if compare_mode == "Parameters":
                            if not selected_params:
                                st.warning("Please select at least one parameter for the line chart.")
                            else:
                                datasets = [param.split(" (")[1].rstrip(")") for param in selected_params]
                                param_names = [param.split(" (")[0] for param in selected_params]
                                if len(set(datasets)) > 1:
                                    st.error(
                                        "Please select parameters from the same dataset (either Water Quality or PHIVOLCS).")
                                else:
                                    dataset = datasets[0]
                                    if dataset == "Water Quality" and not bfar_df.empty:
                                        filtered_df = bfar_df.copy()
                                        if selected_site != 'All Sites':
                                            filtered_df = filtered_df[filtered_df['Site'] == selected_site]
                                    elif dataset == "PHIVOLCS" and not philvolcs_df.empty:
                                        filtered_df = philvolcs_df.copy()
                                    else:
                                        filtered_df = pd.DataFrame()
                                    if not filtered_df.empty:
                                        if start_date and end_date:
                                            start_date = pd.to_datetime(start_date)
                                            end_date = pd.to_datetime(end_date)
                                            if start_date > end_date:
                                                st.error("Error: Start date cannot be after end date.")
                                            else:
                                                filtered_df = filtered_df[(filtered_df['Date'] >= start_date) &
                                                                          (filtered_df['Date'] <= end_date)]
                                        elif start_date:
                                            filtered_df = filtered_df[filtered_df['Date'] >= pd.to_datetime(start_date)]
                                        elif end_date:
                                            filtered_df = filtered_df[filtered_df['Date'] <= pd.to_datetime(end_date)]
                                        data = filtered_df[['Date'] + param_names].dropna(subset=param_names)
                                        data = data.sort_values('Date')
                                        if not data.empty:
                                            melted_data = data.melt(id_vars=['Date'], value_vars=param_names,
                                                                    var_name='Parameter', value_name='Value')
                                            title = f"Line Chart of Selected Parameters ({dataset})"
                                            fig_line = px.line(melted_data, x='Date', y='Value', color='Parameter',
                                                               title=title)
                                            fig_line.update_traces(line=dict(width=2))
                                            fig_line.update_layout(
                                                showlegend=True,
                                                plot_bgcolor='white',
                                                paper_bgcolor='white',
                                                height=500,
                                                title_font=dict(
                                                    size=18,
                                                    family='Montserrat' if font_base64 else 'sans-serif'
                                                ),
                                                title_x=0.03,
                                                margin=dict(l=20, r=20, t=60, b=20),
                                                xaxis_title="Date",
                                                yaxis_title="Value",
                                                font=dict(family='Montserrat' if font_base64 else 'sans-serif')
                                            )
                                            st.plotly_chart(fig_line, use_container_width=True)
                                        else:
                                            st.warning(
                                                f"No data available for the selected parameters after applying filters.")
                                    else:
                                        st.warning(f"No data available for the selected parameters.")
                        else:
                            if not bfar_df.empty:
                                if not selected_sites:
                                    st.warning("Please select at least one site for the line chart.")
                                else:
                                    param_name = selected_param.split(" (")[0]
                                    dataset = selected_param.split(" (")[1].rstrip(")")
                                    if dataset != "Water Quality":
                                        st.error("Site comparison is only available for Water Quality data.")
                                    else:
                                        filtered_df = bfar_df.copy()
                                        filtered_df = filtered_df[filtered_df['Site'].isin(selected_sites)]
                                        if start_date and end_date:
                                            start_date = pd.to_datetime(start_date)
                                            end_date = pd.to_datetime(end_date)
                                            if start_date > end_date:
                                                st.error("Error: Start date cannot be after end date.")
                                            else:
                                                filtered_df = filtered_df[(filtered_df['Date'] >= start_date) &
                                                                          (filtered_df['Date'] <= end_date)]
                                        elif start_date:
                                            filtered_df = filtered_df[filtered_df['Date'] >= pd.to_datetime(start_date)]
                                        elif end_date:
                                            filtered_df = filtered_df[filtered_df['Date'] <= pd.to_datetime(end_date)]
                                        data = filtered_df[['Date', 'Site', param_name]].dropna(subset=[param_name])
                                        data = data.sort_values('Date')
                                        if not data.empty:
                                            title = f"Line Chart of {param_name} Across Sites (Water Quality)"
                                            fig_line = px.line(data, x='Date', y=param_name, color='Site',
                                                               title=title)
                                            fig_line.update_traces(line=dict(width=2))
                                            fig_line.update_layout(
                                                showlegend=True,
                                                plot_bgcolor='white',
                                                paper_bgcolor='white',
                                                height=500,
                                                title_font=dict(
                                                    size=18,
                                                    family='Montserrat' if font_base64 else 'sans-serif'
                                                ),
                                                title_x=0.03,
                                                margin=dict(l=20, r=20, t=60, b=20),
                                                xaxis_title="Date",
                                                yaxis_title=param_name,
                                                font=dict(family='Montserrat' if font_base64 else 'sans-serif')
                                            )
                                            st.plotly_chart(fig_line, use_container_width=True)
                                        else:
                                            st.warning(
                                                f"No data available for {param_name} at the selected sites after applying filters.")
                            else:
                                st.error("No Water Quality data loaded. Cannot display site comparison.")

        elif visualization == "WQI Over Time":
            if not bfar_df.empty:
                numeric_params = sorted([col for col in bfar_df.select_dtypes(include=np.number).columns
                                        if col not in ['Date', 'Site', 'Year', 'Month', 'Weather Condition',
                                                       'Wind Direction']
                                        and bfar_df[col].notna().any()])
                if not numeric_params:
                    st.warning("No numeric parameters available for WQI calculation.")
                else:
                    col1, col2 = st.columns([5, 2])
                    with col2:
                        st.markdown(
                            "<div class='custom-text-primary' style='margin-bottom: 0px; margin-top: 0px; "
                            "font-size: 17px; text-align: justify;'>WQI Over Time Configuration</div>",
                            unsafe_allow_html=True)
                        sites = ['All Sites'] + sorted(bfar_df['Site'].astype(str).unique())
                        selected_site = st.selectbox("Select Site:", sites, key="wqi_site_filter")
                        select_all_params = st.checkbox("Select All Parameters", key="wqi_select_all_params")
                        selected_params = st.multiselect(
                            "Select Parameters for WQI (min 1):",
                            options=numeric_params,
                            default=numeric_params if select_all_params else numeric_params[:min(len(numeric_params), 5)],
                            key="wqi_params"
                        )
                        min_date = bfar_df['Date'].min()
                        max_date = bfar_df['Date'].max()
                        start_date = st.date_input("Start Date (Optional):", value=None, min_value=min_date,
                                                   max_value=max_date, key="wqi_start_date")
                        end_date = st.date_input("End Date (Optional):", value=None, min_value=min_date,
                                                 max_value=max_date, key="wqi_end_date")
                    with col1:
                        try:
                            filtered_df = bfar_df.copy()
                            if selected_site != 'All Sites':
                                filtered_df = filtered_df[filtered_df['Site'] == selected_site]
                            if start_date:
                                start_date = pd.to_datetime(start_date)
                                filtered_df = filtered_df[filtered_df['Date'] >= start_date]
                            if end_date:
                                end_date = pd.to_datetime(end_date)
                                filtered_df = filtered_df[filtered_df['Date'] <= end_date]
                            if start_date and end_date and start_date > end_date:
                                st.error("Error: Start date cannot be after end date.")
                            elif not selected_params:
                                st.info("Please select at least one parameter for WQI calculation.")
                            else:
                                filtered_df = filtered_df.sort_values('Date')
                                values_dict = {param: filtered_df[param].values for param in selected_params}
                                wqi, wqi_remarks = calculate_wqi(values_dict, selected_params)
                                wqi_df = pd.DataFrame({
                                    "Date": filtered_df['Date'],
                                    "Water Quality Index": wqi,
                                    "WQI Remarks": wqi_remarks
                                })
                                if wqi_df.empty or wqi_df['Water Quality Index'].isna().all():
                                    st.warning("No valid WQI data to display after applying filters.")
                                else:
                                    st.dataframe(wqi_df, use_container_width=True)
                                    fig_wqi = px.line(wqi_df, x="Date", y="Water Quality Index",
                                                      title=f"Water Quality Index Over Time ({selected_site})",
                                                      color_discrete_sequence=['#004A99'])
                                    fig_wqi.update_traces(line=dict(width=2))
                                    fig_wqi.update_layout(
                                        showlegend=False,
                                        plot_bgcolor='white',
                                        paper_bgcolor='white',
                                        height=500,
                                        title_font=dict(
                                            size=18,
                                            family='Montserrat' if font_base64 else 'sans-serif'
                                        ),
                                        title_x=0.03,
                                        margin=dict(l=20, r=20, t=60, b=20),
                                        xaxis_title="Date",
                                        yaxis_title="Water Quality Index",
                                        font=dict(family='Montserrat' if font_base64 else 'sans-serif')
                                    )
                                    st.plotly_chart(fig_wqi, use_container_width=True)

                        except Exception as e:
                            st.error(f"Error generating WQI plot: {str(e)}")
            else:
                st.error("Water Quality data not loaded. Cannot display WQI over time.")

        elif visualization == "Descriptive Analytics":
            if not bfar_df.empty:
                st.markdown(
                    "<div class='custom-text-primary' style='margin-bottom: 8px; margin-top: 0px; "
                    "font-size: 20px; text-align: justify;'>Descriptive Analytics</div>",
                    unsafe_allow_html=True)
                numeric_params = [col for col in bfar_df.select_dtypes(include=np.number).columns
                                  if col not in ['Date', 'Site', 'Year', 'Month', 'Weather Condition', 'Wind Direction']
                                  and bfar_df[col].notna().any()]
                if numeric_params:
                    stats_df = bfar_df[numeric_params].describe().T
                    stats_df = stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
                    stats_df = stats_df.rename(columns={
                        'count': 'Count',
                        'mean': 'Mean',
                        'std': 'Std Dev',
                        'min': 'Min',
                        '25%': 'Q1',
                        '50%': 'Median',
                        '75%': 'Q3',
                        'max': 'Max'
                    })
                    stats_df = stats_df.round(2)
                    st.dataframe(stats_df, use_container_width=True)
                else:
                    st.warning("No numeric parameters available for descriptive analytics.")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("Water Quality data not loaded. Cannot display analytics or overview.")
# ==== Prediction ====
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import logging
import uuid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import linregress
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau

with tab4:
    set_active_tab("Prediction")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler('dashboard.log')]
    )
    logger = logging.getLogger(__name__)

    # Custom callback to track loss
    class LossHistory(Callback):
        def __init__(self):
            super().__init__()
            self.losses = []
            self.val_losses = []

        def on_epoch_end(self, epoch, logs=None):
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))

    # Preprocessing functions
    def prepare_univariate_data(data, param, window_size=7, val_split=0.2):
        try:
            values = data[param].dropna().values.reshape(-1, 1)
            if len(values) < window_size + 1:
                logger.warning(f"Insufficient data for {param}: {len(values)} rows")
                return None, None, None, None
            logger.info(f"Pre-normalized {param}: {values[:5].flatten()}")
            X, y = [], []
            for i in range(len(values) - window_size):
                X.append(values[i:i + window_size])
                y.append(values[i + window_size])
            X, y = np.array(X), np.array(y)
            if len(X) < 2:
                logger.warning(f"Not enough data points for {param}: {len(X)} samples")
                return None, None, None, None
            split_idx = int(len(X) * (1 - val_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            logger.info(f"Univariate {param}: {len(X_train)} train, {len(X_val)} val samples")
            return X_train, y_train, X_val, y_val
        except Exception as e:
            logger.error(f"Error preparing univariate data for {param}: {str(e)}")
            return None, None, None, None

    def prepare_multivariate_data(data, params, window_size=7, val_split=0.2):
        try:
            values = data[params].dropna().values
            if len(values) < window_size + 1:
                logger.warning(f"Insufficient multivariate data: {len(values)} rows")
                return None, None, None, None
            logger.info(f"Pre-normalized multivariate: {values[:5]}")
            X, y = [], []
            for i in range(len(values) - window_size):
                X.append(values[i:i + window_size])
                y.append(values[i + window_size])
            X, y = np.array(X), np.array(y)
            if len(X) < 2:
                logger.warning(f"Not enough multivariate data points: {len(X)} samples")
                return None, None, None, None
            split_idx = int(len(X) * (1 - val_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            logger.info(f"Multivariate: {len(X_train)} train, {len(X_val)} val samples")
            return X_train, y_train, X_val, y_val
        except Exception as e:
            logger.error(f"Error preparing multivariate data: {str(e)}")
            return None, None, None, None

    # Optimized model building
    def build_cnn(input_shape):
        try:
            model = Sequential([
                Input(shape=input_shape),
                Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
                BatchNormalization(),
                Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(100, activation='relu'),
                Dropout(0.3),
                Dense(1 if input_shape[-1] == 1 else input_shape[-1])
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            logger.info(f"Optimized CNN built: {input_shape}")
            return model
        except Exception as e:
            logger.error(f"Error building CNN: {str(e)}")
            return None

    def build_lstm(input_shape):
        try:
            model = Sequential([
                Input(shape=input_shape),
                Bidirectional(LSTM(100, activation='relu', return_sequences=True)),
                Dropout(0.3),
                Bidirectional(LSTM(50, activation='relu')),
                Dropout(0.3),
                Dense(1 if input_shape[-1] == 1 else input_shape[-1])
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            logger.info(f"Optimized LSTM built: {input_shape}")
            return model
        except Exception as e:
            logger.error(f"Error building LSTM: {str(e)}")
            return None

    def build_hybrid(input_shape):
        try:
            model = Sequential([
                Input(shape=input_shape),
                Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
                BatchNormalization(),
                Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
                MaxPooling1D(pool_size=2),
                Bidirectional(LSTM(100, activation='relu', return_sequences=False)),
                Dropout(0.3),
                Dense(1 if input_shape[-1] == 1 else input_shape[-1])
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            logger.info(f"Optimized Hybrid built: {input_shape}")
            return model
        except Exception as e:
            logger.error(f"Error building Hybrid: {str(e)}")
            return None

    # Save JSON results
    def save_training_results(results, file_path):
        try:
            with open(file_path, 'w') as f:
                json.dump(results, f)
            logger.info(f"Saved {file_path}")
        except Exception as e:
            logger.error(f"Failed to save {file_path}: {str(e)}")

    # Compute metrics
    def compute_metrics(true, pred):
        try:
            rmse = np.sqrt(mean_squared_error(true, pred))
            mae = mean_absolute_error(true, pred)
            r2 = r2_score(true, pred) if len(true) > 1 else 0
            return rmse, mae, r2
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            return 0, 0, 0

    # Calculate WQI with remarks
    def calculate_wqi(values_dict, params):
        try:
            normalized = []
            for param in params:
                values = values_dict[param]
                normalized_values = np.clip(values, 0, 1)  # Ensure [0, 1]
                logger.info(f"Using pre-normalized {param} for WQI: {normalized_values[:5]}")
                normalized.append(normalized_values)
            normalized = np.array(normalized)
            wqi = np.mean(normalized, axis=0, where=~np.isnan(normalized)) * 100
            wqi = np.clip(wqi, 0, 50)  # Ensure [0, 100]
            remarks = []
            for val in wqi:
                if np.isnan(val):
                    remarks.append("N/A")
                elif val <= 10:
                    remarks.append("Poor")
                elif val <= 20:
                    remarks.append("Fair")
                elif val <= 30:
                    remarks.append("Good")
                elif val <= 40:
                    remarks.append("Very Good")
                else:
                    remarks.append("Excellent")
            logger.info(f"WQI: {wqi[:5]}, Remarks: {remarks[:5]}")
            return wqi, remarks
        except Exception as e:
            logger.error(f"Error calculating WQI: {str(e)}")
            return np.zeros(len(values_dict[params[0]])), ["N/A"] * len(values_dict[params[0]])

    # Get available parameters
    try:
        available_params = sorted([
            col for col in bfar_df.select_dtypes(include=np.number).columns
            if col not in ['Date', 'Site', 'Year', 'Month', 'Weather Condition', 'Wind Direction']
            and bfar_df[col].notna().any()
        ])
        logger.info(f"Found {len(available_params)} parameters: {available_params}")
    except Exception as e:
        logger.error(f"Error identifying parameters: {str(e)}")
        st.error("No data loaded.")
        st.stop()

    if not available_params:
        st.error("No valid parameters available.")
        st.stop()

    colA, colB = st.columns([2, 4])

    with colA:
        col1, col2 = st.columns([10, 9])
        with col1:
            st.markdown(
                "<div class='custom-text-primary' style='margin-bottom: 0px; margin-top: 8px; "
                "font-size: 15px; text-align: justify;'>Prediction Configuration</div>",
                unsafe_allow_html=True)
            prediction_mode = st.radio(
                "Choose a mode for prediction:",
                ["Time Series Forecasting", "Individual Parameter"],
                index=0,
                key="prediction_mode",
                horizontal=False
            )
            selected_model = st.selectbox(
                "Select Model for Prediction:",
                ["CNN", "LSTM", "Hybrid CNN-LSTM"],
                key="pred_model"
            )

            if 'prediction_results' not in st.session_state:
                st.session_state.prediction_results = None
                st.session_state.prediction_params = {}
            if 'comparison_results' not in st.session_state:
                st.session_state.comparison_results = None

            sites = ['All Sites'] + sorted(bfar_df['Site'].astype(str).unique())

            # Prediction functions
            def prepare_prediction_data(data, param, window_size=7):
                try:
                    values = data[param].dropna().values.reshape(-1, 1)
                    if len(values) < window_size + 1:
                        logger.warning(f"Insufficient data for {param}: {len(values)} rows")
                        return np.array([])
                    logger.info(f"Pre-normalized prediction data for {param}: {values[:5].flatten()}")
                    X = []
                    for i in range(len(values) - window_size):
                        X.append(values[i:i + window_size])
                    return np.array(X)
                except Exception as e:
                    logger.error(f"Error preparing prediction data for {param}: {str(e)}")
                    return np.array([])

            def prepare_multivariate_prediction_data(data, params, window_size=7):
                try:
                    values = data[params].dropna().values
                    if len(values) < window_size + 1:
                        logger.warning(f"Insufficient multivariate data: {len(values)} rows")
                        return np.array([])
                    logger.info(f"Pre-normalized multivariate prediction data: {values[:5]}")
                    X = []
                    for i in range(len(values) - window_size):
                        X.append(values[i:i + window_size])
                    return np.array(X)
                except Exception as e:
                    logger.error(f"Error preparing multivariate prediction data: {str(e)}")
                    return np.array([])

            def predict_univariate(model, X, horizon):
                try:
                    predictions = []
                    current_input = X[-1].copy()
                    for _ in range(horizon):
                        pred = model.predict(current_input.reshape(1, current_input.shape[0], 1), verbose=0)
                        predictions.append(pred[0, 0])
                        current_input = np.roll(current_input, -1)
                        current_input[-1] = pred[0, 0]
                    predictions = np.array(predictions).flatten()
                    logger.info(f"Univariate predictions for {horizon} steps: {predictions[:5]}")
                    return predictions
                except Exception as e:
                    logger.error(f"Error in univariate prediction: {str(e)}")
                    return np.array([])

            def predict_multivariate(model, X, horizon, params):
                try:
                    predictions = []
                    current_input = X[-1].copy()
                    for _ in range(horizon):
                        pred = model.predict(current_input.reshape(1, current_input.shape[0], current_input.shape[1]), verbose=0)
                        predictions.append(pred[0])
                        current_input = np.roll(current_input, -1, axis=0)
                        current_input[-1] = pred[0]
                    predictions = np.array(predictions)
                    logger.info(f"Multivariate predictions: {[f'{param}: {predictions[:5, i]}' for i, param in enumerate(params)]}")
                    return predictions
                except Exception as e:
                    logger.error(f"Error in multivariate prediction: {str(e)}")
                    return np.array([])

            if prediction_mode == "Time Series Forecasting":
                selected_site = st.selectbox("Select Site:", sites, key="pred_site_ts")
                prediction_horizon = st.selectbox("Prediction Horizon:", ["1 Week", "2 Weeks", "1 Month", "3 Months",
                                                                         "6 Months", "9 Months", "1 Year"],
                                                  key="pred_horizon")
                if st.button("Train and Predict", key="train_predict_timeseries", type="primary"):
                    with col2:
                        with st.spinner("Training model, please wait..."):
                            st.session_state.prediction_params = {
                                "mode": "Time Series Forecasting",
                                "model": selected_model,
                                "site": selected_site,
                                "horizon": prediction_horizon
                            }
                            filtered_df = bfar_df.copy()
                            if selected_site != 'All Sites':
                                filtered_df = filtered_df[filtered_df['Site'] == selected_site]
                            filtered_df = filtered_df.sort_values('Date')

                            # Train model
                            os.makedirs('models', exist_ok=True)
                            os.makedirs('training_results', exist_ok=True)
                            X_train, y_train, X_val, y_val = prepare_multivariate_data(filtered_df, available_params)
                            if X_train is None:
                                st.error(f"Insufficient data for multivariate training at {selected_site}.")
                                st.stop()

                            model_key = selected_model.replace(' CNN-LSTM', '').lower()
                            model_builders = {'cnn': build_cnn, 'lstm': build_lstm, 'hybrid': build_hybrid}
                            model = model_builders[model_key]((X_train.shape[1], X_train.shape[2]))
                            if model is None:
                                st.error(f"Failed to build {selected_model}.")
                                st.stop()

                            callbacks = [
                                LossHistory(),
                                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
                            ]
                            model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val),
                                      callbacks=callbacks, verbose=0)
                            model.save(f"models/{model_key}_multivariate.keras")
                            logger.info(f"Saved {model_key}_multivariate.keras")

                            # Training results
                            y_pred = model.predict(X_val, verbose=0)
                            training_results = {
                                'epochs': len(callbacks[0].losses),
                                'loss': callbacks[0].losses,
                                'val_loss': callbacks[0].val_losses,
                                'actual': {param: y_val[:, i].tolist() for i, param in enumerate(available_params)},
                                'predicted': {param: y_pred[:, i].tolist() for i, param in enumerate(available_params)}
                            }
                            training_results_path = f"training_results/{model_key}_multivariate_training_results.json"
                            save_training_results(training_results, training_results_path)

                            # Predict future
                            horizon_days = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30, "3 Months": 90,
                                            "6 Months": 180, "9 Months": 270, "1 Year": 364}[prediction_horizon]
                            X_pred = prepare_multivariate_prediction_data(filtered_df, available_params)
                            if X_pred.shape[0] == 0:
                                st.error("Insufficient data for prediction.")
                                st.stop()
                            predictions = predict_multivariate(model, X_pred, horizon_days, available_params)
                            predictions_dict = {param: predictions[:, i] for i, param in enumerate(available_params)}
                            wqi, wqi_remarks = calculate_wqi(predictions_dict, available_params)
                            metrics = {}
                            for param in available_params:
                                validation_true = filtered_df[param].values[-horizon_days:] if len(
                                    filtered_df) >= horizon_days else predictions_dict[param]
                                validation_pred = predictions_dict[param][:len(validation_true)]
                                rmse, mae, r2 = compute_metrics(validation_true, validation_pred)
                                metrics[param] = {"rmse": rmse, "mae": mae, "r2": r2}

                            dates = pd.date_range(start=pd.Timestamp.today(), periods=horizon_days, freq='D')
                            st.session_state.prediction_results = {
                                "model": selected_model,
                                "site": selected_site,
                                "horizon": prediction_horizon,
                                "dates": dates,
                                "values": predictions_dict,
                                "wqi": wqi,
                                "wqi_remarks": wqi_remarks,
                                "rmse": {param: metrics[param]["rmse"] for param in metrics},
                                "mae": {param: metrics[param]["mae"] for param in metrics},
                                "r2": {param: metrics[param]["r2"] for param in metrics},
                                "epochs": training_results['epochs'],
                                "training_results": training_results
                            }
                            st.session_state.view = "Results"
            else:
                selected_param = st.selectbox("Select Parameter to Predict:", available_params, key="pred_param")
                selected_site = st.selectbox("Select Site:", sites, key="pred_site")
                prediction_horizon = st.selectbox("Prediction Horizon:", ["1 Week", "2 Weeks", "1 Month", "3 Months",
                                                                         "6 Months", "9 Months", "1 Year"],
                                                  key="pred_horizon")
                if st.button("Train and Predict", key="train_predict_individual", type="primary"):
                    with col2:
                        with st.spinner("Training model, please wait..."):
                            st.session_state.prediction_params = {
                                "mode": "Individual Parameter",
                                "model": selected_model,
                                "parameter": selected_param,
                                "site": selected_site,
                                "horizon": prediction_horizon
                            }
                            filtered_df = bfar_df.copy()
                            if selected_site != 'All Sites':
                                filtered_df = filtered_df[filtered_df['Site'] == selected_site]
                            filtered_df = filtered_df.sort_values('Date')

                            # Train model
                            os.makedirs('models', exist_ok=True)
                            os.makedirs('training_results', exist_ok=True)
                            X_train, y_train, X_val, y_val = prepare_univariate_data(filtered_df, selected_param)
                            if X_train is None:
                                st.error(f"Insufficient data for training {selected_param} at {selected_site}.")
                                st.stop()

                            model_key = selected_model.replace(' CNN-LSTM', '').lower()
                            model_builders = {'cnn': build_cnn, 'lstm': build_lstm, 'hybrid': build_hybrid}
                            model = model_builders[model_key]((X_train.shape[1], 1))
                            if model is None:
                                st.error(f"Failed to build {selected_model} for {selected_param}.")
                                st.stop()

                            callbacks = [
                                LossHistory(),
                                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
                            ]
                            model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val),
                                      callbacks=callbacks, verbose=0)
                            model.save(f"models/{model_key}_{selected_param}.keras")
                            logger.info(f"Saved {model_key}_{selected_param}.keras")

                            # Training results
                            y_pred = model.predict(X_val, verbose=0).flatten()
                            y_actual = y_val.flatten()
                            logger.info(f"Training results for {selected_param}: actual={y_actual[:5]}, predicted={y_pred[:5]}")
                            training_results = {
                                'epochs': len(callbacks[0].losses),
                                'loss': callbacks[0].losses,
                                'val_loss': callbacks[0].val_losses,
                                'actual': y_actual.tolist(),
                                'predicted': y_pred.tolist()
                            }
                            training_results_path = f"training_results/{model_key}_{selected_param}_training_results.json"
                            save_training_results(training_results, training_results_path)

                            # Predict future
                            horizon_days = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30, "3 Months": 90,
                                            "6 Months": 180, "9 Months": 270, "1 Year": 364}[prediction_horizon]
                            X_pred = prepare_prediction_data(filtered_df, selected_param)
                            if X_pred.shape[0] == 0:
                                st.error("Insufficient data for prediction.")
                                st.stop()
                            predictions = predict_univariate(model, X_pred, horizon_days)

                            # WQI
                            wqi_values = {selected_param: predictions}
                            for param in available_params:
                                if param != selected_param:
                                    recent_values = filtered_df[param].dropna().tail(horizon_days).values
                                    wqi_values[param] = recent_values if len(recent_values) == horizon_days else np.full(horizon_days, np.nan)
                                    if len(recent_values) != horizon_days:
                                        logger.warning(f"Insufficient historical data for {param}")
                            wqi, wqi_remarks = calculate_wqi(wqi_values, available_params)

                            validation_true = filtered_df[selected_param].values[-horizon_days:] if len(
                                filtered_df) >= horizon_days else predictions
                            validation_pred = predictions[:len(validation_true)]
                            rmse, mae, r2 = compute_metrics(validation_true, validation_pred)

                            dates = pd.date_range(start=pd.Timestamp.today(), periods=horizon_days, freq='D')
                            st.session_state.prediction_results = {
                                "model": selected_model,
                                "parameter": selected_param,
                                "site": selected_site,
                                "horizon": prediction_horizon,
                                "dates": dates,
                                "values": predictions,
                                "wqi": wqi,
                                "wqi_remarks": wqi_remarks,
                                "rmse": rmse,
                                "mae": mae,
                                "r2": r2,
                                "epochs": training_results['epochs'],
                                "training_results": training_results
                            }
                            st.session_state.view = "Results"

        with col2:
            st.markdown(
                "<div class='custom-text-primary' style='margin-bottom: 0px; margin-top: 8px; "
                "font-size: 15px; text-align: justify;'>Prediction Summary</div>",
                unsafe_allow_html=True)
            results = st.session_state.prediction_results
            if results:
                st.markdown(
                    f"<div class='custom-text-small'>Model Used:</div>"
                    f"<div class='custom-text-primary'>{results['model']}</div>",
                    unsafe_allow_html=True
                )
                if st.session_state.prediction_params["mode"] == "Individual Parameter":
                    st.markdown(
                        f"<div class='custom-text-small'>Target Parameter:</div>"
                        f"<div class='custom-text-primary'>{results['parameter']}</div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div class='custom-text-small'>Site:</div>"
                        f"<div class='custom-text-primary'>{results['site']}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='custom-text-small'>Site:</div>"
                        f"<div class='custom-text-primary'>{results['site']}</div>",
                        unsafe_allow_html=True
                    )
                st.markdown(
                    f"<div class='custom-text-small'>Time Frame:</div>"
                    f"<div class='custom-text-primary'>{results['horizon']}</div>",
                    unsafe_allow_html=True
                )

            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
            view_options = ["Results", "Evaluation", "Comparison"]
            if 'view' not in st.session_state:
                st.session_state.view = "Results"
            for option in view_options:
                is_selected = st.session_state.view == option
                st.button(
                    option,
                    key=f"view_{option.lower().replace(' ', '_')}",
                    on_click=lambda opt=option: st.session_state.update(view=opt),
                    type="primary" if is_selected else "secondary"
                )

    with colB:
        if st.session_state.prediction_results is None and st.session_state.view != "Comparison":
            st.info("Please configure and run training/prediction to view results.")
        else:
            if st.session_state.view == "Results":
                st.markdown(
                    "<div class='custom-text-primary' style='margin-top: 3px; margin-bottom: 5px; font-size: 20px;'>Prediction Results</div>",
                    unsafe_allow_html=True)
                if st.session_state.prediction_params["mode"] == "Individual Parameter":
                    df_pred = pd.DataFrame({
                        "Date": results["dates"],
                        results["parameter"]: results["values"]
                    })
                    st.markdown(f"**Predicted Values for {results['parameter']} ({results['site']})**")
                    st.dataframe(df_pred, use_container_width=True)

                    fig_pred = px.line(df_pred, x="Date", y=results["parameter"],
                                       title=f"Predicted {results['parameter']} ({results['site']})")
                    fig_pred.update_traces(line=dict(width=2, color='#004A99'))
                    fig_pred.update_layout(
                        showlegend=True,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=400,
                        title_font=dict(size=18, family='Montserrat' if font_base64 else 'sans-serif'),
                        title_x=0.03,
                        xaxis_title="Date",
                        yaxis_title=results["parameter"],
                        font=dict(family='Montserrat' if font_base64 else 'sans-serif')
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                else:
                    selected_params = st.multiselect(
                        "Select Parameters to Plot and Display:",
                        options=available_params,
                        default=available_params[:min(len(available_params), 3)],
                        key="pred_params_plot"
                    )
                    if selected_params:
                        df_pred = pd.DataFrame({"Date": results["dates"]})
                        for param in selected_params:
                            df_pred[param] = results["values"][param]
                        df_pred["Water Quality Index"] = results["wqi"]
                        df_pred["WQI Remarks"] = results["wqi_remarks"]
                        st.dataframe(df_pred[["Date"] + selected_params + ["Water Quality Index", "WQI Remarks"]],
                                     use_container_width=True)
                        melted_data = df_pred.melt(id_vars=["Date"], value_vars=selected_params,
                                                   var_name="Parameter", value_name="Value")
                        fig_pred = px.line(melted_data, x="Date", y="Value", color="Parameter",
                                           title=f"Predicted Parameters ({results['site']})")
                        fig_pred.update_traces(line=dict(width=2))
                        fig_pred.update_layout(
                            showlegend=True,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=400,
                            title_font=dict(size=18, family='Montserrat' if font_base64 else 'sans-serif'),
                            title_x=0.03,
                            xaxis_title="Date",
                            yaxis_title="Value",
                            font=dict(family='Montserrat' if font_base64 else 'sans-serif')
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)

                    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                    st.markdown(
                        "<div class='custom-text-primary' style='margin-top: 0px; margin-bottom: 8px; font-size: 20px;'>Water Quality Index</div>",
                        unsafe_allow_html=True)
                    df_wqi = pd.DataFrame({
                        "Date": results["dates"],
                        "Water Quality Index": results["wqi"],
                        "WQI Remarks": results["wqi_remarks"]
                    })
                    st.dataframe(df_wqi, use_container_width=True)
                    fig_wqi = px.line(df_wqi, x="Date", y="Water Quality Index",
                                      title=f"Water Quality Index ({results['site']})")
                    fig_wqi.update_traces(line=dict(width=2, color='#004A99'))
                    fig_wqi.update_layout(
                        showlegend=True,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=400,
                        title_font=dict(size=18, family='Montserrat' if font_base64 else 'sans-serif'),
                        title_x=0.03,
                        xaxis_title="Date",
                        yaxis_title="Water Quality Index",
                        font=dict(family='Montserrat' if font_base64 else 'sans-serif')
                    )
                    st.plotly_chart(fig_wqi, use_container_width=True)

                    # Recommendations based on WQI
                    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                    st.markdown(
                        "<div class='custom-text-primary' style='margin-top: 0px; margin-bottom: 8px; font-size: 20px;'>Water Quality Recommendations</div>",
                        unsafe_allow_html=True)

                    avg_wqi = np.mean(results["wqi"])
                    primary_remark = max(set(results["wqi_remarks"]), key=results["wqi_remarks"].count)
                    site_text = f" at {results['site']}" if results['site'] != 'All Sites' else ""
                    params_text = ", ".join(selected_params) if selected_params else "multiple parameters"
                    timeframe = results["horizon"]
                    model = results["model"]

                    recommendations = []
                    if primary_remark == "Poor":
                        recommendations.append(
                            f"The predicted Water Quality Index (WQI) for {params_text}{site_text} over {timeframe} "
                            f"using the {model} model indicates poor water quality (average WQI: {avg_wqi:.2f}). "
                            f"Immediate action is recommended, including water treatment, source contamination investigation, "
                            f"and monitoring of key parameters such as {params_text}. Consider restricting water use until quality improves."
                        )
                    elif primary_remark == "Fair":
                        recommendations.append(
                            f"The predicted Water Quality Index (WQI) for {params_text}{site_text} over {timeframe} "
                            f"using the {model} model suggests fair water quality (average WQI: {avg_wqi:.2f}). "
                            f"Regular monitoring of parameters like {params_text} is advised. Implement preventive measures "
                            f"such as reducing pollutant inputs and enhancing filtration systems to maintain or improve water quality."
                        )
                    elif primary_remark == "Good":
                        recommendations.append(
                            f"The predicted Water Quality Index (WQI) for {params_text}{site_text} over {timeframe} "
                            f"using the {model} model indicates good water quality (average WQI: {avg_wqi:.2f}). "
                            f"Continue regular monitoring of {params_text} to ensure sustained quality. Routine maintenance "
                            f"of water systems is recommended to prevent degradation."
                        )
                    elif primary_remark == "Very Good":
                        recommendations.append(
                            f"The predicted Water Quality Index (WQI) for {params_text}{site_text} over {timeframe} "
                            f"using the {model} model shows very good water quality (average WQI: {avg_wqi:.2f}). "
                            f"Maintain current water management practices and periodically check {params_text} to ensure stability."
                        )
                    elif primary_remark == "Excellent":
                        recommendations.append(
                            f"The predicted Water Quality Index (WQI) for {params_text}{site_text} over {timeframe} "
                            f"using the {model} model indicates excellent water quality (average WQI: {avg_wqi:.2f}). "
                            f"No immediate actions are required; continue monitoring {params_text} to sustain this high quality."
                        )
                    else:
                        recommendations.append(
                            f"The predicted Water Quality Index (WQI) for {params_text}{site_text} over {timeframe} "
                            f"using the {model} model could not be reliably assessed due to insufficient data or invalid WQI values. "
                            f"Ensure data completeness for {params_text} and verify model inputs."
                        )

                    # Disclaimer for long-term predictions
                    if timeframe in ["6 Months", "9 Months", "1 Year"]:
                        recommendations.append(
                            "**Disclaimer**: Predictions for longer timeframes such as 6 months, 9 months, or 1 year may have "
                            "reduced accuracy due to potential changes in environmental conditions, data trends, or model limitations. "
                            "Use these predictions as a guide and validate with ongoing monitoring."
                        )

                    for rec in recommendations:
                        st.markdown(f"- {rec}")


            elif st.session_state.view == "Evaluation":
                st.markdown(
                    "<div class='custom-text-primary' style='margin-top: 3px; margin-bottom: 5px; font-size: 20px;'>Model Evaluation</div>",
                    unsafe_allow_html=True)
                training_results = results.get('training_results')
                if not training_results:
                    st.error("Training results not available.")
                    st.stop()

                if st.session_state.prediction_params["mode"] == "Individual Parameter":
                    selected_eval_params = [results["parameter"]]
                else:
                    selected_eval_params = st.multiselect(
                        "Select Parameters for Evaluation:",
                        options=available_params,
                        key="eval_params"
                    )

                if selected_eval_params:
                    for param in selected_eval_params:
                        if param in training_results['actual']:
                            actual_pred_df = pd.DataFrame({
                                "Actual": training_results['actual'][param],
                                "Predicted": training_results['predicted'][param]
                            })
                            st.markdown(f"**Actual vs Predicted Values for {param}**")
                            st.dataframe(actual_pred_df, use_container_width=True)

                            fig_actual_pred = go.Figure()
                            fig_actual_pred.add_trace(go.Scatter(
                                x=actual_pred_df["Actual"], y=actual_pred_df["Predicted"],
                                mode='markers', name='Data',
                                marker=dict(color='#004A99', size=8)
                            ))
                            slope, intercept, _, _, _ = linregress(actual_pred_df["Actual"],
                                                                   actual_pred_df["Predicted"])
                            trend_x = np.array([actual_pred_df["Actual"].min(), actual_pred_df["Actual"].max()])
                            trend_y = slope * trend_x + intercept
                            fig_actual_pred.add_trace(go.Scatter(
                                x=trend_x, y=trend_y,
                                mode='lines', name='Trend Line',
                                line=dict(color='#FF5733', width=2)
                            ))
                            fig_actual_pred.update_layout(
                                title=f"Actual vs Predicted {param} ({results['model']})",
                                xaxis_title="Actual",
                                yaxis_title="Predicted",
                                showlegend=True,
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                height=400,
                                title_font=dict(size=18, family='Montserrat' if font_base64 else 'sans-serif'),
                                title_x=0.03,
                                font=dict(family='Montserrat' if font_base64 else 'sans-serif')
                            )
                            st.plotly_chart(fig_actual_pred, use_container_width=True)

                    if st.session_state.prediction_params["mode"] == "Individual Parameter":
                        metrics_data = [
                            {"Metric": "RMSE", "Value": results["rmse"]},
                            {"Metric": "MAE", "Value": results["mae"]}
                        ]
                        r2_hidden = not (0 <= results["r2"] <= 1)
                        if not r2_hidden:
                            metrics_data.append({"Metric": "R²", "Value": results["r2"]})
                        eval_df = pd.DataFrame(metrics_data)
                        fig_eval = px.bar(eval_df, x="Metric", y="Value",
                                          title=f"Evaluation Metrics for {results['parameter']} ({results['site']})",
                                          color_discrete_sequence=['#004A99'])
                        fig_eval.update_layout(
                            showlegend=False,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=400,
                            title_font=dict(size=18, family='Montserrat' if font_base64 else 'sans-serif'),
                            title_x=0.03,
                            xaxis_title="Metric",
                            yaxis_title="Value",
                            font=dict(family='Montserrat' if font_base64 else 'sans-serif')
                        )
                        st.plotly_chart(fig_eval, use_container_width=True)
                        if r2_hidden:
                            st.info(f"R² for {results['parameter']} cannot be displayed due to insufficient data.")
                    else:
                        eval_data = []
                        r2_hidden_params = []
                        for param in selected_eval_params:
                            param_metrics = {
                                "Parameter": param,
                                "RMSE": results["rmse"][param],
                                "MAE": results["mae"][param]
                            }
                            if 0 <= results["r2"][param] <= 1:
                                param_metrics["R²"] = results["r2"][param]
                            else:
                                r2_hidden_params.append(param)
                            eval_data.append(param_metrics)
                        eval_df = pd.DataFrame(eval_data)
                        value_vars = ["RMSE", "MAE"] + (["R²"] if any("R²" in d for d in eval_data) else [])
                        melted_eval = eval_df.melt(id_vars="Parameter", value_vars=value_vars,
                                                   var_name="Metric", value_name="Value")
                        fig_eval = px.bar(melted_eval, x="Metric", y="Value", color="Parameter",
                                          title=f"Evaluation Metrics ({results['site']})",
                                          barmode="group")
                        fig_eval.update_layout(
                            showlegend=True,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=400,
                            title_font=dict(size=18, family='Montserrat' if font_base64 else 'sans-serif'),
                            title_x=0.03,
                            xaxis_title="Metric",
                            yaxis_title="Value",
                            font=dict(family='Montserrat' if font_base64 else 'sans-serif')
                        )
                        st.plotly_chart(fig_eval, use_container_width=True)
                        for param in r2_hidden_params:
                            st.info(f"R² for {param} cannot be displayed due to insufficient data.")

                st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='custom-text-primary' style='margin-top: 0px; margin-bottom: 8px; font-size: 20px;'>Training Evaluation</div>",
                    unsafe_allow_html=True)
                st.markdown(f"**Epochs Trained:** {results['epochs']}")
                loss_df = pd.DataFrame({
                    "Epoch": range(1, training_results['epochs'] + 1),
                    "Training Loss": training_results['loss'],
                    "Validation Loss": training_results['val_loss']
                })
                melted_loss = loss_df.melt(id_vars="Epoch", value_vars=["Training Loss", "Validation Loss"],
                                           var_name="Loss Type", value_name="Loss")
                fig_loss = px.line(melted_loss, x="Epoch", y="Loss", color="Loss Type",
                                   title=f"Training and Validation Loss ({results['model']})")
                fig_loss.update_traces(line=dict(width=2))
                fig_loss.update_layout(
                    showlegend=True,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=400,
                    title_font=dict(size=18, family='Montserrat' if font_base64 else 'sans-serif'),
                    title_x=0.03,
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    font=dict(family='Montserrat' if font_base64 else 'sans-serif')
                )
                st.plotly_chart(fig_loss, use_container_width=True)


            # ==== Comparison Section (Modified) ====

            elif st.session_state.view == "Comparison":

                st.markdown(

                    "<div class='custom-text-primary' style='margin-top: 3px; margin-bottom: 5px; font-size: 20px;'>Model Comparison</div>",

                    unsafe_allow_html=True)

                if not st.session_state.prediction_params:
                    st.error("Please run a prediction first to compare models.")

                    st.stop()

                if st.button("Run Model Comparison", key="run_comparison", type="primary"):

                    with st.spinner("Running model comparison, this may take a while, please wait..."):

                        comparison_results = []

                        model_builders = {'cnn': build_cnn, 'lstm': build_lstm, 'hybrid': build_hybrid}

                        filtered_df = bfar_df.copy()

                        if st.session_state.prediction_params["site"] != 'All Sites':
                            filtered_df = filtered_df[filtered_df['Site'] == st.session_state.prediction_params["site"]]

                        filtered_df = filtered_df.sort_values('Date')

                        horizon_days = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30, "3 Months": 90,

                                        "6 Months": 180, "9 Months": 270, "1 Year": 364}[

                            st.session_state.prediction_params["horizon"]]

                        if st.session_state.prediction_params["mode"] == "Individual Parameter":

                            selected_param = st.session_state.prediction_params["parameter"]

                            X_train, y_train, X_val, y_val = prepare_univariate_data(filtered_df, selected_param)

                            if X_train is None:
                                st.error(f"Insufficient data for {selected_param}.")

                                st.stop()

                            for model_name, builder in model_builders.items():

                                model = builder((X_train.shape[1], 1))

                                if model is None:
                                    logger.error(f"Failed to build {model_name}.")

                                    continue

                                callbacks = [

                                    LossHistory(),

                                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),

                                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

                                ]

                                model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val),

                                          callbacks=callbacks, verbose=0)

                                y_pred = model.predict(X_val, verbose=0).flatten()

                                y_actual = y_val.flatten()

                                rmse, mae, r2 = compute_metrics(y_actual, y_pred)

                                comparison_results.append({

                                    "Model": model_name.upper(),

                                    "RMSE": rmse,

                                    "MAE": mae,

                                    "R²": r2 if 0 <= r2 <= 1 else None  # Hide R² if negative

                                })

                        else:

                            X_train, y_train, X_val, y_val = prepare_multivariate_data(filtered_df, available_params)

                            if X_train is None:
                                st.error(f"Insufficient multivariate data.")

                                st.stop()

                            for model_name, builder in model_builders.items():

                                model = builder((X_train.shape[1], X_train.shape[2]))

                                if model is None:
                                    logger.error(f"Failed to build {model_name}.")

                                    continue

                                callbacks = [

                                    LossHistory(),

                                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),

                                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

                                ]

                                model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val),

                                          callbacks=callbacks, verbose=0)

                                y_pred = model.predict(X_val, verbose=0)

                                metrics = {}

                                for i, param in enumerate(available_params):
                                    rmse, mae, r2 = compute_metrics(y_val[:, i], y_pred[:, i])

                                    metrics[param] = {"rmse": rmse, "mae": mae, "r2": r2}

                                avg_rmse = np.mean([m["rmse"] for m in metrics.values()])

                                avg_mae = np.mean([m["mae"] for m in metrics.values()])

                                avg_r2 = np.mean([m["r2"] for m in metrics.values()])

                                comparison_results.append({

                                    "Model": model_name.upper(),

                                    "RMSE": avg_rmse,

                                    "MAE": avg_mae,

                                    "R²": avg_r2 if 0 <= avg_r2 <= 1 else None  # Hide R² if negative

                                })

                        st.session_state.comparison_results = comparison_results

                if st.session_state.comparison_results:

                    comp_df = pd.DataFrame(st.session_state.comparison_results)

                    # Identify models with hidden R²

                    hidden_r2_models = comp_df[comp_df['R²'].isna()]['Model'].tolist()

                    # Remove R² column if all values are None

                    display_df = comp_df.drop(columns=['R²']) if comp_df['R²'].isna().all() else comp_df

                    st.dataframe(display_df, use_container_width=True)

                    value_vars = ["RMSE", "MAE"] + (["R²"] if not comp_df['R²'].isna().all() else [])

                    melted_comp = comp_df.melt(id_vars="Model", value_vars=value_vars,

                                               var_name="Metric", value_name="Value")

                    fig_comp = px.bar(melted_comp, x="Metric", y="Value", color="Model",

                                      title="Model Comparison",

                                      barmode="group")

                    fig_comp.update_layout(

                        showlegend=True,

                        plot_bgcolor='white',

                        paper_bgcolor='white',

                        height=400,

                        title_font=dict(size=18, family='Montserrat' if font_base64 else 'sans-serif'),

                        title_x=0.03,

                        xaxis_title="Metric",

                        yaxis_title="Value",

                        font=dict(family='Montserrat' if font_base64 else 'sans-serif')

                    )

                    st.plotly_chart(fig_comp, use_container_width=True)

                    if hidden_r2_models:
                        st.info(
                            f"R² for {', '.join(hidden_r2_models)} cannot be displayed due to insufficient data")

                else:

                    st.info("Click 'Run Model Comparison' to compare models.")


# ==== About ====
with tab_info:
    set_active_tab("About")
    st.markdown(
        "<div class='custom-text-primary' style='font-size: 22px; text-align: justify;'>About the Dataset</div>",
        unsafe_allow_html=True)

    bfar_raw_df = pd.DataFrame()
    philvolcs_raw_df = pd.DataFrame()
    try:
        bfar_raw_df = pd.read_parquet('datasets/BFAR.parquet', engine='pyarrow')
    except FileNotFoundError:
        st.error("BFAR.parquet not found.")
    except Exception as e:
        st.error(f"Error loading BFAR.parquet: {e}")

    try:
        philvolcs_raw_df = pd.read_parquet('datasets/PHIVOLCS.parquet', engine='pyarrow')
    except FileNotFoundError:
        st.error("PHIVOLCS.parquet not found.")
    except Exception as e:
        st.error(f"Error loading PHIVOLCS.parquet: {e}")

    col1, col2, col3 = st.columns([10, 0.5, 10])
    with col1:
        colA, colB, colC = st.columns([3, 0.05, 10])
        with colA:
            try:
                with open("images/BFAR.png", "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode()
                st.markdown(
                    f'<img src="data:image/png;base64,{img_base64}" width="100" alt="BFAR Logo">',
                    unsafe_allow_html=True
                )
            except FileNotFoundError:
                st.warning("BFAR.png not found in the images folder. Please ensure the file exists in the repository.")
            except Exception as e:
                st.error(f"Error loading BFAR.png: {e}")
        with colC:
            st.markdown(""" 
                <div class='custom-text-primary' style='margin-top: 23px; font-size: 30px; text-align: left; color: #023AA8;'>Water Quality Dataset</div>
                <div class='custom-text-secondary' style='margin-bottom: 27px; color: #4E94DC; font-size: 15px; text-align: left;'>(BFAR)</div>
            """, unsafe_allow_html=True)
        if not bfar_raw_df.empty:
            st.markdown(f"**Shape:** {bfar_raw_df.shape[0]} rows × {bfar_raw_df.shape[1]} columns")
            missing = bfar_raw_df.isnull().sum()
            missing_filtered = missing[missing > 0]
            if not missing_filtered.empty:
                st.markdown("**Top 3 Parameters with Missing Values:**")
                for param, count in missing_filtered.sort_values(ascending=False).head(3).items():
                    st.markdown(f"- **{param}**: {count} missing values")
            else:
                st.markdown("No missing values in the Water Quality dataset.")
            st.markdown(f"**Total Missing Cells:** {missing.sum()} cells")
            with st.expander("Water Quality Dataset Preview (First 20 rows)"):
                st.dataframe(bfar_raw_df.head(20), height=250)
        else:
            st.warning("Water Quality data (BFAR.parquet) not loaded.")

    with col3:
        colA_ph, colB_ph, colC_ph = st.columns([3, 0.05, 10])
        with colA_ph:
            try:
                with open("images/PHIVOLCS.png", "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode()
                st.markdown(
                    f'<img src="data:image/png;base64,{img_base64}" width="100" alt="PHIVOLCS Logo">',
                    unsafe_allow_html=True
                )
            except FileNotFoundError:
                st.warning(
                    "PHIVOLCS.png not found in the images folder. Please ensure the file exists in the repository.")
            except Exception as e:
                st.error(f"Error loading PHIVOLCS.png: {e}")
        with colC_ph:
            st.markdown(""" 
                <div class='custom-text-primary' style='margin-top: 18px; font-size: 30px; text-align: left; color: #222831;'>PHIVOLCS Dataset</div>
                <div class='custom-text-secondary' style='margin-bottom: 27px;color: #43B5C3; font-size: 18px; text-align: left;'>(Volcanic Activity)</div>
            """, unsafe_allow_html=True)
        if not philvolcs_raw_df.empty:
            st.markdown(f"**Shape:** {philvolcs_raw_df.shape[0]} rows × {philvolcs_raw_df.shape[1]} columns")
            missing = philvolcs_raw_df.isnull().sum()
            missing_filtered = missing[missing > 0]
            if not missing_filtered.empty:
                st.markdown("**Top 3 Parameters with Missing Values:**")
                for param, count in missing_filtered.sort_values(ascending=False).head(3).items():
                    st.markdown(f"- **{param}**: {count} missing values")
            else:
                st.markdown("No missing values in the PHIVOLCS dataset.")
            st.markdown(f"**Total Missing Cells:** {missing.sum()} cells")
            with st.expander("PHIVOLCS Dataset Preview (First 20 rows)"):
                st.dataframe(philvolcs_raw_df.head(20), height=250)
        else:
            st.warning("PHIVOLCS data (PHIVOLCS.parquet) not loaded.")

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='custom-text-primary' style='margin-bottom:15px; font-size: 23px; text-align: justify;'>About Taal Lake</div>",
        unsafe_allow_html=True)
    col_taal1, col_taal2, col_taal3 = st.columns([5, 0.5, 10])
    with col_taal1:
        try:
            with open("images/Taal-volcano-map.jpg", "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()
            st.markdown(
                f'<img src="data:image/jpeg;base64,{img_base64}" alt="Taal Volcano Map" style="width: 100%;">',
                unsafe_allow_html=True
            )
            st.caption("Image from: ShelterBox USA")
        except FileNotFoundError:
            st.warning(
                "Taal-volcano-map.jpg not found in the images folder. Please ensure the file exists in the repository.")
        except Exception as e:
            st.error(f"Error loading Taal-volcano-map.jpg: {e}")
    with col_taal3:
        st.markdown(f"<div style='text-align: justify; font-size: 19px; color: #393E46;'>{taal_info}</div>",
                    unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    st.markdown(""" 
        <div class='custom-text-primary' style='font-size: 23px; text-align: justify;'>About the Developers</div>
        <div class='custom-text-primary' style='font-size: 15px; text-align: justify; margin-bottom:25px; '>Group 2 - BS CPE 3-1</div>
    """, unsafe_allow_html=True)

    developers = [
        {"name": "BULASO, DAVE PATRICK I.", "phone": "+63XXXXXXXXXX", "email": "main.davepatrick.bulaso@cvsu.edu.ph",
         "img": "1x1/dave.jpg"},
        {"name": "DENNA, ALEXA YVONNE V.", "phone": "+639184719122", "email": "main.alexayvonne.denna@cvsu.edu.ph",
         "img": "1x1/alexa.jpg"},
        {"name": "EJERCITADO, JOHN MARTIN P.", "phone": "+639262333664",
         "email": "main.johnmartin.ejercitado@cvsu.edu.ph", "img": "1x1/martin.png"},
        {"name": "ESPINO, GIAN JERICHO Z.", "phone": "+639108733830", "email": "main.gianjericho.espino@cvsu.edu.ph",
         "img": "1x1/gian.jpg"},
        {"name": "INCIONG, HARLEY EVANGEL J.", "phone": "+639516120316",
         "email": "main.harleyevangel.inciong@cvsu.edu.ph", "img": "1x1/harley.jpg"},
    ]

    dev_col1, dev_col2, dev_col3 = st.columns(3)
    cols = [dev_col1, dev_col2, dev_col3]
    for i, dev in enumerate(developers):
        with cols[i % 3]:
            try:
                with open(dev["img"], "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode()
                st.markdown(
                    f'<img src="data:image/jpeg;base64,{img_base64}" width="100" alt="{dev["name"]} Photo">',
                    unsafe_allow_html=True
                )
            except FileNotFoundError:
                st.warning(
                    f"{dev['img']} not found in the 1x1 folder. Please ensure the file exists in the repository.")
            except Exception as e:
                st.error(f"Error loading {dev['img']}: {e}")
            st.markdown(f"**{dev['name']}**<br>Phone: {dev['phone']}<br>Email: {dev['email']}", unsafe_allow_html=True)

# ==== FOOTER ====
footer_img = "images/footer.png"
try:
    with open(footer_img, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f'<img src="data:image/png;base64,{img_base64}" alt="Footer" class="full-width-footer">',
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.warning(f"{footer_img} not found in the images folder. Please ensure the file exists in the repository.")
except Exception as e:
    st.error(f"Error loading {footer_img}: {e}")
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
