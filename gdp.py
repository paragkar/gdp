# Importing necessary libraries
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Set Streamlit page layout as the very first command
st.set_page_config(layout="wide")

# Function to load the GDP & GVA data
@st.cache_data
def load_gdp_gva():
    df = pd.read_csv("2022_12_22_Indian_GDP_GVA_Comb.csv")
    return df

# Function to process the hover text for the heatmap
@st.cache_data
def process_hovertext(df, timescale):  
    hovertext = []
    for yi, yy in enumerate(df.index):
        hovertext.append([])
        for xi, xx in enumerate(df.columns):
            value = df.values[yi][xi]
            hovertext[-1].append(f'{timescale} End: {xx}<br>Value: Rs {value:.2f} Lakh Cr')
    return hovertext

# Function to prepare data for the heatmap
def create_heatmap_data(df, hovertext):
    data = go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        xgap=1,
        ygap=1,
        hoverinfo='text',
        text=hovertext,  # Applying custom hover text
        colorscale='Hot',
        reversescale=True
    )
    return data

# Function to prepare data for the bar chart
def create_bar_chart_data(coltotaldf, timescale, dimension):
    bar = go.Bar(
        y=coltotaldf[timescale], 
        x=coltotaldf[dimension],
        text=coltotaldf[dimension],
        textposition='auto',
        orientation='h',  # Horizontal bar chart
    )
    return bar

# Hide Streamlit style
hide_st_style = '''
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                '''
st.markdown(hide_st_style, unsafe_allow_html=True)

# Main program starts

# Load data
df = load_gdp_gva()

# Choose a dimension
dimension = st.sidebar.selectbox('Select a Dimension', list(set(df["Type"])))

# Filter aggregated GDP & GVA values from the heatmap
df = df[df["Type"] == dimension]
df = df.drop(columns=["Type", "USD"])

# Choose a timescale
timescale = st.sidebar.selectbox('Select a timescale', ["Quarter", "FYear"])

# Process dataframe based on chosen timescale
pivot_df = df.pivot(index='Description', columns='Date', values='Value')

# Sort dataframe
pivot_df = pivot_df.sort_values(pivot_df.columns[-1], ascending=True)

# Process hovertext of heatmap
hovertext = process_hovertext(pivot_df, timescale)

# Create subplot layout with two rows and one column
fig = make_subplots(rows=2, cols=1, 
                    vertical_spacing=0.15,  # Adjust spacing as needed
                    subplot_titles=(f"{dimension} Heatmap", f"{dimension} Column Totals"))

# Add heatmap to the first row of the subplot
heatmap_data = create_heatmap_data(pivot_df, hovertext)
fig.add_trace(heatmap_data, row=1, col=1)

# Process and add bar chart data to the second row of the subplot
coltotaldf = pivot_df.sum(axis=0).reset_index()
coltotaldf.columns = [timescale, dimension]
bar_chart_data = create_bar_chart_data(coltotaldf, timescale, dimension)
fig.add_trace(bar_chart_data, row=2, col=1)

# Update layout for subplot
fig.update_layout(
    width=1200,  # Adjust width as needed
    height=800,  # Adjust height as needed to accommodate stacked layout
    title_text=f"{dimension} Analysis (Unit: Rs Lakh Cr)",
    margin=dict(l=20, r=20, t=50, b=20)  # Adjust margins (left, right, top, bottom)
)

# Format x-axis and y-axis of the bar chart
fig['layout']['yaxis2'].update(title=f'{timescale}', automargin=True)
fig['layout']['xaxis2'].update(title=f'{dimension} (Rs Lakh Cr)', automargin=True)

# Display the figure
st.plotly_chart(fig, use_container_width=True)
