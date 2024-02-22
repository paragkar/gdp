import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Set Streamlit page layout
st.set_page_config(layout="wide")

# Function to load the data
@st.cache
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

# Main program starts
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

# Create heatmap data
data = go.Heatmap(
    z=pivot_df.values,
    x=pivot_df.columns,
    y=pivot_df.index,
    hoverinfo='text',
    text=hovertext,
    colorscale='Hot',
    reversescale=True
)

# Create a figure and add heatmap
fig = make_subplots(rows=1, cols=1)
fig.add_trace(data)

# Update layout
fig.update_layout(
    title=f"{dimension} Analysis",
    height=600,
)

st.plotly_chart(fig, use_container_width=True)
