#importing libraries
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
from collections import OrderedDict
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
import plotly
import pandas as pd
import plotly.figure_factory as ff
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
from datetime import datetime
import datetime as dt 
import calendar
import time

from PIL import Image

from dateutil import relativedelta

import re

from collections import defaultdict

from dateutil.relativedelta import relativedelta

import io
import msoffcrypto

import pickle
from pathlib import Path
import streamlit_authenticator as stauth

import yaml
from yaml.loader import SafeLoader

from deta import Deta

st.set_page_config(layout="wide")


df = pd.read_csv("2022_12_22_Indian_GDP_GVA_Comb.csv")


Type = list(set(df["Type"]))

feature = st.sidebar.selectbox('Select a Feature', Type)

filter_desc = feature.split(" ")[0]

df = df[df["Type"] == feature]

df = df[(df["Description"] != filter_desc)]


data = [go.Heatmap(
			      z = df["Value"],
			      x = df["Date"],
			      y = df["Description"],
			      xgap = 1,
			      ygap = 1,
			      hoverinfo ='text',
			      text = df["Value"],
			      colorscale='Picnic',
			      texttemplate="%{text:.1f}",
		# 	      reversescale=True,
			      colorbar=dict(
			      tickcolor ="black",
			      tickwidth =2,
			      tickvals = [1,2,3,4]
			      # ticktext = ticktext,
			      # dtick=1, tickmode="array"),
					    ),
				)]

fig = go.Figure(data=data)


# Formatting the x-axis
fig.update_xaxes(
    title_text="Date",  # X-axis title
    tickangle=-45,  # Rotate ticks (e.g., dates) for better readability
    title_font=dict(size=14),  # Font size for x-axis title
    tickfont=dict(size=12),  # Font size for x-axis ticks
)

# Formatting the y-axis
fig.update_yaxes(
    title_text="Description",  # Y-axis title
    title_standoff=25,  # Distance of title from axis
    title_font=dict(size=14),  # Font size for y-axis title
    tickfont=dict(size=12),  # Font size for y-axis ticks
)

fig.update_layout(
    width=1200,  # Adjust width as needed
    height=600,  # Adjust height as needed
    title_text="Heatmap Example",
    margin=dict(l=20, r=20, t=50, b=20)  # Adjust margins (left, right, top, bottom)
)

col1, col2 = st.columns([1, 3])  # Adjust the ratio as needed

# with col1:
#     # Place any sidebar or control elements here

with col1:
	st.plotly_chart(fig)




