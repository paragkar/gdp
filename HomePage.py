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


df = pd.read_csv("2022_12_22_Indian_GDP_GVA_Comb.csv")


Type = list(set(df["Type"]))

feature = st.sidebar.selectbox('Select a Feature', Type)

df = df[df["Type"] == feature]


data = [go.Heatmap(
			      z = df["Value"],
			      x = df["Date"],
			      y = df["Description"],
			      # xgap = xgap_dict[Band],
			      # ygap = 1,
			      # hoverinfo ='text',
			      text = df["Value"],
			      colorscale='Picnic',
		# 	      reversescale=True,
			      # colorbar=dict(
		# 	      tickcolor ="black",
		# 	      tickwidth =1,
			      # tickvals = tickvals,
			      # ticktext = ticktext,
			      # dtick=1, tickmode="array"),
				# 	    ),
				)]

fig = go.Figure(data=data)

fig.show()

