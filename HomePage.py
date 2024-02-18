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


#Set page layout here
st.set_page_config(layout="wide")


hide_st_style = '''
				<style>
				#MainMenu {visibility : hidden;}
				footer {visibility : hidder;}
				header {visibility :hidden;}
				<style>
				'''
st.markdown(hide_st_style, unsafe_allow_html =True)



#--------Functions for loading File Starts---------------------#

@st.cache_resource
def loadgdpgva():

	df = pd.read_csv("2022_12_22_Indian_GDP_GVA_Comb.csv")

	return df


#--------Functions for loading File End---------------------#

#This fuction processes the hovertext
@st.cache_resource
def htext(df, timescale):  
	hovertext = []
	for yi, yy in enumerate(df.index):
		hovertext.append([])
		for xi, xx in enumerate(df.columns):
			# if exptab_dict[Band]==1: #1 means that the expiry table in the excel sheet has been set and working 
			# 	expiry = round(ef.values[yi][xi],2)
			# else:
			# 	expiry = "NA"
			# try:
			#     auction_year = round(ayear.loc[yy,round(xx-xaxisadj_dict[Band],3)])
			# except:
			#     auction_year ="NA"
				
			value = df.values[yi][xi]
			# operatorold = of.values[yi][xi]
			# bandwidth = bandf.values[yi][xi]
			if timescale == "Quarter":
				xx = xx.date()
			else:
				pass
			hovertext[-1].append(
							 timescale+' End: {}\
				              <br>Value: Rs {} Lakh Cr'
					    
					     .format(
					     	xx,
						    round(value,2)
						    )
					     	)
	return hovertext


def figdata(df, hovertext, texttemplate):

	data = [go.Heatmap(
			      z = df.values,
			      x = df.columns,
			      y = df.index,
			      xgap = 1,
			      ygap = 1,
			      hoverinfo ='text',
			      text = df.values,
			      hovertext=hovertext,  # Applying custom hover text
			      colorscale='Hot',
			      texttemplate=texttemplate,
			      reversescale=True,
			      colorbar=dict(
			      tickcolor ="black",
			      tickwidth =2,
			      # tickvals = [1,2,3,4]
			      # ticktext = ticktext,
			      # dtick=1, tickmode="array"),
					    ),
				)]

	return data


#function for preparing the column total chart 
def coltotalchart(coltotaldf, xcolumn, ycolumn):
	bar = alt.Chart(coltotaldf).mark_bar().encode(
	y = alt.Y(ycolumn+':Q', axis=alt.Axis(labels=False)),
	x = alt.X(xcolumn+':O', axis=alt.Axis(labels=False)),
	color = alt.Color(xcolumn+':N', legend=None))

	text = bar.mark_text(size = 10, dx=0, dy=-7, color = 'white').encode(text=ycolumn+':Q')
	
	coltotalchart = (bar + text).properties(width=1120, height =150)
	coltotalchart = coltotalchart.configure_title(fontSize = 20, font ='Arial', anchor = 'middle', color ='black')
	return coltotalchart



#main program starts


#load data
df = loadgdpgva()

#extract dimensions
Type = list(set(df["Type"]))

#choose a dimension
dimension = st.sidebar.selectbox('Select a Dimension', Type)


#filtering aggregrated GDP & GVA values from the heatmap
filter_desc = dimension.split(" ")[0]
df = df[df["Type"] == dimension]
df = df[(df["Description"] != filter_desc)]


#dropping unnecessary columns
df = df.drop(columns = ["Type","USD"])

#choose a time scale
timescale = st.sidebar.selectbox('Select a timescale', ["Quarter", "FYear"])


#processing dataframe based on choosen timescale
if timescale == "Quarter":
	pivot_df = df.pivot(index='Description', columns='Date', values='Value')
if timescale == "FYear":
	df["Date"] = pd.to_datetime(df["Date"])
	df["Year"] = df["Date"].apply(lambda x: x.year)
	df["Month"] = df["Date"].apply(lambda x: x.month)
	df["FYear"] = [int(x)+1 if int(y) >=4 else int(x) for x,y in zip(df["Year"], df["Month"])]
	df = df.groupby(["FYear", "Description"]).agg({"Value": "sum"}).reset_index()
	pivot_df = df.pivot(index='Description', columns='FYear', values='Value')


#sorting dataframe
pivot_df = pivot_df.sort_values(pivot_df.columns[-1], ascending = True)

#processing hovertext of heatmap
hovertext = htext(pivot_df, timescale)

#processing for texttemplete of heatmap
if timescale == "Quarter":
	texttemplate = ""
else:
	texttemplate = "%{text:.1f}"


coltotaldf = pivot_df.sum(axis=0).reset_index()


coltotaldf.columns =[timescale, dimension]


coltotalchart = coltotalchart(coltotaldf, timescale, dimension)


data = figdata(pivot_df,hovertext, texttemplate)
fig = go.Figure(data=data)


# Formatting the x-axis
fig.update_xaxes(
    tickangle=0,  # Rotate ticks (e.g., dates) for better readability
    title_font=dict(size=12, family="Arial"),  # Font size for x-axis title
    tickfont=dict(size=12),  # Font size for x-axis ticks
)

# Formatting the y-axis
fig.update_yaxes(
    title_standoff=10,  # Distance of title from axis
    title_font=dict(size=12, family = "Arial"),  # Font size for y-axis title
    tickfont=dict(size=12, family="Arial, Black"),  # Font size for y-axis ticks
)

fig.update_layout(
    width=1200,  # Adjust width as needed
    height=450,  # Adjust height as needed
    title_text=dimension +" (Unit : Rs Lakh Cr)",
    margin=dict(l=20, r=20, t=50, b=20)  # Adjust margins (left, right, top, bottom)
)

#Drawning a black border around the heatmap chart 
fig.update_xaxes(fixedrange=True,showline=True,linewidth=2,linecolor='black', mirror=True)
fig.update_yaxes(fixedrange=True,showline=True, linewidth=2, linecolor='black', mirror=True)

fig.update_yaxes(tickfont_family="Arial")
fig.update_xaxes(tickfont_family="Arial")

hoverlabel_bgcolor = "#000000" #subdued black

fig.update_traces(hoverlabel=dict(bgcolor=hoverlabel_bgcolor,font=dict(size=12, color='white')))
xdtickangle= 0
xdtickval=1

# Create a container
with st.container():
	st.plotly_chart(fig, use_container_width=True)
	st.altair_chart(coltotalchart, use_container_width=True)



