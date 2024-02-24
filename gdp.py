import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np


#Set page layout here
st.set_page_config(layout="wide")

# Hide Streamlit style
hide_st_style = '''
                <style>
                #MainMenu {visibility : hidden;}
                footer {visibility : hidder;}
                header {visibility :hidden;}
                <style>
                '''
st.markdown(hide_st_style, unsafe_allow_html =True)


#function to loaddata
@st.cache_resource
def loadgdpgva():
    #reset loading dataframe for correct upload
    df = pd.DataFrame()
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
            hovertext[-1].append(f'Category: {yy} <br>{timescale}: {xx.date() if timescale=="Quarter" else xx}<br>Cell Value: {value:.2f}')
    return hovertext


#processing for texttemplete of heatmap
def process_texttemplete(timescale):

    if timescale == "Quarter":
        texttemplate = ""
    else:
        texttemplate = "%{text:.1f}"

    return texttemplate


#processing dataframe based on choosen timescale
def process_df_choosen_timescale(df,timescale, feature):

    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].apply(lambda x: x.year)
    df["Month"] = df["Date"].apply(lambda x: x.month)
    df["FYear"] = [int(x)+1 if int(y) >=4 else int(x) for x,y in zip(df["Year"], df["Month"])]

    if timescale == "Quarter":
        if feature == "Absolute":
            pivot_df = df.pivot_table(index='Description', columns='Date', values='Value')
        if feature == "Percent":
            dftemp = df.groupby(["FYear", "Month"]).agg({"Value": "sum"}).reset_index()
            dftemp = df.merge(dftemp, on =["FYear","Month"], how = 'left')
            dftemp["Value"] = (dftemp["Value_x"]/dftemp["Value_y"])*100
            pivot_df = dftemp.pivot_table(index='Description', columns='Date', values='Value')
        if feature == "Growth":
            pivot_df = df.pivot_table(index='Description', columns='Date', values='Value')
            pivot_df = ((pivot_df - pivot_df.shift(5, axis =1))/pivot_df.shift(5, axis =1))*100
           
    if timescale == "FYear":
        if feature == "Absolute":

            df = df.groupby(["FYear", "Description"]).agg({"Value": "sum"}).reset_index()
            pivot_df = df.pivot(index='Description', columns='FYear', values='Value')

        if feature == "Percent":
            dftemp1 = df.groupby(["FYear"]).agg({"Value": "sum"}).reset_index()
            dftemp2 = df.groupby(["FYear", "Description"]).agg({"Value": "sum"}).reset_index()
            dftemp = dftemp2.merge(dftemp1, on =["FYear"], how ='left')
            dftemp["Value"] = (dftemp["Value_x"]/dftemp["Value_y"])*100
            pivot_df = dftemp.pivot_table(index='Description', columns='FYear', values='Value')
        if feature == "Growth":
            pivot_df = df.pivot_table(index='Description', columns='FYear', values='Value')
            pivot_df = ((pivot_df - pivot_df.shift(1, axis =1))/pivot_df.shift(1, axis =1))*100
            
        
    #sorting the dataframe 
    pivot_df = pivot_df.sort_values(pivot_df.columns[-1], ascending = True)
    return pivot_df


#configuring the data for heatmap
def create_heatmap_data(df, hovertext, texttemplate):

    # Flatten the DataFrame values to a 1D array for calculation
    z_values = df.values.flatten()
    Q1 = np.percentile(z_values, 25)
    Q3 = np.percentile(z_values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR


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
                  showscale=False,
                  zmin=lower_bound,  # Set zmin to lowerbound
                  zmax=upper_bound,   # Set zmax to upperbound
                  colorbar=dict(
                  tickcolor ="black",
                  tickwidth =2,
                  # tickvals = [1,2,3,4]
                  # ticktext = ticktext,
                  # dtick=1, tickmode="array"),
                        ),
                )]

    return data

#configuring heatmap
def configuring_heatmap(fig):
    #Formatting the x-axis
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

    return fig



def create_bar_chart_data(coltotaldf, timescale, dimension):
    bar = go.Bar(
        x=coltotaldf[timescale], 
        y=coltotaldf[dimension],
        text=coltotaldf[dimension],
        textposition='outside',
        texttemplate='%{y:.1f}',  # Format text to one decimal place
        orientation='v',  # Horizontal bar chart
        marker=dict(
        line=dict(color='Black', width=2)
        )  # Sets the border color and width
)
    return bar


def processing_currency(dimension, curreny, timescale, feature, df):

    #filtering aggregrated GDP & GVA values from the heatmap
    # filter_desc = dimension.split(" ")[0]
    df = df[df["Type"] == dimension]
    # df = df[(df["Description"] != filter_desc)]


    #Processing values for Indian Rupees 
    if curreny == "Rupees":
        #dropping unnecessary columns
        df = df.drop(columns = ["Type","USD"])
        #processing dataframe based on choosen timescale and feature
        df = process_df_choosen_timescale(df,timescale,feature)


    #Processing for values for us dollars 
    if (curreny == "USDollars"):
        if dimension in ["GDP Current","GVA Current"]:
            df["Value"] = round((df["Value"]/df["USD"])*1000,2)
            df = df.drop(columns = ["Type", "USD"])
            #processing dataframe based on choosen timescale and feature
            df = process_df_choosen_timescale(df,timescale,feature)
        else:
            st.write("Please Choose Nominal Dimension for Displaying USD Values")
            df = pd.DataFrame()

    return df

#Processing chart heading based on user choice of menues
def chart_heading(dimension,curreny,timescale,feature):

    if feature == "Absolute":
        if curreny == "Rupees":
            title_text =  dimension+" - " +timescale+" Trends"+" (Rs Lakh Cr)"
        if curreny == "USDollars":
            title_text =  dimension+" - " +timescale+" Trends"+" ($ Billion)"
    if feature == "Percent":
            title_text =  dimension+" - " +timescale+" Trends"+" (Percent of Total)"
    if feature == "Growth":
            title_text =  dimension+" - " +timescale+" Trends"+" (Percentage Growth)"

    return title_text


#-----------MAIN PROGRAM STARTS-------------------

#load data
df = loadgdpgva()


#choose a dimension
dimension = st.sidebar.selectbox('Select a Dimension', ["GDP Current", "GDP Constant", "GVA Current","GVA Constant"])

#choose a currency
curreny = st.sidebar.selectbox('Select a Currency', ["Rupees","USDollars"])

#choose a time scale
timescale = st.sidebar.selectbox('Select a Timescale', ["Quarter", "FYear"])

#choose a feature
feature = st.sidebar.selectbox('Select a Feature', ["Absolute","Percent","Growth"])

#processing dataframe with seleted menues 
pivot_df = processing_currency(dimension, curreny, timescale, feature, df)

#filtering aggregrated GDP & GVA values from the heatmap
filter_desc = dimension.split(" ")[0]

filter_desc_df = pivot_df[(pivot_df.index == filter_desc)]

st.write(filter_desc_df)

# pivot_df = pivot_df[~(pivot_df.index == filter_desc)]


if pivot_df.shape[0] != 0:

    #processing hovertext of heatmap
    hovertext = process_hovertext(pivot_df, timescale)

    #processing texttemplete of heatmap
    texttemplate = process_texttemplete(timescale)

    #creating heatmap
    heatmap_data = create_heatmap_data(pivot_df,hovertext, texttemplate)
    fig1 = go.Figure(data = heatmap_data)
    #configuring heatmap
    fig1 = configuring_heatmap(fig1)

    #processing chart for total of all columns 
    coltotaldf = pivot_df.sum(axis=0).round(1).reset_index()
    coltotaldf.columns =[timescale, dimension]
    bar_data = create_bar_chart_data(coltotaldf, timescale, dimension)
    fig2 = go.Figure(data=bar_data)


    # Create a subplot layout with two rows and one column
    combined_fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0,  # Adjust spacing as needed
        shared_xaxes=False,  # Set to True if the x-axes should be aligned
        row_heights=[0.8, 0.2]  # First row is 80% of the height, second row is 20%
    )

    # Add each trace from your first figure to the first row of the subplot
    for trace in fig1.data:
        combined_fig.add_trace(trace, row=1, col=1)

    # Add each trace from your second figure to the second row of the subplot
    for trace in fig2.data:
        combined_fig.add_trace(trace, row=2, col=1)

    title_text = chart_heading(dimension,curreny,timescale,feature)
        
    # Update layout for the subplot
    combined_fig.update_layout(
        title_text = title_text,
        title_x = 0.07,
        title_y = 0.9,
        width=1200,  # Adjust width as needed
        height=640,  # Adjust height as needed to accommodate stacked layout
        title_font=dict(size=24, family="Arial, sans-serif", color="RebeccaPurple"),
    )


    combined_fig.update_layout(
        shapes=[
            # Rectangle border for the first subplot
            dict(
                type="rect",
                xref="paper", yref="paper",
                x0=0, y0=0.2,  # Adjust these values based on the subplot's position
                x1=1, y1=1,
                line=dict(color="Black", width=2),
            ),
            # Rectangle border for the second subplot
            dict(
                type="rect",
                xref="paper", yref="paper",
                x0=0, y0=0,  # Adjust these values based on the subplot's position
                x1=1, y1=0.2,
                line=dict(color="Black", width=2),
            )
        ]
    )

    combined_fig.update_xaxes(showticklabels=False, row=1, col=1)
    combined_fig.update_yaxes(showgrid=False, row=2, col=1)  # Removes horizontal grid lines
    combined_fig.update_yaxes(title_text="", row=2, col=1)   # Removes y-axis label

    #Making the y-axis of the chart start from the point more than Zero
    min_value = coltotaldf[dimension].min()  # Find the minimum value in the column totals
    start_y = min_value * 0.9  # Calculate 90% of the minimum value
    end_y = coltotaldf[dimension].max()*1.2 #set the maximum value of y-axis as 120% of the max bar
    combined_fig.update_yaxes(range=[start_y, end_y], row=2, col=1)

    # Update x-axis and y-axis titles if needed
    # combined_fig.update_xaxes(title_text="X-axis Title Here", row=1, col=1)
    # combined_fig.update_yaxes(title_text="Y-axis Title for Fig1", row=1, col=1)
    # combined_fig.update_xaxes(title_text="X-axis Title Here", row=2, col=1)
    # combined_fig.update_yaxes(title_text="Y-axis Title for Fig2", row=2, col=1)

    # Display the combined figure in Streamlit
    st.plotly_chart(combined_fig, use_container_width=True)





