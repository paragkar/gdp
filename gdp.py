import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

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
def process_texttemplete(timescale, selected_cols):

    if timescale == "Quarter":
        if len(selected_cols) > 25: #No of columns to be selected beyond which text will not display
            texttemplate = ""
        else:
            texttemplate = "%{text:.1f}"

    if timescale == "FYear":
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
            #The dataframe below is multipled by 2 to ensure to take into the effect of aggregrated number in cal
            dftemp["Value"] = (dftemp["Value_x"]/dftemp["Value_y"])*100*2
            pivot_df = dftemp.pivot_table(index='Description', columns='Date', values='Value')
        if feature == "Growth":
            pivot_df = df.pivot_table(index='Description', columns='Date', values='Value')
            pivot_df = ((pivot_df - pivot_df.shift(4, axis =1))/pivot_df.shift(4, axis =1))*100
           
    if timescale == "FYear":
        if feature == "Absolute":
            df = df.groupby(["FYear", "Description"]).agg({"Value": "sum"}).reset_index()
            pivot_df = df.pivot(index='Description', columns='FYear', values='Value')

        if feature == "Percent":
            dftemp1 = df.groupby(["FYear"]).agg({"Value": "sum"}).reset_index()
            dftemp2 = df.groupby(["FYear", "Description"]).agg({"Value": "sum"}).reset_index()
            dftemp = dftemp2.merge(dftemp1, on =["FYear"], how ='left')
            #The dataframe below is multipled by 2 to ensure to take into the effect of aggregrated number in cal
            dftemp["Value"] = (dftemp["Value_x"]/dftemp["Value_y"])*100*2
            pivot_df = dftemp.pivot_table(index='Description', columns='FYear', values='Value')
        if feature == "Growth":
            pivot_df = df.pivot_table(index='Description', columns='FYear', values='Value')
            pivot_df = ((pivot_df - pivot_df.shift(1, axis =1))/pivot_df.shift(1, axis =1))*100

    # pivot_df = pivot_df.sort_values(pivot_df.columns[-1], ascending = True)
            
    return pivot_df

# def sorting_dataframe(pivot_df,feature,dimension):
#     #sorting the dataframe for heatmap display
#     try:
#         if (feature != "Growth"):
#             pivot_df = pivot_df.sort_values(pivot_df.columns[-1], ascending = True)
#         if (feature == "Growth") & ((dimension == "GDP Constant") | (dimension == "GDP Current")):
#             pivot_df.index = pd.Categorical(pivot_df.index, categories=lst_for_sorting_pivot_df1, ordered=True)
#             pivot_df = pivot_df.sort_index(ascending=False)
#             pivot_df.index = pivot_df.index.astype('object')
#         if (feature == "Growth") & ((dimension == "GVA Constant") | (dimension == "GVA Current")):
#             pivot_df.index = pd.Categorical(pivot_df.index, categories=lst_for_sorting_pivot_df2, ordered=True)
#             pivot_df = pivot_df.sort_index(ascending=False)
#             pivot_df.index = pivot_df.index.astype('object')
#     except:
#         pass

#     return pivot_df


#configuring the data for heatmap
def create_heatmap_data(df, hovertext, texttemplate):

    #sort values based on the latest number
    df = df.sort_values(df.columns[-1], ascending = True)

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
        ),  # Sets the border color and width
        )
    return bar


def processing_currency(dimension, currency, timescale, feature, df):

    #filtering aggregrated GDP & GVA values from the heatmap
    filter_desc = dimension.split(" ")[0]
    df = df[df["Type"] == dimension]
    # df = df[(df["Description"] != filter_desc)]
    if filter_desc == "GVA":
        df = df[(df["Description"] != "Net Taxes")]


    #Processing values for Indian Rupees 
    if currency == "Rupees":
        #dropping unnecessary columns
        df = df.drop(columns = ["Type","USD"])
        #processing dataframe based on choosen timescale and feature
        df = process_df_choosen_timescale(df,timescale,feature)


    #Processing for values for US dollars 
    if (currency == "USDollars"):
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
def chart_heading(dimension,currency,timescale,feature):

    # Ensure all inputs are converted to strings before concatenation
    dimension = str(dimension)
    currency = str(currency)
    timescale = str(timescale)
    feature = str(feature)

    if feature == "Absolute":
        if currency == "Rupees":
            title_text =  dimension+" - " +timescale+" Trends"+" (Rs Lakh Cr)"
        if currency == "USDollars":
            title_text =  dimension+" - " +timescale+" Trends"+" ($ Billion)"
    if feature == "Percent":
            title_text =  dimension+" - " +timescale+" Trends"+" (Percent of Total)"
    if feature == "Growth":
            title_text =  dimension+" - " +timescale+" Trends"+" (Percentage Growth)"

    return title_text

#Check for proper combination of dimesion and current before proceeding 
def checkdimcurrcomb(dimension, currency):

    if (dimension in ["GDP Constant", "GDP Current", "GVA Constant","GVA Current"]) and (currency == "Rupees"):
        Flag = True 
    elif (dimension in ["GDP Current", "GVA Current"]) and (currency == "USDollars"):
        Flag = True 
    else:
        Flag = False

    return Flag

def plotingheatmap(pivot_df, dimension,timescale,currency,feature):

    filter_desc = dimension.split(" ")[0]
    #Extract the bar chart datframe first from the combined dataframe
    total_df = pivot_df[~(pivot_df.index != filter_desc)]
    total_df = total_df.replace(0, np.nan).dropna(axis=1)

    #filtering aggregrated GDP & GVA values from the heatmap
    pivot_df = pivot_df[(pivot_df.index != filter_desc)]
    pivot_df = pivot_df.replace(0,np.nan).dropna(axis=1)


    if pivot_df.shape[0] != 0:

        #Processing Slider in case timescale chosen is Quarter
        if timescale == "Quarter":
            selected_min, selected_max = createslider1(pivot_df)
            selected_cols = [x for x in pivot_df.columns if (x <= selected_max) & (x >= selected_min)]
            pivot_df = pivot_df[selected_cols]
            total_df = total_df[selected_cols]
        else:
            selected_cols = pivot_df.columns

        #processing hovertext of heatmap
        hovertext = process_hovertext(pivot_df, timescale)

        #processing texttemplete of heatmap
        texttemplate = process_texttemplete(timescale, selected_cols)

        #creating heatmap
        heatmap_data = create_heatmap_data(pivot_df,hovertext, texttemplate)
        fig1 = go.Figure(data = heatmap_data)
        #configuring heatmap
        fig1 = configuring_heatmap(fig1)


        #processing chart for total of all columns 
        total_df = total_df.T.reset_index()
        total_df.columns =[timescale, dimension]
        bar_data = create_bar_chart_data(total_df, timescale, dimension)
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
        # Add only if the the feature is not Pecent as the total_df data is irrelevent for this situation
        if feature != "Percent":
            for trace in fig2.data:
                combined_fig.add_trace(trace, row=2, col=1)

        #processing title text
        title_text = chart_heading(dimension,currency,timescale,feature)
            
        # Update layout for the subplot
        combined_fig.update_layout(
            title_text = title_text,
            title_x = 0.07,
            title_y = 0.9,
            width=1200,  # Adjust width as needed
            height=640,  # Adjust height as needed to accommodate stacked layout
            title_font=dict(size=24, family="Arial, sans-serif", color="RebeccaPurple"),
        )


        #Adding border in both the charts
        #Do not draw border in the chart in the feature is "Percent"
        if feature == "Percent":
            x1 = 0
            y1 = 0
        else:
            x1 = 1
            y1 = 0.2

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
                    x1=x1, y1=y1,
                    line=dict(color="Black", width=2),
                )
            ]
        )


        combined_fig.update_xaxes(showticklabels=False, row=1, col=1)
        combined_fig.update_yaxes(showgrid=False, row=2, col=1)  # Removes horizontal grid lines
        combined_fig.update_yaxes(title_text="", row=2, col=1)   # Removes y-axis label

        
        #Setting of dtick of the final bar chart
        if timescale == 'FYear':
            combined_fig.update_xaxes(tickvals=pivot_df.columns.unique(), ticktext=[str(year) for year in pivot_df.columns.unique()], row=2, col=1)
        if timescale == 'Quarter':
            years = sorted(set([x for x in pivot_df.columns.year]))
            combined_fig.update_xaxes(tickvals=years, ticktext=[str(year) for year in years], row=2, col=1)


        #Making the y-axis of the chart start from the point more than Zero
        min_value = total_df[dimension].min()  # Find the minimum value in the column totals
        start_y = min_value * 0.85  # Calculate 90% of the minimum value
        end_y = total_df[dimension].max()*1.35 #set the maximum value of y-axis as 120% of the max bar
        combined_fig.update_yaxes(range=[start_y, end_y], row=2, col=1)

        # Update x-axis and y-axis titles if needed
        # combined_fig.update_xaxes(title_text="X-axis Title Here", row=1, col=1)
        # combined_fig.update_yaxes(title_text="Y-axis Title for Fig1", row=1, col=1)
        # combined_fig.update_xaxes(title_text="X-axis Title Here", row=2, col=1)
        # combined_fig.update_yaxes(title_text="Y-axis Title for Fig2", row=2, col=1)

        # Display the combined figure in Streamlit

    return    st.plotly_chart(combined_fig, use_container_width=True)

#Create a slider for scatter normal and heatmap
def createslider1(pivot_df):
    slider_min_date = pivot_df.columns.min().to_pydatetime()
    slider_max_date = pivot_df.columns.max().to_pydatetime()

    # Create a list of quarter-end dates within the range
    date_range = pd.date_range(start=slider_min_date, end=slider_max_date, freq='QE').to_pydatetime().tolist()

    # Create a mapping of indices to dates
    index_to_date = {i: date for i, date in enumerate(date_range)}

    # Find index for the default point 8 quarters back
    default_index = max(0, len(date_range) - 20)

    # Use an integer slider
    selected_indices = st.slider(
        "Select Quarters to View",
        min_value=0,
        max_value=len(date_range) - 1,
        value=(default_index, len(date_range) - 1),
        format="%d"
    )
    # Map indices back to dates
    selected_min, selected_max = index_to_date[selected_indices[0]], index_to_date[selected_indices[1]]

    return selected_min, selected_max

def plotingscatter(pivot_df, dimension,timescale,currency,feature):

    pivot_df = pivot_df.dropna(axis=1)

    #Change of the dimension "imports" from negative to positive
    if dimension in ["GDP Constant", "GDP Current"]:
        pivot_df.loc["Imports",:] = pivot_df.loc["Imports",:].apply(lambda x: x*-1)

    #Processing Slider in case timescale chosen is Quarter
    if timescale == "Quarter":
        selected_min, selected_max = createslider1(pivot_df)
        selected_cols = [x for x in pivot_df.columns if (x <= selected_max) & (x >= selected_min)]
        pivot_df = pivot_df[selected_cols]
        st.write("Selected No of Quarters: ", len(selected_cols),", Start Date: ",selected_cols[0].date(), ", End Date : ", selected_cols[-1].date())
    else:
        selected_cols = pivot_df.columns

    # Determine the number of rows and columns for the subplot grid
    num_dimensions = len(pivot_df.index)
    cols = 3  # Set to 3 columns as requested
    rows = -(-num_dimensions // cols)  # Calculate rows needed, rounding up

    # Generate scatter plots with trendlines for each dimension
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, vertical_spacing=0.05, horizontal_spacing=0.05)

    if timescale == "Quarter":
        x_data = pivot_df.columns
    elif timescale == "FYear":
        pivot_df.columns = [pd.Timestamp(year=x, month=3, day=31) for x in pivot_df.columns]
        x_data = pivot_df.columns

    # Iterate over each dimension to create a scatter plot
    for i, dim in enumerate(pivot_df.index, start=1):
        y_data = pivot_df.loc[dim]
        
        # Determine the position of the current plot
        row = (i - 1) // cols + 1
        col = (i - 1) % cols + 1
        
        # Generate timestamps for linear regression
        timestamps = [x.timestamp() for x in x_data]

        # Add scatter plot for the current dimension
        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers+lines', line=dict(dash='dot'), name=f'{dim} Trend'), row=row, col=col)

        fig.update_yaxes(row=row, 
                        col=col, 
                        title_standoff=7)
                        
        # Add trendline using a linear fit
        trend = np.polyfit(timestamps, y_data, 1)
        trendline = np.poly1d(trend)(timestamps)
        fig.add_trace(go.Scatter(x=x_data, y=trendline, mode='lines', name=''), row=row, col=col)


    # Draw a rectangular box around the whole subplot area
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0, y0=-0.042,
        x1=1, y1=1.02,
        line=dict(color="Black", width=2),
    )


    title_text = chart_heading(dimension,currency,timescale,feature)+" - Normal"

    # Update layout to accommodate the new grid structure and enhance readability
    fig.update_layout(
        height=250 * rows,  # Adjust the height based on the number of rows
        width=900,  # Set a fixed width or adjust as necessary
        title_text= title_text,
        showlegend=False
    )

    # Adjust axis titles and format for each subplot
    for i in range(1, num_dimensions + 1):
        row = (i - 1) // cols + 1
        col = (i - 1) % cols + 1
        fig.update_yaxes(title_text=pivot_df.index[i - 1], row=row, col=col, tickformat = '.1f')

    # Display the figure in Streamlit

    return st.plotly_chart(fig, use_container_width=True)


#create slider for the scatter function for Scatter Forecast
def createslider2(extended_x_data):
    # Ensure all datetime data is in Python datetime format, not pandas Timestamp
    extended_x_data = [pd.to_datetime(date).to_pydatetime() for date in extended_x_data]

    # Determine the minimum and maximum dates for the slider
    slider_min_date = min(extended_x_data)
    slider_max_date = max(extended_x_data)

    # Create and display the slider in Streamlit, ensuring it returns datetime objects
    selected_range = st.slider(
        "Select Date Range:",
        min_value=slider_min_date,
        max_value=slider_max_date,
        value=(slider_min_date, slider_max_date),
        format="YYYY"  # Adjust date format as needed
    )

    return selected_range

def plotingscatterforecast(pivot_df, dimension, timescale, currency, feature, forecast_period):

    pivot_df = pivot_df.dropna(axis=1)

    # Sidebar input for user-defined bias percentage with default 0%
    bias_percentage = st.sidebar.number_input('Enter Trendline Bias Percentage:', value=0.0, step=1.0, format='%f')
    
    # Convert negative values for "imports" dimension to positive, if necessary
    if dimension in ["GDP Constant", "GDP Current"]:
        pivot_df.loc["Imports", :] *= -1

    # Initialize subplot structure
    num_dimensions = len(pivot_df.index)
    cols = 3
    rows = -(-num_dimensions // cols)
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, vertical_spacing=0.05, horizontal_spacing=0.05)

    # Logic for quarters
    if timescale == "Quarter":
        original_x_data = pivot_df.columns
        last_date = original_x_data[-1]
        # Forecast future dates for quarters
        forecast_dates = [last_date + relativedelta(months=3 * k) for k in range(1, forecast_period + 1)]
        extended_x_data = list(original_x_data) + forecast_dates
        selected_min, selected_max = createslider2(extended_x_data)
        #This block ensures that users are restricted to selecting min value NOT greater than last value
        if selected_min > last_date:
            selected_min = last_date
        else:
            pass
        selected_cols = [x for x in extended_x_data if (x <= selected_max) & (x >= selected_min)]
        st.write("Selected Quarters For Forecast: ", len(selected_cols),", Start Date: ",selected_cols[0].date(), ", End Date : ", selected_cols[-1].date())
    # Logic for fiscal years
    elif timescale == "FYear":
        pivot_df.columns = [pd.Timestamp(year=x, month=3, day=31) for x in pivot_df.columns]
        original_x_data = pivot_df.columns
        last_year = original_x_data[-1].year
        # Forecast future dates for fiscal years
        forecast_dates = [pd.Timestamp(year=last_year + k, month=3, day=31) for k in range(1, forecast_period + 1)]
        extended_x_data = list(original_x_data) + forecast_dates
        selected_min, selected_max = createslider2(extended_x_data)
        #This block ensures that users are restricted to selecting min value NOT greater than last value
        if selected_min > original_x_data[-1]:
            selected_min = original_x_data[-1]
        else:
            pass
        selected_cols = [x for x in extended_x_data if (x <= selected_max) & (x >= selected_min)]
        st.write("Selected Years For Forecast: ", len(selected_cols),", Start Date: ",selected_cols[0].date(), ", End Date : ", selected_cols[-1].date())

    # Plotting logic
    for i, dim in enumerate(pivot_df.index, start=1):
        row, col = (i - 1) // cols + 1, (i - 1) % cols + 1
        # Common plotting logic for both quarters and fiscal years
        historical_x_data = [x for x in selected_cols if x in original_x_data]
        timestamps = np.array([pd.Timestamp(x).timestamp() for x in historical_x_data])
        y_data = pivot_df.loc[dim, historical_x_data].dropna()

        if len(timestamps) != len(y_data):
            raise ValueError("The lengths of timestamps and y_data do not match.")

        # trend = np.polyfit(timestamps, y_data, 1)
        # trend_poly = np.poly1d(trend)

       # Calculate the trend without bias
        trend = np.polyfit(timestamps, y_data, 1)
        slope, intercept = trend
        
        # Check the sign of the gradient (slope) to decide how to apply the bias
        if slope >= 0:
            # For positive slope, increase the intercept
            adjusted_intercept = intercept * (1 + bias_percentage / 100.0)
        else:
            # For negative slope, decrease the intercept to shift the line up
            adjusted_intercept = intercept * (1 - bias_percentage / 100.0)
        
        # Create the adjusted trend polynomial
        trend_poly = np.poly1d([slope, adjusted_intercept])

        # Plot historical data
        fig.add_trace(go.Scatter(x=historical_x_data, y=y_data, mode='markers+lines', name=f'{dim} Data'), row=row, col=col)

        # Apply the trend to display data for visualization
        all_timestamps = np.array([pd.Timestamp(x).timestamp() for x in selected_cols])
        all_y_data = trend_poly(all_timestamps)

        fig.add_trace(go.Scatter(x=selected_cols, y=all_y_data, mode='lines', name=f'{dim} Trend', line=dict(dash='dot')), row=row, col=col)

        fig.update_yaxes(title_text=dim, title_standoff=7, row=row, col=col, tickformat='.1f')

    # Finalizing plot with layout and rectangle box
    fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, y0=-0.042, x1=1, y1=1.02, line=dict(color="Black", width=2))
    title_text = chart_heading(dimension, currency, timescale, feature)+" - Forecast"
    fig.update_layout(height=250 * rows, width=900, title_text=title_text, showlegend=False)

    return st.plotly_chart(fig, use_container_width=True)


#-----------MAIN PROGRAM STARTS-------------------

#load data
df = loadgdpgva()

# Set the title at the top of the sidebar
st.sidebar.title("Indian GDP Trends and Forecasting Model")

#choose a dimension
dimension = st.sidebar.selectbox('Select a Dimension', ["GDP Current", "GDP Constant", "GVA Current","GVA Constant"])
#choose a currency
currency = st.sidebar.selectbox('Select a Currency', ["Rupees","USDollars"])
#choose a time scale
timescale = st.sidebar.selectbox('Select a Timescale', ["Quarter", "FYear"])
#choose a feature
feature = st.sidebar.selectbox('Select a Feature', ["Absolute","Percent","Growth"])

#processing dataframe with seleted menues 
pivot_df = processing_currency(dimension, currency, timescale, feature, df)
# #sorting dataframe for visulization in heatmap
# pivot_df = sorting_dataframe(pivot_df,feature, dimension)

#Choose a plot type
plot_type = st.sidebar.selectbox('Select a Plot Type', ["Heatmap", "Scatter"])


#This flag is to check that processing takes place for a correct dimension and currency combination
#For example GDP Constant is not paired with USDollars
Flag = checkdimcurrcomb(dimension, currency)


#Picking up a plot type, note that Heatmap charts do not have forcast facility
if plot_type == "Heatmap" and Flag:
    plotingheatmap(pivot_df, dimension,timescale,currency,feature)

if plot_type == "Scatter" and Flag:
    # Add a radio button in the sidebar to select between 'Normal' and 'Forecast'
    mode_selection = st.sidebar.radio("Select Mode:", ['Normal', 'Forecast'])

    if mode_selection == "Normal":
        plotingscatter(pivot_df, dimension,timescale,currency,feature)

    if mode_selection == "Forecast":
        if timescale == "Quarter":
            # Place an integer input box in the sidebar to get the number of quarters.
            forecast_period = st.sidebar.number_input('Enter Quarter Nos to Forecast', min_value=0, max_value=400, value=4, step=1, format='%d')
        if timescale == "FYear":
            # Place an integer input box in the sidebar to get the number of years.
            forecast_period = st.sidebar.number_input('Enter FYear Nos to Forecast', min_value=0, max_value=100, value=4, step=1, format='%d')

        plotingscatterforecast(pivot_df, dimension, timescale, currency, feature, forecast_period)




