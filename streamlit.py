import os
import pickle
from pandas.core.api import CategoricalDtype
import streamlit as st
from streamlit_option_menu import option_menu

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

theme_plotly=None

# Set page configuration
st.set_page_config(
    page_title="DS FinPro - HelloWorld.ipynb team",
    layout="wide",
    page_icon="📊"
)

# model
global_model = pickle.load(open('Predict Global Air Pollution modelling.sav','rb'))

#scaler
scaler = pickle.load(open('scaler.sav','rb'))

# accuracy
accuracy = pickle.load(open('accuracy_value.sav','rb'))

img = Image.open('logo logo.jpg')
st.sidebar.image(img)

# sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'Air Pollution Conditions and Prediction',
         [
             'Visualization',
             'AQI Category Prediction'
         ],
        menu_icon='cast',
        icons=['kanban', 'activity'],
        default_index=0
)

########################################    VISUALIZATION PAGE  ######################################################################
########################################    VISUALIZATION PAGE  ######################################################################
# Visualization Page
if selected == 'Visualization':
    
    # membuka file pickle untuk visualization page
    data_glob = pickle.load(open('df_global.pickle','rb'))

     # Sidebar visualization page
    st.sidebar.header('Visualization')
    st.sidebar.write('This is a page for Exploratory Data Analysis (EDA) by visualization.')

    st.sidebar.header('Data Filters') 

    # Create for AQI Category
    AQI_Category=st.sidebar.multiselect('Pick AQI Category', data_glob['AQI Category'].unique())
    if not AQI_Category:
      data_global=data_glob.copy()
    else:
      data_global=data_glob[data_glob['AQI Category'].isin(AQI_Category)]

    # Create for Country
    countryy=st.sidebar.multiselect('Pick the Country', data_global['Country'].unique())
    if not countryy:
      filter_df=data_global.copy()
    else:
      filter_df=data_global[data_global['Country'].isin(countryy)]
    
    #-------------------------------------------------------------------

    # tab1 dan tab2
    tab1, tab2 = st.tabs(["About", "Dashboard"])

    #%%%%%%%%%%%%%%%%%%%%%%%%%%  ABOUT TAB  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #---------------------------------------------------------------    
    # tab 1 ( About )   
    #---------------------------------------------------------------   

    # with tab1:
    with tab1:
        # page title
        st.title('About')

        st.markdown(
        """Global air pollution is a global environmental problem that impacts worldwide air quality, human health, ecosystems, and climate. 
        Global warming, extreme climate change, and increased risks of respiratory and public health diseases are the result of emissions of 
        greenhouse gases and pollutant particles from various sources, such as industry, transportation, and biomass burning.
        """)
        st.markdown(
        """With this visualization and prediction dashboard, it is hoped that the entire community, especially the government, can know the 
        air quality conditions of cities in the world and can serve as a reference or help in overcoming air pollution.
        """)
        st.markdown(
        """There are two pages in this application, namely the visualization page and the prediction page. On the visualization page there 
        are two tabs, namely the 'About' tab and the 'Dashboard' tab. The 'About' tab (this tab) contains a brief explanation of the problem and 
        a description of the data used. The 'Dashboard' tab displays several data visualizations from the Global Air Pollution dataset from which 
        we can draw insights. Furthermore, the prediction page contains the application of a classification machine learning model which can be 
        used by inputting some AQI value data so that you can immediately see the prediction results for the AQI category.
        """)
        st.markdown(
        """The following is a table that displays the contents of the data set used, consisting of 12 columns.
        """)

        # menampilkan data
        st.subheader('Dataset')
        intro_text = """
        This is a detailed view of the data
        """
        st.write(f'<p style="color:#9c9d9f">{intro_text}</p>', unsafe_allow_html=True)

        st.dataframe(filter_df)

        st.markdown("The description of each column is as follows.")
        st.markdown(
          """
          - **Country**            : Name of the country
          - **City**               : Name of the city 
          - **AQI Value**          : Overall AQI value of the city
          - **AQI Category**       : Overall AQI category of the city
          - **CO AQI Value**       : AQI value of Carbon Monoxide of the city
          - **CO AQI Category**    : AQI category of Carbon Monoxide of the city
          - **Ozone AQI Value**    : AQI value of Ozone of the city
          - **Ozone AQI Category** : AQI category of Ozone of the city
          - **NO2 AQI Value**      : AQI value of Nitrogen Dioxide of the city
          - **NO2 AQI Category**   : AQI category of Nitrogen Dioxide of the city
          - **PM2.5 AQI Value**    : AQI value of Particulate Matter with a diameter of 2.5 micrometers or less of the city
          - **PM2.5 AQI Category** : AQI category of Particulate Matter with a diameter of 2.5 micrometers or less of the city
          """)

        st.subheader("|   Our Team")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
          st.write('MARTIN - ***Modeler***')
        with col2:
          st.write('DELVISA - ***Modeler***')
        with col3:
          st.write('YOHAN - ***Project Leader***')
        with col4:
          st.write('ENDAH - ***Visualizer***')
        with col5:
          st.write('CINDY - ***Analyst***')

        st.markdown("<p style='text-align: center;'>  ___ </p>", unsafe_allow_html=True)

        st.markdown("<p style='text-align: center;'><b>HelloWorld.ipynb</b> --- our <strong>Data Science</strong> project team. </p>", unsafe_allow_html=True)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%  GLOBAL TAB  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #---------------------------------------------------------------    
    # tab 2 ( Global )   
    #---------------------------------------------------------------  
    with tab2:
       # page title
        st.title('Global Air Pollution Dashboard')

        # metric untuk data global tanpa pengaruh filter
        col1,col2 = st.columns(2)
        with col1:
             glob_tot_country=data_glob.Country.nunique()
             st.metric('Total Country', value=glob_tot_country, delta=None, help="Total countries from the dataset (no filter)") 
             st.caption(f"There are {glob_tot_country} countries from the dataset.")

        with col2:
             glob_tot_city=data_glob.City.nunique()
             st.metric('Total City', value=glob_tot_city, delta=None, help='Total cities from the dataset (no filter)')
             st.caption(f"There are {glob_tot_city} cities from the dataset.")

        st.markdown("""---""")
        
        # Descriptive Statistics
        st.subheader("|  **Descriptive Statistics**")
        st.write(data_glob.describe())

        # Correlation Matrix
        st.subheader("|  **Correlation Matrix**")
        numeric_df = data_glob.select_dtypes(include=['int64', 'float64', 'float32'])
        corr_matrix = numeric_df.corr()
        desired_order = ['AQI Value', 'CO AQI Value',  'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']
        corr_matrix = corr_matrix.loc[desired_order, desired_order]
        annotations = []
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                value = round(corr_matrix.iloc[i, j], 2)
                text = str(int(value)) if value == 1 else str(value)
                annotations.append(
                    go.layout.Annotation(
                        x=corr_matrix.columns[j],
                        y=corr_matrix.index[i],
                        text=text,
                        showarrow=False,
                        font=dict(color="black" if abs(value) < 0.5 else "white")
                    )
                )
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            colorbar=dict(
              tickvals=[-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Nilai keterangan
              ticktext=['-0.2', '0.0', '0.2', '0.4', '0.6', '0.8', '1.0']),
            zmin=-0.3, zmax=1,
            text=corr_matrix.values,
            hoverinfo='x+y+z',  
            hovertemplate='x :  %{x}<br>y :  %{y}<br>Correlation :  %{z}<extra></extra>'
        ))
        fig.update_layout(
            autosize=False,
            width=500, 
            height=400, 
            annotations=annotations,
            xaxis=dict(title=''),
            yaxis=dict(title='', autorange='reversed')
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
        """
        A correlation matrix is a statistical tool used to evaluate the relationship between variables. 
        The correlation value range is between -1 to 1, where a value of 1 indicates a perfect positive 
        correlation, a value of -1 indicates a perfect negative correlation, and a value of 0 indicates 
        no correlation.
        """)

        st.markdown(
        """
       In the correlation matrix of this dataset, there is a high correlation of 0.98 between PM2.5 AQI 
       Value and AQI Value, which indicates that an increase in the PM2.5 AQI Value tends to be 
       accompanied by an increase in the AQI Value. This strong relationship indicates that PM 2.5 greatly 
       influences air quality or AQI Value in a city.
        """)

        st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

        st.markdown("<p style='text-align: center;'>You can filter the data you want to display in the visualization below.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>(Filter the data in the sidebar on the left). </p>", unsafe_allow_html=True)

        #########################################  metric data yang difilter
        st.markdown("""---""")

        st.title('Global Air Pollution Dashboard')
        st.subheader('(Selected/Filtered)')

        def metrics():

            col1,col2,col3,col4 = st.columns(4)

            with col1:
                # Metrics for Country
                if not AQI_Category and not countryy:
                  tot_country = data_glob["Country"].nunique()
                  st.metric('Total Country', value=tot_country, delta=None, help="Total countries from the dataset")
                  st.caption(f"There are **{tot_country}** **countries** from the dataset.")
                else:
                  filtercountries = filter_df["Country"].unique()
                  filteredcountry_count = len(filtercountries)
                  filteredcountry_list = ', '.join(filtercountries)

                  if len(filtercountries) == 1:
                    st.metric(label=f"Total Country", value=filteredcountry_count,delta=None, help="Filtering 1 country from the dataset")
                    st.caption(f"The filtered country is ***{countryy[0]}***.")
                  elif len(filtercountries) <= 10:
                    st.metric(label="Filtered Countries", value=filteredcountry_count, delta=None, help=f"Filtered {filteredcountry_count} countries from the dataset")
                    st.caption(f"There are {filteredcountry_count} countries from the filtered dataset.")
                    st.caption(f"{filteredcountry_list}") 
                  else:
                    st.metric(label="Filtered Countries", value=filteredcountry_count, delta=None, help=f"Filtered {filteredcountry_count} countries from the dataset")
                    st.caption(f"There are {filteredcountry_count} countries from the filtered dataset.")
                    st.caption(f"*)  Country names are not displayed because there are more than 10. Select the checkbox to see them.")
                    if st.checkbox('Show filtered countries'):
                        st.caption(f"{filteredcountry_list}") 
      
            with col2:
              # Metrics for City
              if not AQI_Category and not countryy:
                 # No filters applied
                 tot_city = data_glob["City"].nunique()
                 st.metric('Total City', value=tot_city, delta=None, help="Total cities from the dataset")
                 st.caption(f"There are **{tot_city}** **cities** from the dataset.")
              else:
                 filtered_cities = filter_df["City"].unique()
                 filteredcity_count = len(filtered_cities)
                 filteredcity_list = ', '.join(filtered_cities)
                
                 if len(filtercountries) == 1:
                     st.metric(label=f"Total City", value=filteredcity_count, delta=None, help=f"Total city in ***{countryy[0]}***")
                     if filteredcity_count == 1:
                        st.caption(f"There is 1 city in ***{countryy[0]}***.")
                     else:
                        st.caption(f"There are {filteredcity_count} cities in ***{countryy[0]}***.")
                 else:
                     st.metric(label="Total City", value=filteredcity_count, delta=None, 
                            help=f"The total cities of the filtered dataset is 66 {filteredcity_count}")
                     st.caption(f"There are {filteredcity_count} cities from the filtered dataset.") 

              with col3:
                # Metrics for Highest AQI Value
                highest_aqi = filter_df["AQI Value"].max()
                highest_aqi_city = filter_df[filter_df["AQI Value"] == highest_aqi]["City"].values[0]

                if not AQI_Category and not countryy:
                # No filters applied
                   highest_aqi_tot = data_glob['AQI Value'].max()
                   highest_aqi_city_tot = data_glob[data_glob['AQI Value'] == highest_aqi_tot]["City"].values[0]
                   st.metric('Highest AQI Value', value=highest_aqi_tot, delta=None, help="Highest AQI Value")
                  #  st.caption(f"The highest AQI value is **{highest_aqi_tot}**.")
                   st.caption(f"The highest AQI value is **{highest_aqi_tot}** in {highest_aqi_city_tot} city.")
                elif len(countryy) == 1:
                   st.metric(label=f"Highest AQI Value", value=highest_aqi, delta=None, help=f"Highest AQI Value from ***{countryy[0]}***")
                   st.caption(f"The highest AQI value in ***{countryy[0]}*** country is {highest_aqi}, in {highest_aqi_city} city.")
                else :
                  st.metric(label="Highest AQI Value", value=highest_aqi, delta=None, help=f"Highest AQI Value from the filtered dataset")
                  st.caption(f"The city with the highest AQI is {highest_aqi_city} with AQI value {highest_aqi}.")

              with col4:
              # Metrics for Lowest AQI Value
                lowest_aqi = filter_df["AQI Value"].min()
                lowest_aqi_city = filter_df[filter_df["AQI Value"] == lowest_aqi]["City"].values[0]

                if not AQI_Category and not countryy:
                # No filters applied
                   lowest_aqi_tot = data_glob['AQI Value'].min()
                   lowest_aqi_city_tot = data_glob[data_glob['AQI Value'] == lowest_aqi_tot]["City"].values[0]
                   st.metric('Lowest AQI Value', value=lowest_aqi_tot, delta=None, help="Lowest AQI Value")
                  #  st.caption(f"The lowest AQI value is **{lowest_aqi_tot}**.")
                   st.caption(f"The lowest AQI value is **{lowest_aqi_tot}** in {lowest_aqi_city_tot} city.")
                elif len(countryy) == 1:
                   st.metric(label=f"Lowest AQI Value", value=lowest_aqi, delta=None, help=f"Lowest AQI Value from ***{countryy[0]}***.")
                   st.caption(f"The lowest AQI value in ***{countryy[0]}*** country is {lowest_aqi}, in {lowest_aqi_city} city.")
                else :
                   st.metric(label="Lowest AQI Value", value=lowest_aqi, delta=None, help=f"Lowest AQI Value from the filtered dataset.")
                   st.caption(f"The city with the lowest AQI is {lowest_aqi_city} with AQI value {lowest_aqi}.") 

        metrics()

        # dataset dimensions
        rows = filter_df.shape[0]
        columns = filter_df.shape[1]
        st.write(" Global Air Pollution Dataset Dimensions (filter) :", rows , "Rows,", columns, "Columns" )

        if st.checkbox('Show Dataset (can be filtered)'):
          st.dataframe(filter_df)  

          if st.checkbox('Show Column Descriptions'):
             st.markdown("> The description of each column is as follows.")
             st.markdown(
          """
          - **Country**            : Name of the country
          - **City**               : Name of the city 
          - **AQI Value**          : Overall AQI value of the city
          - **AQI Category**       : Overall AQI category of the city
          - **CO AQI Value**       : AQI value of Carbon Monoxide of the city
          - **CO AQI Category**    : AQI category of Carbon Monoxide of the city
          - **Ozone AQI Value**    : AQI value of Ozone of the city
          - **Ozone AQI Category** : AQI category of Ozone of the city
          - **NO2 AQI Value**      : AQI value of Nitrogen Dioxide of the city
          - **NO2 AQI Category**   : AQI category of Nitrogen Dioxide of the city
          - **PM2.5 AQI Value**    : AQI value of Particulate Matter with a diameter of 2.5 micrometers or less of the city
          - **PM2.5 AQI Category** : AQI category of Particulate Matter with a diameter of 2.5 micrometers or less of the city
          """)      

        st.markdown("""---""")

        div1,div2=st.columns(2)
        # pie chart
        def pie():
            with div1:
                # st.subtitle('Cities by AQI Category')
                st.subheader('|  Cities by AQI Category')                
                
                # citycount_aqi_cat = data_glob.groupby(by='AQI Category', as_index=False)['City'].count()
                citycount_aqi_cat = filter_df.groupby('AQI Category')['City'].nunique().reset_index()
                citycount_aqi_cat.columns = ['AQI Category', 'Total_Cities']

                fig=px.pie(citycount_aqi_cat, values='Total_Cities', names='AQI Category', title='AQI Category Distribution')
                fig.update_layout(legend_title='AQI Category',legend_y=0.9)
                fig.update_traces(textinfo=None,textposition='inside')

                st.plotly_chart(fig,use_container_width=True, theme=theme_plotly)
        pie()

        def bar():
            with div2:
                st.subheader('|  AQI Category Distribution')

                countAQIcat=filter_df.groupby(by='AQI Category', as_index=False)['City'].count()

                fig=px.bar(countAQIcat, x='AQI Category', y='City', text = ['{:}'.format(x) for x in countAQIcat['City']],
                          template='seaborn')
                st.plotly_chart(fig, use_container_width=True, height=200)
        bar()

        def tes():
              st.subheader('|  Top Countries with the Highest AQI Values')
              # Mengonversi kolom AQI Value ke tipe numerik
              aqii = filter_df['AQI Value']
              pd.to_numeric(aqii, errors='coerce')

              # Menghitung nilai maksimum AQI per negara
              country_max_aqi = filter_df.groupby('Country')['AQI Value'].max().reset_index()

              # Mengambil 10 negara dengan nilai AQI tertinggi
              top_10_countries = country_max_aqi.nlargest(10, 'AQI Value')

              # Mengurutkan berdasarkan nilai AQI secara menaik
              top_10_countries = top_10_countries.sort_values(by='AQI Value', ascending=True)

              # Membuat plot menggunakan plotly express
              fig = px.bar(top_10_countries, x='AQI Value', y='Country', orientation='h', text='AQI Value')
              fig.update_traces(marker_color='brown', textposition='inside')
              fig.update_layout(xaxis_title='AQI Value', yaxis_title='Country')
              st.plotly_chart(fig)
        tes()

        def tes2():
              st.subheader('|  Top Countries with the Lowest AQI Values')
              # Mengonversi kolom AQI Value ke tipe numerik
              aqiiii = filter_df['AQI Value'] 
              pd.to_numeric(aqiiii, errors='coerce')

              # Menghitung nilai minimum AQI per negara
              country_min_aqi = filter_df.groupby('Country')['AQI Value'].min().reset_index()

              # Mengambil 10 negara dengan nilai AQI terendah
              top_10_countries = country_min_aqi.nsmallest(10, 'AQI Value')

              # Mengurutkan berdasarkan nilai AQI secara menaik
              top_10_countries = top_10_countries.sort_values(by='AQI Value', ascending=True)

              # Membuat plot menggunakan plotly express
              fig = px.bar(top_10_countries, x='AQI Value', y='Country', orientation='h', text='AQI Value')
              fig.update_traces(marker_color='navy', textposition='inside')
              fig.update_layout(xaxis_title='AQI Value', yaxis_title='Country')
              st.plotly_chart(fig)
        tes2()

        def tes3():
              st.subheader('|  Top Countries with the Most Polluted Cities')

              # Menghitung jumlah kota berpolusi per negara
              country_city_count = filter_df.groupby('Country')['City'].count().reset_index()

              # Mengambil 10 negara dengan jumlah kota berpolusi tertinggi
              top_10_countries = country_city_count.nlargest(10, 'City')

              # Mengurutkan berdasarkan jumlah kota berpolusi secara menurun
              top_10_countries = top_10_countries.sort_values(by='City', ascending=False)

              # Membuat plot menggunakan plotly express
              fig = px.bar(top_10_countries, x='City', y='Country', orientation='h', text='City')
              fig.update_traces(marker_color='darkseagreen', textposition='inside')

              # Memperbarui layout untuk membalikkan urutan sumbu y
              fig.update_layout(
              xaxis_title='Number of Polluted Cities',
              yaxis={'categoryorder':'total ascending'})

              # Menampilkan grafik
              st.plotly_chart(fig)
        tes3()

        def tes4():
              st.subheader('|  Countries with the Most Clean Cities')
              # Mengubah kolom 'AQI Value' menjadi tipe numerik
              filter_df['AQI Value'] = pd.to_numeric(filter_df['AQI Value'], errors='coerce')

              # Menentukan ambang batas untuk kota yang tidak berpolusi
              clean_air_threshold = 50

              # Memfilter data untuk kota yang tidak berpolusi
              clean_cities = filter_df[filter_df['AQI Value'] < clean_air_threshold]

              # Menghitung jumlah kota tidak berpolusi per negara
              clean_cities_count = clean_cities.groupby('Country')['City'].nunique().reset_index()
              clean_cities_count.columns = ['Country', 'Number of Clean Cities']

              # Mengambil 10 negara dengan jumlah kota tidak berpolusi terbanyak
              top_10_clean_countries = clean_cities_count.nlargest(10, 'Number of Clean Cities')

              # Mengurutkan berdasarkan jumlah kota tidak berpolusi
              top_10_clean_countries = top_10_clean_countries.sort_values(by='Number of Clean Cities', ascending=False)

              # Membuat plot menggunakan plotly express
              fig_clean = px.bar(top_10_clean_countries, x='Number of Clean Cities', y='Country', orientation='h', text='Number of Clean Cities')
              fig_clean.update_traces(marker_color='lightsteelblue', textposition='inside')

              # Memperbarui layout untuk membalikkan urutan sumbu y
              fig_clean.update_layout(yaxis={'categoryorder':'total ascending'})

              # Menampilkan grafik
              st.plotly_chart(fig_clean)
        tes4()

        c1,c2=st.columns(2)
        def dess():
            with c1:
              st.subheader("|  **Descriptive Statistics**")
              st.write(filter_df.describe())

              st.markdown(
                """
                 The table above provides a summary of descriptive statistics of numerical data, such as amount of data, 
                 mean, standard deviation, minimum value, quartiles, and maximum value.
                """)
        dess()

        def core():
            with c2:
              st.subheader("|  **Correlation Matrix**")
              numeric_df = filter_df.select_dtypes(include=[ 'int64'])
              corr_matrix = numeric_df.corr()
                #   Plot Correlation Matrix
              plt.figure(figsize=(5,3))
              sns.heatmap(corr_matrix, annot=True, cmap='RdBu', linewidths=0.5)
              plt.title('Correlation Matrix')
              st.pyplot(plt)

              st.markdown(
                """
                In the correlation matrix heatmap above, the color blue indicates that two variables are positively 
                correlated with each other (close to 1), meaning that when one variable goes up, the other tends to go 
                up too. Conversely, red indicates a negative correlation (close to -1), which means that when one 
                variable rises, the other tends to fall. The darker the blue, the stronger the correlation, while the 
                darker the red, the weaker the correlation.
                """)
        core()

        st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

########################################   AQI PREDICTION PAGE  ######################################################################
########################################   AQI PREDICTION PAGE  ######################################################################
# AQI Prediction Page
if selected == 'AQI Category Prediction':

    # Sidebar prediction page
    st.sidebar.header('AQI Category Prediction')
    st.sidebar.write(
      '''
      This is a page to display AQI category prediction results. 
      ''')
    st.sidebar.write(
      '''
      \nTo find out the prediction results for the AQI category, please enter the values for AQI CO, Ozone, NO2, and PM2.5 by moving the slider.
      Before that, you can enter the name of the city that will be predicted or you can ignore it.
      Than, click the 'AQI Category Prediction Results' button. 
      ''')
    st.sidebar.write(
      '''
      AQI category prediction results will be displayed.
      ''')

    st.sidebar.markdown("---")
    st.sidebar.write("Below is the accuracy of the prediction model:")
    st.sidebar.metric(label="Accuracy", value=f"{accuracy * 100:.2f}%", help='Accuracy of model prediction')

    # page title
    st.title('AQI Category Prediction using ML (Classification)')

    input_city = st.text_input('Enter the city name')
    city_name = input_city.upper()

    st.caption('Choose the AQI value of Carbon Monoxide (CO), Ozone (O3), Nitrogen Dioxide (NO2), and Particulate Matter (PM2.5)')

    col1, col2 = st.columns(2)
    with col1:
      CO_AQI_val    = st.slider('CO AQI Value', 0,500)
    with col2:
      Ozone_AQI_val = st.slider('OZONE AQI Value', 0,500)
    with col1:
      NO2_AQI_val   = st.slider('NO2 AQI Value', 0,500)
    with col2:
      PM25_AQI_val  = st.slider('PM2.5 AQI Value', 0,500)

    st.caption('Press this button if you have entered all AQI values.')
    
    def categAQI (category):
        if category == 'Good':
           return 'GOOD'
        elif category == 'Moderate':
            return 'MODERATE'
        elif category == 'Unhealthy for Sensitive Groups':
           return 'UNHEALTHY FOR SENSITIVE GROUPS'
        elif category == 'Unhealthy':
           return 'UNHEALTHY'
        elif category == 'Very Unhealthy':
           return 'VERY UNHEALTHY'
        elif category == 'Hazardous':
           return 'HAZARDOUS'
        else:
           return 'Unknown Category'
        
    # creating a button for Prediction
    if st.button('AQI Category Prediction Result'):
        user_input = np.array([CO_AQI_val, Ozone_AQI_val, NO2_AQI_val, PM25_AQI_val]).reshape(1, -1)
        # user_input = [int(x) for x in user_input]

        std = scaler.transform(user_input)
        
        # Prediksi dengan model
        prediction = global_model.predict(std)[0]
  
       # Menampilkan hasil prediksi
        AQIcateg_result = categAQI(prediction)

        if city_name:
           st.success(f'The predicted Air Quality Index (AQI) category in {city_name} City is ***{AQIcateg_result}***')
        else:
           st.success(f'The predicted Air Quality Index (AQI) category is ***{AQIcateg_result}***')


#=======================================================================================================================================
