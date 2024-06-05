import numpy as np
import pandas as pd
import streamlit as st 
from streamlit_option_menu import option_menu 
import warnings as wr
wr.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt


import sweetviz as sv
import codecs

from scipy.stats import skew 
import datetime as dt

import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report

from PIL import Image
import io
import json
import pickle




# ***** STREAMLIT PAGE ICON ***** 

icon = Image.open("C:/Users/mahes/Downloads/icon.png")
# SETTING PAGE CONFIGURATION...........
st.set_page_config(page_title='SINGAPORE FLAT RESALE',page_icon=icon,layout="wide")

html_temp = """
        <div style="background-color:#fb607f;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">SINGAPORE FLAT RESALE ML MODEL PRICE PREDICTION</h1>
        </div>"""

# components.html("<p style='color:red;'> Streamlit Components is Awesome</p>")
components.html(html_temp)
style = "<style>h2 {text-align: center;}</style>"
style1 = "<style>h3 {text-align: left;}</style>"


selected = option_menu(None,
                       options = ["Home","Data View and EDA","Selling Price Predicton"],
                       icons = ["house-door-fill","bar-chart-line-fill","bi-binoculars-fill"],
                       default_index=0,
                       orientation="horizontal",
                       styles={"container": {"width": "100%"},
                               "icon": {"color": "white", "font-size": "24px"},
                               "nav-link": {"font-size": "24px", "text-align": "center", "margin": "-2px"},
                               "nav-link-selected": {"background-color": "#480607"}})

data = pd.read_csv("C:/Users/mahes/OneDrive/Desktop/singapore flat resale/singapore.csv")
df = pd.DataFrame(data)
df1 = df.copy()

def pre_processing():
    flat_type_mappings ={Type : 'MULTI GENERATION' if Type == 'MULTI-GENERATION'  else Type  for Type in df1['flat_type']}

    df1['flat_type'] = df1['flat_type'].map(flat_type_mappings)
    flat_model_mappings = {model : model.upper()  for model in df1['flat_model']}

    df1['flat_model'] = df1['flat_model'].map(flat_model_mappings)
    #month column converted with only day.
    df1['month'] = pd.to_datetime(df1['month'], format = '%Y-%m').dt.to_period('M')
    df1['year'] = df1['month'].dt.year
    df1['month'] = df1['month'].dt.month
    # Extract the minimum storey and maximum storey from 'storey_range' and convert it to integer

    df1['minimum_storey']=df1['storey_range'].str.split(' TO ').str[0].astype(int)
    df1['maximum_storey']=df1['storey_range'].str.split(' TO ').str[1].astype(int)
    df1['is_remaining_lease'] = np.where(df1['remaining'].isna()==True, 0, 1)
    df1['remaining'] = np.where(df1['remaining'].isna()==True, 'Not Specified', df1['remaining'])

    
    

def st_display_sweetviz(report_html, width=1000, height = 500):
    report_file = codecs.open(report_html,'r')
    page = report_file.read()
    components.html(page, width=width, height=height, scrolling=True)
    
    
def show_shape():
    st.write(df.shape)   

def show_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    return s

def show_values(df3):
    missing_values_count = df3.isnull().sum()
    st.table(missing_values_count)
    

    

def eda():
    tab1,tab2,tab3=st.tabs(["UNIVARIATE","BIVARIATE","AUTOEDA"])
    pre_processing()
    
    with tab1:
        
        #univariate analysis 
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure(figsize=(15,8))
        ax = sns.countplot(data=df,x=df.flat_type)
        ax.set_title("Flat_type")
        st.pyplot()
        
        plt.figure(figsize=(15,8))
        ax = sns.countplot(data=df,x=df.storey_range)
        ax.set_title("storey_range")
        st.pyplot()
    
        
        #univariate analysis 
        #analysing numerical variables using boxplot
        plt.figure(figsize=(10,8))
        ax = sns.boxplot(data=df,x=df.resale_price)
        ax.set_title("resale_price")
        st.pyplot()
        
        
        numerical_columns = ['floor_area_sqm', 'resale_price']
        skewness_plot(df, *numerical_columns)
        st.pyplot()
        Square_Root_Transformation(df, *numerical_columns)
        outlier_plot(df)
        st.pyplot()        

        
    with tab2:
        #Plotting a pair plot for bivariate analysis
        g = sns.PairGrid(df1,vars=['floor_area_sqm','resale_price','minimum_storey','maximum_storey'])
        #setting color
        g.map_upper(sns.scatterplot, color='crimson')
        g.map_lower(sns.scatterplot, color='limegreen')
        g.map_diag(plt.hist, color='orange')
        #show figure
        st.pyplot(g)
          
    with tab3:
        report = sv.analyze(df)
        report.show_html()
        st_display_sweetviz("SWEETVIZ_REPORT.html")   
        
def skewness_plot(df, *column):
        nrow = len(column)
        plot_no=0
        for col_name in column:
            if  'sqrt' in col_name:
                title= "After Treatment"
            else:
                title = "Before Treatment"

            plt.figure(figsize=(16, 8))

            plot_no+=1
            plt.subplot(nrow, 3, plot_no)
            sns.boxplot(x=col_name, data=df)
            plt.title('Boxplot - '+ title)

            plot_no+=1
            plt.subplot(nrow, 3, plot_no)
            sns.histplot(df[col_name])
            plt.title(f'histplot - Skewness: {skew(df[col_name])}')

            plot_no+=1
            plt.subplot(nrow, 3, plot_no)
            sns.violinplot(x=col_name, data=df)
            plt.title('Violinplot - ' + title)

        plt.tight_layout()
    
        return plt.show()
    
        
def Square_Root_Transformation(df, *column):

    for col_name in column:
        # Square Root Tansformation
        df[col_name+'_sqrt'] = np.sqrt(df[col_name])
        

    column =[i for i in df.columns if 'sqrt' in i]

    return skewness_plot(df, * column)

    
       
def outlier_plot(df):

    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    sns.boxplot(x='floor_area_sqm', data=df)
    plt.title('Boxplot - floor area sqm')

    plt.subplot(2, 2, 2)
    sns.boxplot(x='floor_area_sqm_sqrt', data=df)
    plt.title('Boxplot - floor area sqm sqrt')

    plt.subplot(2, 2, 3)
    sns.boxplot(x='resale_price', data=df)
    plt.title('Boxplot - '+ 'resale price')

    plt.subplot(2, 2, 4)
    sns.boxplot(x='resale_price_sqrt', data=df)
    plt.title('Boxplot - '+ 'resale price sqrt')
    plt.tight_layout()
    
    return plt.show()

def regression_model(test_data):
    with open(r'Decision_Tree_Model.pkl', 'rb' ) as file:
        model = pickle.load(file)
        data = model.predict(test_data)[0] ** 2
        return data
    
def flat_price_prediction():
    with open(r'Category_Columns_Encoded_Data.json', 'r') as file:
            data = json.load(file)
            st.title(":red[Singapore Resale] :blue[Flat Prices] :orange[Prediction]")

    col1, col2 = st.columns(2, gap= 'large')
    with col1:
        date = st.date_input("Select the **Item Date**", dt.date(2017, 1,1), min_value= dt.date(1990, 1, 1), max_value= dt.date(2023, 9,1))
        town = st.selectbox('Select the **Town**', data['town'])
        flat_type = st.selectbox('Select the **Flat Type**', data['flat_type'])
        block = st.selectbox('Select the **Block**', data['block']) 
        street_name = st.selectbox('Select the **Street Name**', data['street_name'])
    with col2:
        storey_range = st.selectbox('Select the **Storey Range**', data['storey_range'])
        floor_area_sqm = st.number_input('Enter the **Floor Area** in square meter', min_value = 28.0, max_value= 173.0, value = 60.0 )
        flat_model	= st.selectbox('Select the **flat_Model**', data['flat_model'])
        lease_commence_date	=st.number_input('Enter the **Lease Commence Year**', min_value = 1966.0, max_value= 2022.0, value = 2017.0 )
        remaining_lease	= st.selectbox('Select the **Remainig Lease**', data['remaining'])
    
    storey = storey_range.split(' TO ')
    if remaining_lease == 'Not Specified':
            is_remaining_lease = 0
    else:
            is_remaining_lease = 1

    test_data = [[date.month, data['town'][town], data['flat_type'][flat_type], data['block'][block], data['street_name'][street_name],
                            data['storey_range'][storey_range], floor_area_sqm, data['flat_model'][flat_model], lease_commence_date,
                            data['remaining'][remaining_lease], date.year, int(storey[0]), int(storey[1]),is_remaining_lease, 
                            np.sqrt(floor_area_sqm)]]
    st.markdown('Click below button to predict the **Flat Resale Price**')
    prediction = st.button('**Predict**')
    if prediction and test_data:
            st.markdown(f"### :bule[Flat Resale Price is] :green[$ {round(regression_model(test_data),3)}]")
            st.markdown(f"### :bule[Flat Resale Price in INR] :green[₹ {round(regression_model(test_data)*61.99,3)}]")

    

def home():
    html_temp = """
        <div style="background-color:#915c83;padding:10px;border-radius:10px">
        <h3 style="color:white;text-align:center;">HOME</h3>
        </div>"""
    # components.html("<p style='color:red;'> Streamlit Components is Awesome</p>")
    components.html(html_temp)
    col1,col2 = st.columns(2)
    with col1:
            st.image(Image.open("C:\\Users\\mahes\\Downloads\\imageml.png"),width=600)
            st.markdown("## :red[Done by] : UMAMAHESWARI S")
            st.markdown(style,unsafe_allow_html=True)
            st.markdown(":red[Githublink](https://github.com/mahes101)")
                     
    with col2:
            st.header(':red[SINGAPORE FLAT RESALE PRICE PREDICTION]')  
            st.markdown(style, unsafe_allow_html=True)    
            st.write("The copper industry deals with less complex data related to sales and pricing.")
            st.markdown(style1, unsafe_allow_html=True)
            st.header(':red[SKILLS OR TECHNOLOGIES]')
            st.markdown(style, unsafe_allow_html=True)
            st.write("Python scripting, Data Preprocessing, Visualization, EDA, Streamlit,ML Algorithm")
            st.markdown(style1, unsafe_allow_html=True)
            st.header(':red[DOMAIN]')
            st.markdown(style, unsafe_allow_html=True)
            st.write("REAL ESTATE")
            st.markdown(style1, unsafe_allow_html=True)
            st.header(':red[ML PREDICTION]')
            st.markdown(style, unsafe_allow_html=True)
            st.write("ML Regression model which predicts continuous variable ‘Selling_Price OF FLAT")
            st.markdown(style1, unsafe_allow_html=True)      

if selected == "Home":
    home()
    
if selected == "Data View and EDA":
    html_temp = """
        <div style="background-color:#915c83;padding:10px;border-radius:10px">
        <h3 style="color:white;text-align:center;">Data View and EDA</h3>
        </div>"""
    # components.html("<p style='color:red;'> Streamlit Components is Awesome</p>")
    components.html(html_temp)
    choice = st.sidebar.selectbox("Choose an option",["Data View","EDA"])
    if choice == "Data View":
        with st.expander("DATA VIEW"):
            st.dataframe(df)
        st.subheader("Number of rows and columns")
        show_shape()
        st.subheader("Information of dataset")
        s = show_info(df)
        st.text(s)
        st.subheader("Missing values count of each columns")
        show_values(df)
    elif choice == "EDA":
        eda()
    else:
        pass  
     
if selected == "Selling Price Predicton":
    flat_price_prediction()
    