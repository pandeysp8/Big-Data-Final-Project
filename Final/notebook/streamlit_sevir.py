#H
#Core Packages
#from nowcast_reader import read_data
import os

os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'
import sys
import numpy as np
sys.path.append('../src/')
from urllib.parse import urlparse,urlsplit
import AuthStatus,json,requests, random, time, streamlit as st, numpy as np, pandas as pd
from PIL import Image
import h5py
import boto3
import s3fs
import matplotlib.pyplot as plt
#x_test,y_test = read_data('inp.txt',end=10)
from readers.synrad_reader import read_data
#import readers.nowcast #import now
from display.display import get_cmap
#from readers.synrad import syn
#filename = './data/nowcast_testing.h5'

x_test_syn,y_test_syn = read_data('../data/sample/synrad_testing.h5',end=10)

#x_test_now,y_test_now = read_data_now('../data/sample/nowcast_testing.h5',end=50)
#x_test_now
#Def Vars
jobID = ""
st.set_page_config(page_title='Sevir Data', page_icon=None, layout='centered', initial_sidebar_state='auto')
st.sidebar.markdown('**Available Options for Sevir Data** :tv:')
st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
#Radio Condtion
chosenRadioButton = st.sidebar.radio(
    "Available News Analysis Services",
    ("Home","Sign Up|In","Synrad Analytics","Nowcast Analytics")
)

if chosenRadioButton == "Home":
    selectedRadio="Home :house:"
elif chosenRadioButton == "Sign Up|In":
    selectedRadio="Sign Up:lock:|In:key:"
elif chosenRadioButton == "Synrad Analytics":
    selectedRadio="Synrad Model :bar_chart:"
elif chosenRadioButton == "Nowcast Analytics":
    selectedRadio="Nowcast Model:"
    

st.sidebar.markdown(f"**Selection: {selectedRadio} **")
st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
st.markdown('<style>body{background-color: #A3A6A9;}</style>',unsafe_allow_html=True)
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

#Radio Button Condition Home
if chosenRadioButton == 'Home':
    local_css("home.css")
    st.title("Sevir Data")
    AuthStatus.Status = False
    # st.info("Please press FastAPI Button to view FastAPI Documentation Page Landing")
    # docs = st.button('FastAPI')
    # st.info("Please press Documentation Button to view FastAPI Re-Documentation Page")
    # redoc = st.button('Documentation')
    # while docs:
    #     st.write(f'<iframe src="https://gbqrkn96z7.execute-api.us-east-1.amazonaws.com/prod/docs", width=900, height=600  , scrolling=True></iframe>', unsafe_allow_html=True)
    #     break
    # while redoc:
    #     st.write(f'<iframe src="https://gbqrkn96z7.execute-api.us-east-1.amazonaws.com/prod/redoc", width=900, height=600  , scrolling=True></iframe>', unsafe_allow_html=True)
    #     break

    st.image("banner.gif", caption='',use_column_width=True)
    
    # Make it a list BBC and Huzlers if you get time....
    #st.image("bbc.png", caption=st.write("check out artciles here [BBC](https://www.bbc.com/news)"),use_column_width=True)

    
    st.image("aws1.jpg", caption=st.write("Dataset available on AWS Open Registry: [Dataset](https://registry.opendata.aws/sevir/)"),use_column_width=True)
    

#Use the logic used in last Sentiment one! for authentication and add cognito validation on top!!!

#Radio Button Condition Sign Up/In
if chosenRadioButton == 'Sign Up|In':
    local_css("style.css")
    st.title(':male-factory-worker: **_User Sign Up_** :lock: | **_In_** :unlock: ')
    image = Image.open('login.jpg')
    st.image(image, caption='',use_column_width=True)
    st.subheader('_Please enter valid username and password_')
    username = st.text_input('Username','Username')
    password = st.text_input('Password', 'Password', type="password")
    Create = st.button('Create')
    if Create:
        response = requests.get(f"http://127.0.0.1:8000/sign_up?userid=userid7&password=password")#&current_user={token}")
        data_list = response.json()
        checker = str(data_list)
        if checker == "Already Exists":
            st.error(f"Error this User : {data_list}")
            st.info(f"Please try with a new user if you think is a error or get in touch with our Admin: HT")
        else:
            st.success(f"Token is : {data_list}")
            st.balloons()

    Login = st.button('Login')
    if Login:
        response = requests.get(f"https://gbqrkn96z7.execute-api.us-east-1.amazonaws.com/prod/Authentication?usrName={username}&usrPassword={password}")#&current_user={token}")
        data_list = response.json()
        print(f"value is : {data_list}")
        verified = data_list
        if verified == True:
            data_list = "Signed In Successfully"
            st.success(data_list) 
            AuthStatus.Status = True
            st.balloons()
        else:
            data_list = "Invalid Username/Password Combination"
            st.error(data_list)
            st.info("Please retry with valid Username and Password Combination or Create a new user if you are new!")

#Radio Button Condition Scrape Article
if chosenRadioButton == 'Scrape Article':
    st.title('**Scrape Article** :scroll:')
    local_css("scrape.css")
    image = Image.open('scraper.jpg')
    st.image(image, caption='',use_column_width=True)
    authValueFlag = AuthStatus.Status

    if authValueFlag != True:
        st.warning('Seems you are not authenticated')
    st.subheader('_Please enter link of Article to be scraped_')
    URL = st.text_input("Input URL","URL")
    auth_key = st.text_input("Input Token","Token")
    
    parser = urlsplit(URL)
    base = parser [1]
    base=str(base)
    #
    pre_auth = 'Bearer ' 
    #auth_key = 'eyJraWQiOiJWb1FQZFFHcnk1S3JKWlhRaWF5MTlKYUhmXC9OcTJQNGFuNW9hOVpQMkxQOD0iLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIyYzdkN2M5Ni03YTExLTRiOGEtYjcyOS00NTMzMzJhMmZkMzQiLCJhdWQiOiI1ZzVsNzY3Zm1vdTY4NDE4aTRsMzM1bGZzaiIsImV2ZW50X2lkIjoiYzAxOWFiYTYtYzBhZC00ZWQwLWI1MWItYmNmMDdmMWQ2ZDkwIiwidG9rZW5fdXNlIjoiaWQiLCJhdXRoX3RpbWUiOjE2MDc5NDM5MjQsImlzcyI6Imh0dHBzOlwvXC9jb2duaXRvLWlkcC51cy1lYXN0LTEuYW1hem9uYXdzLmNvbVwvdXMtZWFzdC0xX3c2TXNjNkNXQyIsImNvZ25pdG86dXNlcm5hbWUiOiJyYWphdmlwYWdhbCIsImV4cCI6MTYwNzk0NzUyNCwiaWF0IjoxNjA3OTQzOTI0fQ.JseCwVZ9BObAFQOf9K5r-cG6QmB9-r4SI1v4tUhclt4M6a0acaqf0Rgliv292sF5fmahXXfYNrJmcXvfXJWc-zEkvrtYcs9QEb3LpJI-POqmC0XgVLv7wXO-15L5ejQ8vJ9GcR5k5HpaOfPoq7a8-tCNqt5CTOAvP-VZs5UGo-muLimBop9ikWODKbWrwZbJDHVINcrvxTWl5CLuKGHQ8oji1ZPnUmaUMM4fHhDaaymjbyWrpWYJgHD9eNUCdedyYFBLojfnph6xd0sW47RrxNv_LNeGCgdVEHwtUayarQ0BlhGcR_eyoU9fb0s4wo-zxZ4KLDNZhWEE95pWlu_CmQ'
    auth = pre_auth + auth_key    
    headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': 'Bearer ' + auth_key
    }
    endPointBBC=(f"https://5lwkohy209.execute-api.us-east-1.amazonaws.com/prod/scrapeBBCNews?url={URL}")
    endPointHuzlers=(f"https://5lwkohy209.execute-api.us-east-1.amazonaws.com/prod/scrapeHuzlersNews?url={URL}")
    #
    if len(auth_key) < 12:
        st.warning("Token Seems Invalid|Null")
    if  st.button('Scrape Article'):
        if authValueFlag == True:
            if len(auth_key) < 12:
                auth_key = ""
                pre_auth = 'Bearer ' 
                #auth_key = 'eyJraWQiOiJWb1FQZFFHcnk1S3JKWlhRaWF5MTlKYUhmXC9OcTJQNGFuNW9hOVpQMkxQOD0iLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIyYzdkN2M5Ni03YTExLTRiOGEtYjcyOS00NTMzMzJhMmZkMzQiLCJhdWQiOiI1ZzVsNzY3Zm1vdTY4NDE4aTRsMzM1bGZzaiIsImV2ZW50X2lkIjoiYzAxOWFiYTYtYzBhZC00ZWQwLWI1MWItYmNmMDdmMWQ2ZDkwIiwidG9rZW5fdXNlIjoiaWQiLCJhdXRoX3RpbWUiOjE2MDc5NDM5MjQsImlzcyI6Imh0dHBzOlwvXC9jb2duaXRvLWlkcC51cy1lYXN0LTEuYW1hem9uYXdzLmNvbVwvdXMtZWFzdC0xX3c2TXNjNkNXQyIsImNvZ25pdG86dXNlcm5hbWUiOiJyYWphdmlwYWdhbCIsImV4cCI6MTYwNzk0NzUyNCwiaWF0IjoxNjA3OTQzOTI0fQ.JseCwVZ9BObAFQOf9K5r-cG6QmB9-r4SI1v4tUhclt4M6a0acaqf0Rgliv292sF5fmahXXfYNrJmcXvfXJWc-zEkvrtYcs9QEb3LpJI-POqmC0XgVLv7wXO-15L5ejQ8vJ9GcR5k5HpaOfPoq7a8-tCNqt5CTOAvP-VZs5UGo-muLimBop9ikWODKbWrwZbJDHVINcrvxTWl5CLuKGHQ8oji1ZPnUmaUMM4fHhDaaymjbyWrpWYJgHD9eNUCdedyYFBLojfnph6xd0sW47RrxNv_LNeGCgdVEHwtUayarQ0BlhGcR_eyoU9fb0s4wo-zxZ4KLDNZhWEE95pWlu_CmQ'
                auth = pre_auth + auth_key    
                headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': 'Bearer ' + auth_key
                }

            if base == "www.bbc.com":
                response = requests.get(endPointBBC, headers=headers)
                data_list = response.json()
                if type(data_list) == dict:
                    st.error("Invalid Token | Not Authenticated")
                    st.info(response)
                else:
                    st.success(data_list)
                    st.info(response)

            elif base == "www.huzlers.com":
                response = requests.get(endPointHuzlers, headers=headers)
                data_list = response.json()
                if type(data_list) == dict:
                    st.error("Invalid Token | Not Authenticated")
                    st.info(response)
                else:
                    st.success(data_list)
                    st.info(response)

            elif base == "":
                st.error("Please enter a valid URL of the article you would like to scrape, URL cannot be empty.")

            else:
                st.warning(f"we are still working to get {base} on-board, Stay Tuned!!!")
        else:
            st.error("Please Login and provide valid Token to use the service!!!")

#Radio Button Condition Fack|Fact
if chosenRadioButton == 'Fake|Fact':
    st.title('**Meet our AI** :snowman:')
    local_css("fakefact.css")
    image = Image.open('fakefact.jpg')
    st.image(image, caption='',use_column_width=True)
    authValueFlag = AuthStatus.Status

    if authValueFlag != True:
        st.warning('Seems you are not authenticated')
    st.subheader('_Please select Approach_')
    
    box = st.selectbox('',('Article URL', 'Manual Input'))
    st.write('You selected', box)
    if box == "Article URL":
        URL = st.text_input("Input URL","URL")
        auth_key = st.text_input("Input Token","Token")
        
        parser = urlsplit(URL)
        base = parser [1]
        base=str(base)
        #
        pre_auth = 'Bearer ' 
        #auth_key = 'eyJraWQiOiJWb1FQZFFHcnk1S3JKWlhRaWF5MTlKYUhmXC9OcTJQNGFuNW9hOVpQMkxQOD0iLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIyYzdkN2M5Ni03YTExLTRiOGEtYjcyOS00NTMzMzJhMmZkMzQiLCJhdWQiOiI1ZzVsNzY3Zm1vdTY4NDE4aTRsMzM1bGZzaiIsImV2ZW50X2lkIjoiYzAxOWFiYTYtYzBhZC00ZWQwLWI1MWItYmNmMDdmMWQ2ZDkwIiwidG9rZW5fdXNlIjoiaWQiLCJhdXRoX3RpbWUiOjE2MDc5NDM5MjQsImlzcyI6Imh0dHBzOlwvXC9jb2duaXRvLWlkcC51cy1lYXN0LTEuYW1hem9uYXdzLmNvbVwvdXMtZWFzdC0xX3c2TXNjNkNXQyIsImNvZ25pdG86dXNlcm5hbWUiOiJyYWphdmlwYWdhbCIsImV4cCI6MTYwNzk0NzUyNCwiaWF0IjoxNjA3OTQzOTI0fQ.JseCwVZ9BObAFQOf9K5r-cG6QmB9-r4SI1v4tUhclt4M6a0acaqf0Rgliv292sF5fmahXXfYNrJmcXvfXJWc-zEkvrtYcs9QEb3LpJI-POqmC0XgVLv7wXO-15L5ejQ8vJ9GcR5k5HpaOfPoq7a8-tCNqt5CTOAvP-VZs5UGo-muLimBop9ikWODKbWrwZbJDHVINcrvxTWl5CLuKGHQ8oji1ZPnUmaUMM4fHhDaaymjbyWrpWYJgHD9eNUCdedyYFBLojfnph6xd0sW47RrxNv_LNeGCgdVEHwtUayarQ0BlhGcR_eyoU9fb0s4wo-zxZ4KLDNZhWEE95pWlu_CmQ'
        auth = pre_auth + auth_key    
        headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': 'Bearer ' + auth_key
        }
        endPointBBC=(f"https://5lwkohy209.execute-api.us-east-1.amazonaws.com/prod/scrapeBBCNews?url={URL}")
        endPointHuzlers=(f"https://5lwkohy209.execute-api.us-east-1.amazonaws.com/prod/scrapeHuzlersNews?url={URL}")
        inputURL=(f"https://ggwys2ggi9.execute-api.us-east-1.amazonaws.com/prod/predict")
        #
        if len(auth_key) < 12:
            st.warning("Token Seems Invalid|Null")
        if  st.button('Scrape Article'):
            if authValueFlag == True:
                if len(auth_key) < 12:
                    auth_key = ""
                    pre_auth = 'Bearer ' 
                    #auth_key = 'eyJraWQiOiJWb1FQZFFHcnk1S3JKWlhRaWF5MTlKYUhmXC9OcTJQNGFuNW9hOVpQMkxQOD0iLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIyYzdkN2M5Ni03YTExLTRiOGEtYjcyOS00NTMzMzJhMmZkMzQiLCJhdWQiOiI1ZzVsNzY3Zm1vdTY4NDE4aTRsMzM1bGZzaiIsImV2ZW50X2lkIjoiYzAxOWFiYTYtYzBhZC00ZWQwLWI1MWItYmNmMDdmMWQ2ZDkwIiwidG9rZW5fdXNlIjoiaWQiLCJhdXRoX3RpbWUiOjE2MDc5NDM5MjQsImlzcyI6Imh0dHBzOlwvXC9jb2duaXRvLWlkcC51cy1lYXN0LTEuYW1hem9uYXdzLmNvbVwvdXMtZWFzdC0xX3c2TXNjNkNXQyIsImNvZ25pdG86dXNlcm5hbWUiOiJyYWphdmlwYWdhbCIsImV4cCI6MTYwNzk0NzUyNCwiaWF0IjoxNjA3OTQzOTI0fQ.JseCwVZ9BObAFQOf9K5r-cG6QmB9-r4SI1v4tUhclt4M6a0acaqf0Rgliv292sF5fmahXXfYNrJmcXvfXJWc-zEkvrtYcs9QEb3LpJI-POqmC0XgVLv7wXO-15L5ejQ8vJ9GcR5k5HpaOfPoq7a8-tCNqt5CTOAvP-VZs5UGo-muLimBop9ikWODKbWrwZbJDHVINcrvxTWl5CLuKGHQ8oji1ZPnUmaUMM4fHhDaaymjbyWrpWYJgHD9eNUCdedyYFBLojfnph6xd0sW47RrxNv_LNeGCgdVEHwtUayarQ0BlhGcR_eyoU9fb0s4wo-zxZ4KLDNZhWEE95pWlu_CmQ'
                    auth = pre_auth + auth_key    
                    headers = {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    'Authorization': 'Bearer ' + auth_key
                    }

                if base == "www.bbc.com":
                    response = requests.get(endPointBBC, headers=headers)
                    data_list = response.json()
                    if type(data_list) == dict:
                        st.error("Invalid Token | Not Authenticated")
                        st.info(response)
                    else:
                        st.success(data_list)
                        st.info(response)
                        response2 = requests.post(inputURL, headers=headers)
                        ans = response2.json()
                        st.success(ans)
                        st.info(response2)

                elif base == "www.huzlers.com":
                    response = requests.get(endPointHuzlers, headers=headers)
                    data_list = response.json()
                    if type(data_list) == dict:
                        st.error("Invalid Token | Not Authenticated")
                        st.info(response)
                    else:
                        st.success(data_list)
                        st.info(response)
                        response2 = requests.post(inputURL, headers=headers)
                        ans = response2.json()
                        st.success(ans)
                        st.info(response2)

                elif base == "":
                    st.error("Please enter a valid URL of the article you would like to scrape, URL cannot be empty.")

                else:
                    st.warning(f"we are still working to get {base} on-board, Stay Tuned!!!")
            else:
                st.error("Please Login and provide valid Token to use the service!!!")
    else:
        auth_key = st.text_input("Input Token","Token")
        if len(auth_key) < 12:
            st.warning("Token Seems Invalid|Null")
        pre_auth = 'Bearer ' 
        #auth_key = 'eyJraWQiOiJWb1FQZFFHcnk1S3JKWlhRaWF5MTlKYUhmXC9OcTJQNGFuNW9hOVpQMkxQOD0iLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIyYzdkN2M5Ni03YTExLTRiOGEtYjcyOS00NTMzMzJhMmZkMzQiLCJhdWQiOiI1ZzVsNzY3Zm1vdTY4NDE4aTRsMzM1bGZzaiIsImV2ZW50X2lkIjoiYzAxOWFiYTYtYzBhZC00ZWQwLWI1MWItYmNmMDdmMWQ2ZDkwIiwidG9rZW5fdXNlIjoiaWQiLCJhdXRoX3RpbWUiOjE2MDc5NDM5MjQsImlzcyI6Imh0dHBzOlwvXC9jb2duaXRvLWlkcC51cy1lYXN0LTEuYW1hem9uYXdzLmNvbVwvdXMtZWFzdC0xX3c2TXNjNkNXQyIsImNvZ25pdG86dXNlcm5hbWUiOiJyYWphdmlwYWdhbCIsImV4cCI6MTYwNzk0NzUyNCwiaWF0IjoxNjA3OTQzOTI0fQ.JseCwVZ9BObAFQOf9K5r-cG6QmB9-r4SI1v4tUhclt4M6a0acaqf0Rgliv292sF5fmahXXfYNrJmcXvfXJWc-zEkvrtYcs9QEb3LpJI-POqmC0XgVLv7wXO-15L5ejQ8vJ9GcR5k5HpaOfPoq7a8-tCNqt5CTOAvP-VZs5UGo-muLimBop9ikWODKbWrwZbJDHVINcrvxTWl5CLuKGHQ8oji1ZPnUmaUMM4fHhDaaymjbyWrpWYJgHD9eNUCdedyYFBLojfnph6xd0sW47RrxNv_LNeGCgdVEHwtUayarQ0BlhGcR_eyoU9fb0s4wo-zxZ4KLDNZhWEE95pWlu_CmQ'
        auth = pre_auth + auth_key    
        headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': 'Bearer ' + auth_key
        }

        Text = st.text_input("Input Text","Text should be atleast 100 words.")
        textLen = len(Text.split())

        manual=(f"https://ggwys2ggi9.execute-api.us-east-1.amazonaws.com/prod/predict?url_choice={Text}")

       
        if textLen < 100:
            st.warning("Article text seems too short!!! Please make sure its more than 100 words.")
        
        if  st.button('Scrape Article'):
            if authValueFlag == True:
                if len(auth_key) < 12:
                    auth_key = ""
                    pre_auth = 'Bearer ' 
                    auth = pre_auth + auth_key    
                    headers = {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    'Authorization': 'Bearer ' + auth_key
                    }

                if textLen > 100:
                    response = requests.post(manual, headers=headers)
                    data_list = response.json()
                    if type(data_list) == dict:
                        st.error("Invalid Token | Not Authenticated")
                        st.info(response)
                    else:
                        st.success(data_list)
                        st.info(response)
                else:
                    st.error("Text too short to give a prediction!!!")
            else:
                st.error("Please Login and provide valid Token to use the service!!!")


#Radio Button Condition Analytics Dashboard
if chosenRadioButton == 'Nowcast Analytics':
    
    image = Image.open('stre.png')
    # st.image(image, caption='',use_column_width=True)
    st.markdown(f'<iframe src="https://bucket-satellite.s3.amazonaws.com/nowcast_1.PNG" align="left" width="1800" height="1200"></iframe>', unsafe_allow_html=True)
    # DashboardList = ['Bigdata_16082005038410/Dashboard1','Bigdata2_16082843072930/Dashboard2','Bigdata3/Dashboard1','Bigdata4_16082854623570/Dashboard1','Bigdata5/Dashboard1']
    # URL = random.choice(DashboardList)
#/

#Radio Button Condition Analytics Dashboard
if chosenRadioButton == 'Synrad Analytics':
    image = Image.open('stre.png')
    # st.image(image, caption='',use_column_width=True)
    st.markdown(f'<iframe src="https://bucket-satellite.s3.amazonaws.com/radar.PNG" align="left" width="1800" height="1200"></iframe>', unsafe_allow_html=True)
    # DashboardList = ['Bigdata_16082005038410/Dashboard1','Bigdata2_16082843072930/Dashboard2','Bigdata3/Dashboard1','Bigdata4_16082854623570/Dashboard1','Bigdata5/Dashboard1']
    # URL = random.choice(DashboardList)
    # #URL = "Bigdata_16082005038410/Dashboard1"
    # st.markdown(f'<iframe src="https://public.tableau.com/views/{URL}?:showVizHome=no&:embed=true" align="left" width="1800" height="1200"></iframe>', unsafe_allow_html=True)
    # # Dashboard = st.button('Dashboard')
    # # while Dashboard:
    # #     st.write(f'<iframe src="https://public.tableau.com/views/ADBMS1/Industry?:showVizHome=no&:embed=true" align="left" width="1800" height="1200"></iframe>', unsafe_allow_html=True)
    # #     break
    
    



    

#T