#H
#Core Packages
#from nowcast_reader import read_data
import os
import h5py
import s3fs
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'
import sys
import numpy as np
sys.path.append('../src/')
from urllib.parse import urlparse,urlsplit
import AuthStatus,json,requests, random, time, streamlit as st, numpy as np, pandas as pd
from PIL import Image
import h5py
import tensorflow as tf
import boto3
import s3fs
import matplotlib.pyplot as plt

from readers.synrad_reader import read_data
#import readers.nowcast #import now

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
    ("Home","Sign Up|In","Synrad","Nowcast")
)

if chosenRadioButton == "Home":
    selectedRadio="Home :house:"
elif chosenRadioButton == "Sign Up|In":
    selectedRadio="Sign Up:lock:|In:key:"
elif chosenRadioButton == "Synrad":
    selectedRadio="Synrad Model :bar_chart:"
elif chosenRadioButton == "Nowcast":
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


nowcast_DATA = ('s3://'
         'bucket-satellite/data/nowcast_testing.h5')
    
def nowcast_read_data(filename, rank=0, size=1, end=None, dtype=np.float32, MEAN=33.44, SCALE=47.54):
    x_keys = ['IN']
    y_keys = ['OUT']
    s = np.s_[rank:end:size]
    H5PY_DEFAULT_READONLY=1
    s3 = s3fs.S3FileSystem()
    with s3.open(filename,'rb') as s3file:
        with h5py.File(s3file, 'r') as hf:
            IN  = hf['IN'][s]
            OUT = hf['OUT'][s]
    IN = (IN.astype(dtype)-MEAN)/SCALE
    OUT = (OUT.astype(dtype)-MEAN)/SCALE
    return IN,OUT
    
x_test,y_test = nowcast_read_data(nowcast_DATA,end=50)
    
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

mse_file  = 'models/nowcast/mse_model.h5'
mse_model = tf.keras.models.load_model(mse_file,compile=False,custom_objects={"tf": tf})





def Nowcast_visualize_result(models,x_test,y_test,idx,ax,labels):
    fs=10
    cmap_dict = lambda s: {'cmap':get_cmap(s,encoded=True)[0],
                           'norm':get_cmap(s,encoded=True)[1],
                           'vmin':get_cmap(s,encoded=True)[2],
                           'vmax':get_cmap(s,encoded=True)[3]}
    for i in range(1,13,3):
        xt = x_test[idx,:,:,i]*norm['scale']+norm['shift']
        ax[(i-1)//3][0].imshow(xt,**cmap_dict('vil'))
    ax[0][0].set_title('Inputs',fontsize=fs)
    
    pers = persistence().predict(x_test[idx:idx+1])
    pers = pers*norm['scale']+norm['shift']
    x_test = x_test[idx:idx+1]
    y_test = y_test[idx:idx+1]*norm['scale']+norm['shift']
    y_preds=[]
    for i,m in enumerate(models):
        yp = m.predict(x_test)
        if isinstance(yp,(list,)):
            yp=yp[0]
        y_preds.append(yp*norm['scale']+norm['shift'])
    
    for i in range(0,12,3):
        ax[i//3][2].imshow(y_test[0,:,:,i],**cmap_dict('vil'))
    ax[0][2].set_title('Target',fontsize=fs)
    
    # Plot Persistence
    for i in range(0,12,3):
        plot_hit_miss_fa(ax[i//3][4],y_test[0,:,:,i],pers[0,:,:,i],74)
    ax[0][4].set_title('Persistence\nScores',fontsize=fs)
    
    for k,m in enumerate(models):
        for i in range(0,12,3):
            ax[i//3][5+2*k].imshow(y_preds[k][0,:,:,i],**cmap_dict('vil'))
            plot_hit_miss_fa(ax[i//3][5+2*k+1],y_test[0,:,:,i],y_preds[k][0,:,:,i],74)

        ax[0][5+2*k].set_title(labels[k],fontsize=fs)
        ax[0][5+2*k+1].set_title(labels[k]+'\nScores',fontsize=fs)
        
    for j in range(len(ax)):
        for i in range(len(ax[j])):
            ax[j][i].xaxis.set_ticks([])
            ax[j][i].yaxis.set_ticks([])
    for i in range(4):
        ax[i][1].set_visible(False)
    for i in range(4):
        ax[i][3].set_visible(False)
    ax[0][0].set_ylabel('-45 Minutes')
    ax[1][0].set_ylabel('-30 Minutes')
    ax[2][0].set_ylabel('-15 Minutes')
    ax[3][0].set_ylabel('  0 Minutes')
    ax[0][2].set_ylabel('+15 Minutes')
    ax[1][2].set_ylabel('+30 Minutes')
    ax[2][2].set_ylabel('+45 Minutes')
    ax[3][2].set_ylabel('+60 Minutes')
    
    legend_elements = [Patch(facecolor=hmf_colors[1], edgecolor='k', label='False Alarm'),
                   Patch(facecolor=hmf_colors[2], edgecolor='k', label='Miss'),
                   Patch(facecolor=hmf_colors[3], edgecolor='k', label='Hit')]
    ax[-1][-1].legend(handles=legend_elements, loc='lower right', bbox_to_anchor= (-5.4, -.35), 
                           ncol=5, borderaxespad=0, frameon=False, fontsize='16')
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    
#Radio Button Condition Analytics Dashboard
if chosenRadioButton == 'Nowcast':
    user_input = st.text_input("Nowcast Index", key = "12")
    if st.button("Nowcast"):
        fig,ax = plt.subplots(4,13,figsize=(24,8), gridspec_kw={'width_ratios': [1,.2,1,.2,1,1,1,1,1,1,1,1,1]})
        Nowcast_visualize_result([mse_model],x_test,y_test,idx,ax,labels=['MSE','SC','MSE+SC','cGAN+MAE'])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
#/

# Load weights from best model on val set

synrad_DATA = ('s3://'
         'seviringestion/data/synrad_testing.h5')
         
    mse_weights_file = 'models/synrad/gan_mae_weights.h5'
    mse_model = tf.keras.models.load_model(mse_weights_file,compile=False,custom_objects={"tf": tf})

    mse_vgg_weights_file = 'models/synrad/mse_vgg_weights.h5'
    mse_vgg_model = tf.keras.models.load_model(mse_vgg_weights_file,compile=False,custom_objects={"tf": tf})

    gan_weights_file = 'models/synrad/gan_mae_weights.h5'
    gan_model = tf.keras.models.load_model(gan_weights_file,compile=False,custom_objects={"tf": tf})

    X,Y = synrad_read_data(synrad_DATA,end=64)
    


def Synrad_visualize_result(Y,y_preds,idx,ax):
    cmap_dict = lambda s: {'cmap':get_cmap(s,encoded=True)[0], 'norm':get_cmap(s,encoded=True)[1],
                           'vmin':get_cmap(s,encoded=True)[2], 'vmax':get_cmap(s,encoded=True)[3]}
    ax[0].imshow(X['ir069'][idx,:,:,0],**cmap_dict('ir069'))
    ax[1].imshow(X['ir107'][idx,:,:,0],**cmap_dict('ir107'))
    ax[2].imshow(X['lght'][idx,:,:,0],cmap='hot',vmin=0,vmax=10)
    ax[3].imshow(Y['vil'][idx,:,:,0],**cmap_dict('vil')) 
    for k in range(len(y_preds)):
        if isinstance(y_preds[k],(list,)):
            yp=y_preds[k][0]
        else:
            yp=y_preds[k]
        ax[4+k].imshow(yp[idx,:,:,0],**cmap_dict('vil'))
    for i in range(len(ax)):
        ax[i].xaxis.set_ticks([])
        ax[i].yaxis.set_ticks([])

#Radio Button Condition Analytics Dashboard
if chosenRadioButton == 'Synrad':
    input_data = st.text_input("Synrad Index", key = "13")
    if st.button("Synrad"):
        test_idx = [idx1,idx2,idx3]
            N=len(test_idx)
            fig,ax = plt.subplots(N,7,figsize=(12,4))
            for k,i in enumerate(test_idx):
                Synrad_visualize_result(Y,[y_pred_mse,y_pred_mse_vgg,y_pred_gan], i, ax[k] )
            
            ax[0][0].set_title('Input ir069')
            ax[0][1].set_title('Input ir107')
            ax[0][2].set_title('Input lght')
            ax[0][3].set_title('Truth')
            ax[0][4].set_title('Output\nMSE Loss')
            ax[0][5].set_title('Output\nMSE+VGG Loss')
            ax[0][6].set_title('Output\nGAN+MAE Loss')
            plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.05,
                                wspace=0.35)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        



    

#T