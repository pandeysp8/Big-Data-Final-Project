#H
#Core Packages
from urllib.parse import urlparse,urlsplit
import AuthStatus,json,requests, random, time, streamlit as st, numpy as np, pandas as pd
from PIL import Image
#Def Vars
jobID = ""
st.set_page_config(page_title='Sevir Data', page_icon=None, layout='centered', initial_sidebar_state='auto')
st.sidebar.markdown('**Sevir Data**** :tv:')
st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
#Radio Condtion
chosenRadioButton = st.sidebar.radio(
    "Available News Analysis Services",
    ("Home","Sign Up|In","Scrape Article","Fake|Fact","Social Media","Analytics Dashboard","About Us")
)

if chosenRadioButton == "Home":
    selectedRadio="Home :house:"
elif chosenRadioButton == "Sign Up|In":
    selectedRadio="Sign Up:lock:|In:key:"
elif chosenRadioButton == "Scrape Article":
    selectedRadio="Scrape Article :clipboard:"
elif chosenRadioButton == "Fake|Fact":
    selectedRadio="Fake :question: | Fact :exclamation:"
elif chosenRadioButton == "Social Media":
    selectedRadio="Social Media :bird:" 
elif chosenRadioButton == "Analytics Dashboard":
    selectedRadio="Analytics Dashboard :bar_chart:"
elif chosenRadioButton == "About Us":
    selectedRadio="About Us :office:"
    

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
    st.title("News Article Impact & Analytics :computer:")
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

    st.image("banner.jpg", caption='',use_column_width=True)
    
    # Make it a list BBC and Huzlers if you get time....
    st.image("bbc.png", caption=st.write("check out artciles here [BBC](https://www.bbc.com/news)"),use_column_width=True)

    st.markdown(
    """<a style='display: block; text-align: center;' href="https://www.bbc.com/news">BBC News :newspaper:</a>
    """,
    unsafe_allow_html=True,
    )
    st.image("huzlers.png", caption=st.write("check out artciles here [Huzlers](https://www.huzlers.com/)"),use_column_width=True)
    st.markdown(
    """<a style='display: block; text-align: center;' href="https://www.huzlers.com/">Huzlers|Trending Content :rolled_up_newspaper:</a> 
    """,
    unsafe_allow_html=True,
    )

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
        response = requests.get(f"https://gbqrkn96z7.execute-api.us-east-1.amazonaws.com/prod/sign_up?userid={username}&password={password}")#&current_user={token}")
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




if chosenRadioButton == 'About Us':
    st.title(':bank: CYSE 7245 Team 6 :tm:')
    local_css("AboutUs.css")
    image = Image.open('team6.jpg')
    st.image(image, caption='',use_column_width=True)
    
    st.markdown("""**About Us:** A short summary :notebook: -> Journey :car:
    [**_Totally unnecessary but if you know me_** ] :stuck_out_tongue_winking_eye: """)
    st.markdown("It took us about 21 Tutorials :blue_book: & :closed_book: 4 Assignments... I guess more :books:. Timelines were stressing :disappointed_relieved: but we had fun :tada: or lets just assume that we did. :grin: ")
    st.markdown("The lecture has definitely upskilled us :wrench: or atleast most of us. :hammer: Also, added wonderfull friday nightlife memories. :city_sunset: As due to Covid :mask: there was no thrill on weekends. Glad I had this! :wink: ")
    st.markdown("Well, lets keep it short. :sweat_smile: Thankyou CSYE 7245 for making our semester great. :beers: Lets make sure and run into each-other whenever we are back to old normal. :sunglasses: - HT :tm:")
    st.markdown("**Special Thanks** -- **_Prof.Sri Krishnamurthy_** :man: & **_TA.Gurjot Kaur_** :girl:")

    

#T