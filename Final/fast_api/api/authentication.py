#H
import boto3
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi_cloudauth.cognito import Cognito, CognitoCurrentUser, CognitoClaims
from fastapi.security.api_key import APIKeyQuery, APIKeyCookie, APIKeyHeader, APIKey
#from mangum import Mangum
app = FastAPI()

# app = FastAPI(root_path="/prod")
userRegion = "us-east-1"
userClientId = ""
userPool = ""
auth = Cognito(region= userRegion, userPoolId= userPool, client_id= userClientId)
getUser = CognitoCurrentUser(region= userRegion, userPoolId= userPool, client_id= userClientId)

@app.get("/sign_up", tags=["User Sign Up"])
async def sign_up(userid: str, password: str):    
    def sign_up_cognito(userClientId,userPool,uid,pwd):
        cidp = boto3.client('cognito-idp')
    
        # try:
        res = cidp.sign_up(ClientId= userClientId, Username= uid, Password= pwd) #, UserAttributes=[{'Name': uid},],ValidationData=[{'Name': uid},],)
        print(res)
            # user.Status = "UnConfirmed"
            # try:
        if res.get('UserConfirmed') == False :    
            conf = cidp.admin_confirm_sign_up(UserPoolId= userPool,Username= uid)
            print(conf)
        else:
            print("already confirmed")
                # user.Status = "Confirmed"
            # except:
                # user.Status = "UnConfirmed"
                
            # try:     
        jwt = cidp.admin_initiate_auth( 
            UserPoolId= userPool,
            ClientId= userClientId,
            AuthFlow= "ADMIN_NO_SRP_AUTH",
            AuthParameters= {
            "USERNAME": uid,
            "PASSWORD": pwd
            })
            
            
        r = cidp.admin_initiate_auth( 
            UserPoolId= userPool,
            ClientId= userClientId,
            AuthFlow= "REFRESH_TOKEN_AUTH",
            AuthParameters= {
            "REFRESH_TOKEN" : jwt["AuthenticationResult"]["RefreshToken"]
            })
  
        Token = r.get('AuthenticationResult').get('AccessToken')
        return Token
                

    uid = userid # take from user
    pwd = password # take from user     
   
    AccessToken = sign_up_cognito(userClientId,userPool,uid,pwd)

    return AccessToken
           

@app.get("/Authentication", tags=["Auth"])
async def userauthentication(current_user: CognitoClaims = Depends(auth)):
    print("works")
    return ("works")
   
    
#handler = Mangum(app)
#T