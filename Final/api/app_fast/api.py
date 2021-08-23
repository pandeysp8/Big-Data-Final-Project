import boto3
import numpy
import s3fs
import h5py
from fastapi import FastAPI
#import jwt
from fastapi import FastAPI, Depends, HTTPException, status

from fastapi.middleware.cors import CORSMiddleware
#from passlib.hash import bcrypt
#from tortoise import fields 
#from tortoise.contrib.fastapi import register_tortoise
#from tortoise.contrib.pydantic import pydantic_model_creator
#from tortoise.models import Model 
from app_1.v1 import routers as api_router
#from app.api.api_v2.api import router as api_router2
from mangum import Mangum
from enum import Enum

app = FastAPI(title='SEVIR',
    description='Prediction')


@app.post('/token')
async def token(form_data: OAuth2PasswordRequestForm = Depends()):
    return {'access_token' : form_data.username + 'token'}

@app.get('/')
async def read_root():
    return {"Message": "Hello"} 
    
@app.get("/Index")
async def root():
    return {"message`": "Get Index of Algorithm of your choice and do Inference!"}

    
client = boto3.client('s3', aws_access_key_id='AKIARRCUX3UHKNYFQFFM', aws_secret_access_key='RJFySe3pRJ4bP8Mrjxu+oJqEj+jukL9Xy33w3nFk')
client._request_signer.sign = (lambda *args, **kwargs: None)

s3 = boto3.resource('s3')
bucket=s3.Bucket('bucket-satellite')

app.include_router(api_router, prefix="/app_1/v1")
#app.include_router(api_router2, prefix="/api/v2")
#handler = Mangum(app)

def handler(event, context):
    print(event)

    handler = Mangum(app)
    response = handler(event, context) # Call the instance with the event arguments

    return response
