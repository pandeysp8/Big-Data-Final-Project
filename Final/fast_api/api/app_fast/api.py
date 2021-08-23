import boto3
import numpy
import s3fs
import h5py
from fastapi import FastAPI
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from app_fast.v1 import routers as api_router
from mangum import Mangum


app = FastAPI()


@app.get('/')
async def read_root():
    return {"Message": "Hello"} 
    

    
client = boto3.client('s3', aws_access_key_id='', aws_secret_access_key='')
client._request_signer.sign = (lambda *args, **kwargs: None)

s3 = boto3.resource('s3')
bucket=s3.Bucket('bucket-satellite')

app.include_router(api_router, prefix="/app_fast/v1")

handler = Mangum(app)

