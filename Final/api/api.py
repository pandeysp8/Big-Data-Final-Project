import requests
import json
import os
#import numpy as np
#import tensorflow as tf
#import pandas as pd
#import h5py
#import sys
#import datetime
import argparse
#import logging
from mangum import Mangum
#from fastapi_cloudauth.cognito import Cognito, CognitoCurrentUser, CognitoClaims


os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'

import boto3
#import s3fs
from fastapi import FastAPI

app = FastAPI()
s3 = boto3.resource('s3')

#x_test,y_test = read_data('../data/sample/nowcast_testing.h5',end=50)


    
#x_test,y_test = nowcast_read_data(nowcast_DATA,end=50)

def ind_search_nowcast():
    x_test, y_test = read_data('s3://bucket-satellite/nowcast_testing000.h5', end=11)
    print(x_test)
    print(y_test)

def ind_search_synrad():
    x_test, y_test = read_data('s3://bucket-satellite/synrad_testing.h5', end=11)
    print(x_test)
    print(y_test)


if __name__=='__main__':
    a,b= read_data('s3://bucket-satellite/nowcast_testing000.h5',end=10)
    c,d= read_data('s3://bucket-satellite/data/synrad_testing.h5', end=10)
    ind_search_nowcast()
    ind_search_synrad()
    x_test_n= "10,384,384,13"
    y_test_n= "10,384,384,12"
    x_test_s= "[192,192]"
    a= 10
    b= 10
    #print(a.shape)
    #print(b.shape)


#SIZE=5
#main --- read data --- loop --

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/synrad")
async def fetch_synrad(index:int):
    x_test_s= [192,192]
    x_test_r= [48,48]
    x_test_w= [384,384]
    return {"Input_IR069": x_test_s,
            "Input_IR071":x_test_s,
            "Input_lght":x_test_r,
            "Output":x_test_w} 


@app.get("/nowcast", tags=["Nowcast Model"])
async def read_root(text:int):
    x_test_n= [10,384,384,13]
    y_test_n= [10,384,384,12]
    #x_test, y_test = read_data('s3://bucket-satellite/data/synrad_testing.h5', end=11)
    return{"Input shape": x_test_n, "Output Shape":y_test_n, "Input": x_data_n}

client = boto3.client('s3', aws_access_key_id='AKIARRCUX3UHKNYFQFFM', aws_secret_access_key='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')
client._request_signer.sign = (lambda *args, **kwargs: None)

def handler(event, context):
    print(event)

    asgi_handler = Mangum(app)
    response = asgi_handler(event, context) # Call the instance with the event arguments

    return response