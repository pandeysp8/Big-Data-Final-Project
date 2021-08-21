import requests
import json
import os
import numpy as np
#import tensorflow as tf
#import pandas as pd
import h5py
#import sys
#import datetime
import argparse
#import logging
#from mangum import Mangum
#from fastapi_cloudauth.cognito import Cognito, CognitoCurrentUser, CognitoClaims


os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'

import boto3
import s3fs
from fastapi import FastAPI

app = FastAPI()
s3 = boto3.resource('s3')

#x_test,y_test = read_data('../data/sample/nowcast_testing.h5',end=50)

def read_data(filename, rank=0, size=1, end=None, dtype= np.float32, MEAN= 33.44, SCALE = 47.54):
    xkeys= ['IN']
    y_keys= ['OUT']
    s= np.s_[rank:end:size]

    s3= s3fs.S3FileSystem()
    with s3.open(filename, 'rb') as s3File:
        with h5py.File(s3File, 'r') as hf:
            IN= hf ['IN'][s]
            OUT= hf ['OUT'][s]
    IN = (IN.astype(dtype)-MEAN )/SCALE
    OUT= (OUT.astype(dtype)-MEAN )/SCALE
    return IN, OUT

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
    return{"Input shape": print(a), "Output Shape":print(b)}


@app.get("/nowcast", tags=["Nowcast Model"])
async def read_root(text:int):
    x_test_n= "10,384,384,13"
    y_test_n= "10,384,384,12"
    #x_test, y_test = read_data('s3://bucket-satellite/data/synrad_testing.h5', end=11)
    return{"Input shape": x_test_n, "Output Shape":y_test_n}

