## Big-Data-Systems-and-Int-Analytics

## CLAAT Link

https://codelabs-preview.appspot.com/?file_id=1kn-Tiq_mV8OgOf5_saPDB4s3bOHq9e_8_zKZNijl70c#6

## About

The aim of this assignment is creating an API using FAST API to serve the model, using AWS Cognito for authentication and then building a reference app in Streamlit. 

## Dataset

Dataset provided to us is available on https://registry.opendata.aws/sevir/


Task performed

Step 1: FastAPI
Step 2-: Authorization using Cognito
Step 4: Streamlit to test the API


## Architechture

![image](https://user-images.githubusercontent.com/59777007/130330580-e56ef227-8c63-424c-83f9-c6b21d3d9c72.png)

## Requirements

- Python 3.7+
- Signup for an AWS Account [here](https://portal.aws.amazon.com/billing/signup#/start).
- Install the `requirements.txt` file with command `pip install -r requirements.txt`
- Configure AWS CLI 
  * Open command line tool of choice on your machine and run `aws configure`. Enter your access and secret access keys and leave the default region name and output format as null. 

    ```
    $ aws configure
    AWS Access Key ID [None]: <access-key-from-aws-account>
    AWS Secret Access Key [None]: <secret-access-key-from-aws-account>
    Default region name [None]: 
    Default output format [None]: json
    ```
