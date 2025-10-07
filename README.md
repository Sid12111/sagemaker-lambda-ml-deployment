# AWS SageMaker and Lambda ML Deployment

This project demonstrates deploying, managing, and monitoring ML models using AWS SageMaker (real-time or serverless inference) and AWS Lambda.

## Features
- Train a sample model and deploy a SageMaker endpoint
- Invoke the endpoint via Lambda
- Monitor endpoint metrics with CloudWatch
- Easy to extend with your own models

## Prerequisites
- AWS account with IAM permissions for SageMaker, Lambda, CloudWatch
- Python 3.8+
- Packages: sagemaker, boto3, scikit-learn, joblib
