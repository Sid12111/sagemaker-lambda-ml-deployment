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

## Quickstart

1) Deploy a model endpoint
```bash
export SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/SageMakerExecutionRole
# Optional: use instance-based instead of serverless
# export USE_SERVERLESS=0
python sagemaker/train_model.py
This trains a sample scikit-learn model locally, uploads model.tar.gz to S3, creates a SageMaker Model with inference.py, and deploys an endpoint. Note the printed endpoint name.

 **Deploy the Lambda function**
Bash
export ENDPOINT_NAME=<your-endpoint-name>
cd lambda_function
# If you added any dependencies to requirements.txt:
# pip install -r requirements.txt -t .
zip -r lambda_function.zip .
# Create the Lambda function via AWS Console or AWS CLI and set the ENDPOINT_NAME env var

Test Lambda
Example event:
JSON

{
  "instances": [
    [5.1, 3.5, 1.4, 0.2],
    [6.7, 3.1, 4.7, 1.5]
  ]
}
Monitor endpoint metrics
Bash
- export ENDPOINT_NAME=<your-endpoint-name>
- python cloudwatch/monitor_endpoint.py

Cleanup to avoid charges
Bash
- aws sagemaker delete-endpoint --endpoint-name <your-endpoint-name>
- aws sagemaker delete-endpoint-config --endpoint-config-name <your-endpoint-name>
- aws sagemaker delete-model --model-name <your-endpoint-name>-model
Notes
Endpoint type: By default this uses Serverless Inference (lower cost for spiky/low-traffic). Set USE_SERVERLESS=0 to deploy on ml.m5.large.
Lambda: Do not ship boto3 in your ZIP; it’s included in the runtime.
IAM: The SageMaker execution role must allow S3 read of your model artifacts and SageMaker hosting.
text


Extra tips
- Use an environment variable for the endpoint in Lambda (already done).
- Add autoscaling or serverless for cost control. For provisioned endpoints, set Application Auto Scaling policies.
- Consider API Gateway in front of Lambda for a public API and add CORS headers if needed.
- For private VPC endpoints, attach a VPC config to the SageMaker endpoint and Lambda.
- Add CloudWatch alarms on Invocation5xxErrors, ModelLatency p95, and Invocations to catch anomalies.

If you want, I can:
- Add a deploy_model.py that updates an existing endpoint via new endpoint configs.
- Provide IaC (Terraform/CloudFormation) to create IAM roles, endpoint, and Lambda.
- Wire this up to API Gateway with a ready-to-use OpenAPI spec.
