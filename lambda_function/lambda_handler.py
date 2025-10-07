import os
import json
import logging
import boto3
from botocore.config import Config

logger = logging.getLogger()
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Use environment variable for endpoint name
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "your-sagemaker-endpoint-name")

runtime = boto3.client(
    "sagemaker-runtime",
    config=Config(connect_timeout=2, read_timeout=10, retries={"max_attempts": 2})
)

def _extract_instances(event):
    # Supports both direct invocation and API Gateway proxy events
    body = event.get("body", event)
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            body = {"instances": body}

    if isinstance(body, dict):
        for key in ("instances", "inputs", "input"):
            if key in body:
                return body[key]

    # If the entire event is the instances
    if isinstance(body, list):
        return body

    raise ValueError('Provide "instances" (or "input"/"inputs") in the request body.')

def lambda_handler(event, context):
    try:
        instances = _extract_instances(event)
        payload = json.dumps({"instances": instances})

        resp = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=payload
        )
        result = json.loads(resp["Body"].read())
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result)
        }
    except Exception as e:
        logger.exception("Inference failed")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }
