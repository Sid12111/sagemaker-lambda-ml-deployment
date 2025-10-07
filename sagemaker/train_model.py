import os
import tarfile
import uuid
import joblib
import boto3
import sagemaker

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.serverless import ServerlessInferenceConfig

def get_role():
    # If running in a SageMaker notebook/Studio, this works; otherwise use env var
    try:
        from sagemaker import get_execution_role
        return get_execution_role()
    except Exception:
        role = os.environ.get("SAGEMAKER_ROLE_ARN")
        if not role:
            raise RuntimeError("Set SAGEMAKER_ROLE_ARN env var with your SageMaker execution role ARN.")
        return role

def main():
    # 1) Train locally (sample Iris model)
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, "sample_model.pkl")

    # 2) Tar the model as model.tar.gz (SageMaker expects a tarball)
    with tarfile.open("model.tar.gz", "w:gz") as tar:
        tar.add("sample_model.pkl", arcname="model.pkl")

    # 3) Upload to S3
    sess = sagemaker.Session()
    role = get_role()
    endpoint_name = os.environ.get("SM_ENDPOINT_NAME", f"iris-rf-{uuid.uuid4().hex[:8]}")
    key_prefix = f"models/iris/{endpoint_name}"
    model_s3_uri = sess.upload_data(path="model.tar.gz", key_prefix=key_prefix)
    print("Uploaded model to:", model_s3_uri)

    # 4) Create SageMaker Model using the built-in SKLearn container + custom inference script
    # inference.py must sit next to this file (source_dir = this folder)
    source_dir = os.path.dirname(__file__)
    sklearn_model = SKLearnModel(
        model_data=model_s3_uri,
        role=role,
        entry_point="inference.py",
        source_dir=source_dir,
        framework_version="1.2-1",
        py_version="py3",
        name=f"{endpoint_name}-model"
    )

    # 5) Deploy (serverless by default; switch to instance-based if needed)
    use_serverless = os.environ.get("USE_SERVERLESS", "1") == "1"
    if use_serverless:
        serverless_config = ServerlessInferenceConfig(
            memory_size_in_mb=2048,
            max_concurrency=2
        )
        predictor = sklearn_model.deploy(
            serverless_inference_config=serverless_config,
            endpoint_name=endpoint_name
        )
    else:
        predictor = sklearn_model.deploy(
            instance_type="ml.m5.large",
            initial_instance_count=1,
            endpoint_name=endpoint_name
        )

    print("SageMaker endpoint deployed:", predictor.endpoint_name)

if __name__ == "__main__":
    main()
