import os
from datetime import datetime, timedelta, timezone
import boto3

cloudwatch = boto3.client("cloudwatch")
endpoint_name = os.environ.get("ENDPOINT_NAME", "your-sagemaker-endpoint-name")

end = datetime.now(timezone.utc)
start = end - timedelta(hours=24)

def fetch(metric, stat="Average", period=300):
    resp = cloudwatch.get_metric_statistics(
        Namespace="AWS/SageMaker",
        MetricName=metric,
        Dimensions=[{"Name": "EndpointName", "Value": endpoint_name}],
        StartTime=start,
        EndTime=end,
        Period=period,
        Statistics=[stat],
    )
    dps = sorted(resp.get("Datapoints", []), key=lambda x: x["Timestamp"])
    print(f"{metric} ({stat}) — points: {len(dps)}")
    if dps:
        latest = dps[-1]
        print("  Latest:", latest["Timestamp"].isoformat(), latest[stat])

if __name__ == "__main__":
    for m in ["Invocations", "ModelLatency", "OverheadLatency", "Invocation4xxErrors", "Invocation5xxErrors"]:
        fetch(m)
