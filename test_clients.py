import json
import grpc
from google.protobuf.json_format import ParseDict
import matcher_pb2
import matcher_pb2_grpc


def run_case(name: str, json_path: str, host="localhost", port=50051, timeout=20.0):
    print(f"\n=== Running test case: {name.upper()} ===")
    with open(json_path, "r") as f:
        data = json.load(f)
    job_dict = data["job"]
    applicant_dict = data["applicant"]

    # Convert dicts to proto messages
    job_msg = matcher_pb2.Job()
    applicant_msg = matcher_pb2.Applicant()
    ParseDict(job_dict, job_msg, ignore_unknown_fields=True)
    ParseDict(applicant_dict, applicant_msg, ignore_unknown_fields=True)

    # Create stub and call
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = matcher_pb2_grpc.MatcherStub(channel)
    try:
        resp = stub.Score(matcher_pb2.ScoreRequest(job=job_msg, applicant=applicant_msg), timeout=timeout)
    except grpc.RpcError as e:
        print(f"RPC Error: {e.code().name} - {e.details()}")
        return

    # Print formatted result
    print(f"Verdict: {resp.verdict}  |  Total Score: {resp.total_score:.2f}/100")
    print("Breakdown:")
    for f in sorted(resp.factors, key=lambda x: x.score, reverse=True):
        print(f"  - {f.name:22s} {f.score:6.2f}  :: {f.details}")


if __name__ == "__main__":
    # assumes JSON files are in ./tests/
    cases = {
        "strong": "json/strong.json",
        "medium": "json/medium.json",
        "weak": "json/weak.json"
    }
    for name, path in cases.items():
        run_case(name, path)
