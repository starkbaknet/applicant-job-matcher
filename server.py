# server.py
from __future__ import annotations
from concurrent import futures
import threading
import time
import signal
import sys
import grpc
from google.protobuf.json_format import MessageToDict

import matcher_pb2
import matcher_pb2_grpc

# Import your new semantic scorer
# scoring.score(...) should be the function you pasted earlier
import scoring

# --- Optional: gRPC reflection so tools can discover your service without the proto ---
try:
    from grpc_reflection.v1alpha import reflection
    _HAS_REFLECTION = True
except Exception:
    _HAS_REFLECTION = False


class MatcherServicer(matcher_pb2_grpc.MatcherServicer):
    def Score(self, request, context):
        # If client set a deadline, you can read/heed it (optional)
        # deadline = context.time_remaining()  # seconds (float)
        try:
            # Convert protobuf → plain dict with snake_case keys
            job_dict = MessageToDict(request.job, preserving_proto_field_name=True)
            applicant_dict = MessageToDict(request.applicant, preserving_proto_field_name=True)

            total, factors, verdict = scoring.score(job_dict, applicant_dict)

            res = matcher_pb2.ScoreResponse(
                total_score=total,
                verdict=verdict,
            )
            for f in factors:
                # Defensive: tolerate missing fields in factor dicts
                res.factors.add(
                    name=str(f.get("name", "")),
                    score=float(f.get("score", 0.0)),
                    details=str(f.get("details", "")),
                )
            return res

        except grpc.RpcError:
            # If scoring raised a grpc error itself, just propagate
            raise
        except Exception as e:
            # Map unexpected exceptions to INTERNAL with a friendly message
            context.set_details(f"Scoring failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return matcher_pb2.ScoreResponse()  # empty response on error


def _warmup():
    """
    Warm the semantic model so first real request isn’t slow.
    (Safe no-op if scorer falls back or already cached.)
    """
    try:
        # Minimal texts that exercise the embedding/TF-IDF path
        job = {"title": "Warmup", "roleSummary": "Initialize embeddings"}
        applicant = {"bio": "Warmup applicant"}
        # A tiny sleep to ensure server finishes starting before heavy load
        time.sleep(0.2)
        _ = scoring.score(job, applicant)
        print("[warmup] scoring backend primed.")
    except Exception as e:
        print(f"[warmup] skipped or failed: {e}")


def serve(port: int = 50051, max_workers: int = 10):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    matcher_pb2_grpc.add_MatcherServicer_to_server(MatcherServicer(), server)

    # Enable reflection (optional but handy for Postman/grpcurl)
    if _HAS_REFLECTION:
        SERVICE_NAMES = (
            matcher_pb2.DESCRIPTOR.services_by_name['Matcher'].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)

    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Matcher gRPC server running on port {port}")

    # Start a background warmup so first user call is fast
    threading.Thread(target=_warmup, daemon=True).start()

    # Graceful shutdown on SIGINT/SIGTERM
    def handle_signal(signum, frame):
        print("\nShutting down gRPC server...")
        # Give in-flight RPCs up to 5 seconds to finish
        server.stop(grace=None).wait(5)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
