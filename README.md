# Applicant–Job Matcher (Semantic gRPC Service)

> **Note**: This project is integrated into the jobs.af **Applicant Tracking System (ATS)**, where it is used to score applicants based on a comparison of their profiles against specific job details.

A **semantic applicant–job matching microservice** built on **gRPC**.  
Send complete **Job** and **Applicant** messages defined in `matcher.proto` — no raw JSON parsing in clients.

The server uses a hybrid **semantic + heuristic scorer** with explainable factor breakdowns (skills, experience, education, etc.).

---

## Features

Structured gRPC interface (no REST or ad-hoc JSON)  
Semantic similarity via **Sentence-Transformers** (field-wise cosine)  
Automatic fallback to **TF-IDF** if embeddings unavailable  
Explainable factor breakdown:

- Tech/Skill match
- Experience
- Education
- Functional area alignment
- Location
- Languages
- Awards & Certificates  
   Ready-made Postman and Python test cases (Strong / Medium / Weak)

---

## Requirements

- Python **3.9+**
- Install dependencies:
  ```bash
  pip install grpcio grpcio-tools protobuf sentence-transformers scikit-learn numpy
  ```

> If you see NumPy 2.x ABI errors, downgrade:
>
> ```bash
> pip install "numpy<2"
> ```

---

## Generate gRPC Stubs

After editing `matcher.proto`:

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. matcher.proto
```

This produces:

- `matcher_pb2.py`
- `matcher_pb2_grpc.py`

---

## Run the Server

Start the gRPC matcher service:

```bash
python server.py
```

You’ll see:

```
Matcher gRPC server running on port 50051
[warmup] scoring backend primed.
```

---

## Scoring Overview

The logic lives in [`scoring.py`](./scoring.py).  
It combines **semantic vector similarity** and **rule-based heuristics**.

| Factor           | Weight | Method                                       |
| :--------------- | :----- | :------------------------------------------- |
| Tech/Skill Match | 40%    | Sentence-Transformer cosine (field-wise max) |
| Experience       | 25%    | Years vs job range                           |
| Education        | 10%    | Level comparison                             |
| Functional Area  | 8%     | Taxonomy overlap                             |
| Location         | 7%     | Country / Province match                     |
| Languages        | 5%     | Language overlap                             |
| Extras           | 5%     | Awards & Certificates bonus                  |

If `SentenceTransformer` can’t load (e.g., no NumPy), it gracefully falls back to TF-IDF cosine.

---

## Run Test Clients

We’ve included three test cases:  
`tests/strong.json`, `tests/medium.json`, and `tests/weak.json`.

Run all at once:

```bash
python test_clients.py
```

Output example:

```
=== Running test case: STRONG ===
Verdict: Strong match  |  Total Score: 82.70/100
Breakdown:
  - Skill Match              33.60  :: Embeddings cosine (field-max): ...
  - Experience               22.50  :: 4.5 yrs vs range 1.0–5.0
  - Education                10.00  :: Meets required level (bachelor)
  - Functional Area           8.00  :: Area matches (software engineering)
  - Location                  7.00  :: Same country, Same province
  - Languages                 3.50  :: Applicant languages: dari, english, pashto
  - Awards & Certificates     5.00  :: 1 award(s); 1 certificate(s)
```

---

## Using Postman (or `grpcurl`)

### Postman

1. Import `matcher.proto`.
2. Method → `Score`
3. Address → `localhost:50051`
4. Paste any JSON from `tests/strong.json`, `tests/medium.json`, or `tests/weak.json`.
5. Hit **Invoke** → see full factor response.

### grpcurl (CLI)

```bash
grpcurl -plaintext -d @ localhost:50051 matcher.Matcher/Score < tests/strong.json
```

---

## Environment Variables

| Variable         | Default            | Description                              |
| ---------------- | ------------------ | ---------------------------------------- |
| `SENTENCE_MODEL` | `all-MiniLM-L6-v2` | HuggingFace model for semantic matching  |
| `DISABLE_WARMUP` | unset              | Skip initial model load (useful for dev) |

---

## Project Structure

```
applicant-scorizer/
├── matcher.proto
├── matcher_pb2.py
├── matcher_pb2_grpc.py
├── server.py
├── scoring.py
├── test_clients.py
└── json/
    ├── strong.json
    ├── medium.json
    └── weak.json
```

---

## Example gRPC Flow

```python
import grpc
from google.protobuf.json_format import ParseDict
import matcher_pb2, matcher_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = matcher_pb2_grpc.MatcherStub(channel)

job, applicant = {...}, {...}  # from JSON
job_msg = matcher_pb2.Job()
applicant_msg = matcher_pb2.Applicant()
ParseDict(job, job_msg)
ParseDict(applicant, applicant_msg)

resp = stub.Score(matcher_pb2.ScoreRequest(job=job_msg, applicant=applicant_msg))
print(resp.total_score, resp.verdict)
```
