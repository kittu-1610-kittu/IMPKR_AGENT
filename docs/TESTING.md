# Testing Guide

The codebase is protected by a suite of integration tests that verify database timings, multi-agent coordination, and staging security.

---

## 🏃 Running the Tests

To run the backend test suite, open a terminal at the project root and execute:

```bash
python -m pytest backend/tests/test_all.py -v -s
```

---

## 🔬 What is Tested

1. **Parallel Concurrency Latency**: Asserts that concurrent execution timings obey latency matching, verifying that the parallel fetch time is close to the slowest retriever delay.
2. **Shared Blackboard**: Tests lock-managed session states and plan write-back functions.
3. **Evidence Sync & Fusion**: Tests standard Softmax rankings and diversity exclusions.
4. **FastAPI Security**: Asserts that auth headers are enforced and concurrent request floods trigger `429 Too Many Requests`.
5. **Fault Tolerance**: Tests circuit breaker triggers and retry loops.
6. **Multi-Hop Traversal**: Verifies BFS graph path lookups up to depth 2.
