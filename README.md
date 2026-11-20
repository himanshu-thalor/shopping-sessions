# Shopping Sessions — C++20 Native Pipeline

High-performance, zero-dependency C++20 implementation for the **Databricks - Shopping Sessions** Buildathon track.

## Layout

- `src/` — Zero-dependency C++20 pipeline (`main.cpp` + modular headers for mmap I/O, exact CSV/timestamp parsing, DSU ID linking, cache-friendly sharded processing, and direct Parquet v1 output).
- `Dockerfile` — Multi-stage build (`gcc:13-bookworm` -> `debian:bookworm-slim`, `uid 1000`, read-only rootfs compatible).
- `fixtures/` — Official 36 edge-case fixtures (`E1..E20`, `A1..A12`, `L1..L4`).
- `scripts/` — Official verification scripts (`run_fixtures.py`, `compare.py`, `docker_run.sh`).
- `rules.md` — Track rules specification.

## Build and Test

```sh
# Native build & fixture suite
make native
make fixtures-native

# Generate 1M-event dev dataset & verify exact match against dev-answer/
make dev
make run-dev-native
make check-dev

# Docker build & run with scorer flags (linux/amd64, 8 vCPUs, 24 GB, read-only root, no network)
make build
make fixtures
make run-dev
make check-dev
```
