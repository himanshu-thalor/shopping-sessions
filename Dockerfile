# Shopping Sessions: native C++20 zero-dependency pipeline.
# Builds for linux/amd64 natively on the Intel Xeon Cascade Lake+ scorers.
FROM gcc:13-bookworm AS builder

WORKDIR /build
COPY src/ /build/src/

RUN g++ -O3 -std=c++20 -march=x86-64-v3 -mtune=cascadelake -flto -DNDEBUG \
    -Wall -Wextra -Werror -pthread -static-libstdc++ -static-libgcc \
    /build/src/main.cpp -o /build/shopping_sessions && \
    strip --strip-all /build/shopping_sessions

FROM debian:bookworm-slim

RUN useradd --uid 1000 --no-create-home --home-dir /scratch app && \
    mkdir -p /app && chown 1000:1000 /app

COPY --from=builder /build/shopping_sessions /app/shopping_sessions

USER 1000:1000
WORKDIR /app
CMD ["/app/shopping_sessions"]
