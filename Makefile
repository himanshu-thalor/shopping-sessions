IMAGE    ?= shopping-sessions:latest
CPUS     ?= 8
MEM      ?= 24g
PLATFORM ?= linux/amd64
DEV      ?= dev
ANSWER   ?= dev-answer
FIXTURES ?= fixtures
export IMAGE CPUS MEM PLATFORM

.PHONY: native build fixtures fixtures-native dev run-dev run-dev-native check-dev clean

native:
	mkdir -p build
	g++ -O3 -std=c++20 -march=x86-64-v3 -mtune=cascadelake -flto=auto -DNDEBUG -Wall -Wextra -Werror -pthread src/main.cpp -o build/shopping_sessions

build:
	docker buildx build --platform $(PLATFORM) --load -t $(IMAGE) .

fixtures: build
	python3 scripts/run_fixtures.py $(FIXTURES) -- scripts/docker_run.sh

fixtures-native: native
	python3 scripts/run_fixtures.py $(FIXTURES) -- ./build/shopping_sessions

dev:
	generate --seed 1 --events 1000000 --clicks 250000 --output $(DEV)

run-dev: build
	rm -rf out && time scripts/docker_run.sh $(DEV) out

run-dev-native: native
	rm -rf out && time DATA_DIR=$(DEV) OUTPUT_DIR=out SCRATCH_DIR=/tmp NCPU=$(CPUS) MEM_GB=24 ./build/shopping_sessions

check-dev:
	python3 scripts/compare.py out $(ANSWER)

clean:
	rm -rf build out
