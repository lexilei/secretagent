# True Detective benchmark Makefile
# All experiments use DeepSeek V3.1 via Together AI

SHELL := /bin/bash
export TOGETHER_API_KEY := $(shell cat ../../.env | grep TOGETHER_API_KEY | cut -d'"' -f2)

# Smoke tests (N=20)
smoke-workflow:
	uv run python expt.py run --config-file conf/workflow.yaml dataset.n=20

smoke-zeroshot:
	uv run python expt.py run --config-file conf/zeroshot.yaml dataset.n=20

smoke-selfdiscover:
	uv run python expt.py run --config-file conf/self_discover.yaml dataset.n=20

# Full runs (all 191 puzzles)
full-workflow:
	uv run python expt.py run --config-file conf/workflow.yaml dataset.n=null

full-zeroshot:
	uv run python expt.py run --config-file conf/zeroshot.yaml dataset.n=null

# Results analysis
results:
	uv run -m secretagent.cli.results average results/* --latest 0 --metric correct

compare:
	uv run -m secretagent.cli.results pair results/* --latest 0 --metric correct

download:
	uv run python data/download.py
