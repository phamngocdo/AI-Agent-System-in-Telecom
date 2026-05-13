#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

vllm serve phamngocdo/telcollm-qwen \
  --served-model-name Telco-LLM \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype auto \
  --trust-remote-code

#vllm serve models/qwen3-8b/version3 \
#   --served-model-name Telco-LLM \
#   --host 0.0.0.0 \
#   --port 8001 \
#   --dtype auto \
#   --trust-remote-code