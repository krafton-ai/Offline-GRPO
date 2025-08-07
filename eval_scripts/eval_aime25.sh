#!/bin/bash

# Check if model name is provided
if [ $# -eq 0 ]; then
	echo "Usage: $0 <model_name_or_folder>"
	echo "Example: $0 my_model_name"
	exit 1
fi

MODEL_NAME="$1"

echo "Running skythought evaluation for model: $MODEL_NAME"
echo "========================================"

# Run AIME25_1 evaluation
echo "Starting AIME25_1 evaluation..."
skythought evaluate \
	--model "$MODEL_NAME" \
	--task aime25_1 \
	--backend vllm \
	--backend-args tensor_parallel_size=4,seed=0,dtype="float32" \
	--sampling-params temperature=0.6,top_p=0.95 \
	--n 64

echo "AIME25_1 evaluation completed."
echo ""

# Run AIME25_2 evaluation
echo "Starting AIME25_2 evaluation..."
skythought evaluate \
	--model "$MODEL_NAME" \
	--task aime25_2 \
	--backend vllm \
	--backend-args tensor_parallel_size=4,seed=0,dtype="float32" \
	--sampling-params temperature=0.6,top_p=0.95 \
	--n 64

echo "AIME25_2 evaluation completed."
echo ""
