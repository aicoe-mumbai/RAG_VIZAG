#!/bin/bash
export HOME_PATH=$(pwd)
echo "1. Create and map a data folder for the models."
mkdir ${HOME_PATH}/tgi
mkdir ${HOME_PATH}/tgi/data
mkdir ${HOME_PATH}/tgi/app
export TGI_VOLUME=${HOME_PATH}/tgi/data
ls ${TGI_VOLUME}
echo "2. Create Python test client"
cat > ${HOME_PATH}/tgi/app/tgi_test.py <<EOF
import requests
headers = {#!/bin/bash
export HOME_PATH=$(pwd)
echo "1. Create and map a data folder for the models."
mkdir ${HOME_PATH}/tgi
mkdir ${HOME_PATH}/tgi/data
mkdir ${HOME_PATH}/tgi/app
export TGI_VOLUME=${HOME_PATH}/tgi/data
ls ${TGI_VOLUME}
echo "2. Create Python test client"
cat > ${HOME_PATH}/tgi/app/tgi_test.py <<EOF
import requests
headers = {
    "Content-Type": "application/json",
}
data = {
    "inputs": "What is Deep Learning?",
    "parameters": {
        "max_new_tokens": 20,
    },
}
response = requests.post('http://localhost:8080/generate', headers=headers, json=data)
print(response.json())
EOF
ls ${HOME_PATH}/tgi/app/

echo "4. (Optional) Stop and remove the tgi_server container."
docker stop tgi_server
docker container ls
docker rm tgi_server
echo "5. Define model and run the tgi_server"

# Set local model path
export LOCAL_MODEL_PATH=/home/test/Downloads/LLMOPS/LLMOPS/Qwen/Qwen3-30B-A3B-Instruct-2507
export LOCAL_MODEL_PATH=/home/test/Downloads/LLMOPS/LLMOPS/Qwen/qwen3-32b

# If you have the model downloaded locally, set it in the model directory
# Ensure the model files are downloaded to $LOCAL_MODEL_PATH first
# If not, you can download the model using Hugging Face or any other method to store it locally

export MODEL=$LOCAL_MODEL_PATH
export TAG=latest

# Run the Docker container with the local model
docker run --name tgi_server \
           --gpus '"device=0,1"' \
           --shm-size 30g \
           -p 8080:80 \
           -v ${TGI_VOLUME}:/data \
           -v ${MODEL}:/models \
           ghcr.io/huggingface/text-generation-inference:${TAG} \
           --model-id /models \
           --max-batch-prefill-tokens 28000 \
           --cuda-memory-fraction 0.95 \
           --max-batch-total-tokens 32000
           

