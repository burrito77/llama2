#!/bin/bash

# Define color codes 🤩
green='\033[0;32m'
red='\033[0;31m'
nc='\033[0m'  # No color

# Define the Docker image name
image_name="container2/test:2.0" #user/image:tag

# Check if the Docker image exists
if docker image inspect "$image_name" &> /dev/null; then
  echo -e "${green}Docker image '$image_name' already exists.${nc}"
else
  # Build the Docker image from the './docker/' directory
  echo -e "${green}Building Docker image...${nc}"
  #docker build ./docker/ --tag "$image_name"
  docker build ./docker/ -t "$image_name" 
  #--pull --no-cache

  # Check if the image build was successful
  if [ $? -eq 0 ]; then
    echo -e "${green}Docker image '$image_name' built successfully.${nc}"
  else
    echo -e "${red}Docker image build failed.${nc}"
    exit 1  # Exit the script with an error code
  fi
fi

# Run the Docker container with a bind mount
echo -e "${green}Now running within Docker container. Restart this shell script to build again / update after any changes to requirements.txt. Type 'exit' to exit. ${nc}"
echo -e "${green}Lost? run 'python3 train.py --config_file_path configs/<your_config.json> ${nc}'"

docker run --gpus all -u $(id -u):$(id -g) -v "$PWD:/app" -it "$image_name"