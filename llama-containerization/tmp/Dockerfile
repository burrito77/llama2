# Start from a base image that has Python 3.10
#FROM python:3.10
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# APT INSTALLS
RUN apt update && apt install -y libsm6 libxext6 libxrender-dev git
RUN apt -y install python3-pip

# PIP
RUN python3 -m pip install --upgrade pip

# Install any required packages or dependencies for your Python application
RUN pip install llama-cpp-python==0.1.78

# Set the directory where the command will be executed
# You must define the APP_DIR environment variable if you're going to use it
#ENV APP_DIR=/app

WORKDIR /app

# By default, start a bash shell when the container launches
#CMD ["bash"]
