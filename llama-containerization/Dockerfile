# Start from a base image that has Python
#FROM python:3.9-slim
FROM python:3.10

#FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# APT INSTALLS
#RUN apt update && apt install -y libsm6 libxext6 libxrender-dev git
#RUN apt -y install python3-pip

# PIP
#RUN python3 -m pip install --upgrade pip

#RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && apt install -y python3.10

# Install any required packages or dependencies for your Python application
# For example, if llama_cpp is a package, you'd install it here.
pip install --force-reinstall --ignore-installed --no-cache-dir llama-cpp-python
pip install --force-reinstall --ignore-installed --no-cache-dir llama-cpp-python==0.1.48

# Set the directory where the command will be executed
WORKDIR $APP_DIR

# By default, execute the Python script, but allow the user to override this command
#CMD ["python", "app.py"]
