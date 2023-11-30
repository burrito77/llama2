# Project Name

Short description of your project.

## Table of Contents

- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)

## Project Overview

This is a dockerized version of https://github.com/oobabooga/text-generation-webui, which is a webui for LLM's. You can build the docker container yourself or use the provided pre-built containers.

## Prerequisites

The docker container can be ran on a laptop with 16GB RAM, where a gpu is optional. Actual minimal requirements are unknown, but users will also need at least 10GB storage to use the smallest pre-built container.

## File Structure
The expected file structure for minimal usage of the webui is
```
.
├── Dockerfile
├── README.md
├── config
│   ├── models
│   │   ├── Llama-2-7B-Chat-GGML
│   │   │   ├── config.json
│   │   │   └── llama-2-7b-chat.ggmlv3.q2_K.bin
│   │   └── place-your-models-here.txt
├── docker-compose.build.yml
├── docker-compose.yml
├── requirements-cpu.txt
└── scripts
    ├── build_extensions.sh
    ├── checkout_src_version.sh
    ├── docker-entrypoint.sh
    └── extensions_runtime_rebuild.sh
```
## Installation
This code will only build a container that only uses the CPU
To build yourself, run 
```
docker build --target llama-cpu -t image/name:tag . 
```
To run the built container, run 
```
docker run -it -e EXTRA_LAUNCH_ARGS="--listen --verbose" -p 7860:7860 stargazingv3/llama2:size
```

To use a pre-built container with GPU support, run 
```
docker pull stargazingv3/llama2:single
docker run -it -e EXTRA_LAUNCH_ARGS="--share" -p 7860:7860 --gpus all stargazingv3/llama2:single
```

To use a pre-built container with CPU-only support, run
```
docker pull stargazingv3/llama2:cpu
docker run -it -e EXTRA_LAUNCH_ARGS="--share" -p 7860:7860 stargazingv3/llama2:cpu
```

## Usage
- The pre-built cpu-only container comes with a Llama-2 model with 7 billion parameters, while the pre-built GPU support container comes with both a 7 billion and a 13 billion parameter model.
- After running the docker run command, the user can interact with the webui by either visiting local host at the specified port if running on a local machine or, by using the --share argument, the .gradio link if running on a remote machine, which can be additionally be shared with others to use.
- Upon visiting the webui, users can go to the model tab and select a model to load from the dropdown, along with a transformer to use for the model and then press load. If using a GGML model, it is recommended to use the _ transformer, and the _ transformer if using a GPTQ model.
    - Users can also download other models by putting their link on the right side of the model tab, and can visit the _ tab to modify the parameters to use when loading in a model, such as _ if desired.
- After loading a model, the user can go to the _ tab and select the type of inference they would like to do, where _ is for a normal chatbot, _ is for and _ is for code questions, and then begin chatting.
    - This is the main intended and tested feature. However, there are also many others the user can do, such as training the model.