## Table of Contents

- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)

## Project Overview

This is a dockerized implementation of https://github.com/oobabooga/text-generation-webui, which is a webui for LLM's. You can build the docker image yourself or use provided pre-built image.

## Prerequisites

Both pre-built docker images can be ran on a laptop with 16GB RAM. A GPU is required for running the GPU pre-built image, but is not required to run the other one. Actual minimal requirements are unknown, but users will also need at least 10GB storage to use the cpu-only pre-built image, and 26GB to use the GPU-supported image.

If you would like to build it yourself, the model does not come with this repo as it is too large, but can be downloaded from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML, or you can download your own.

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
This code will only build a image that only uses the CPU. Building an image can take  over 50 minutes.
To build it yourself, run 
```
docker build --target llama-cpu -t image/name:tag . 
```
To run the built image, run 
```
docker run -it -e EXTRA_LAUNCH_ARGS="--listen --verbose" -p 7860:7860 image/name:tag
```

To use a pre-built image with GPU support (26GB), run 
```
docker pull stargazingv3/llama2:local
docker run -it -e EXTRA_LAUNCH_ARGS="--share" -p 7860:7860 --gpus all stargazingv3/llama2:local
```

To use a pre-built image with CPU-only support (10GB), run
```
docker pull stargazingv3/llama2:cpu2
docker run -it -e EXTRA_LAUNCH_ARGS="--share" -p 7860:7860 stargazingv3/llama2:cpu2
```

Optionally, the user can add -rm to the docker command to automatically delete the container once done with running an image instance.

## Usage
- Both the cpu-only and the GPU support come pre-built with a 7 billion parameter llama2 model for use. The GPU support also contains a 13 billion llama2 model specialized in code generation.
- After running the docker run command, the user can interact with the webui by either visiting local host at the specified port if running on a local machine or, by using the --share argument, the .gradio link if running on a remote machine, which can be additionally be shared with others to use. When using --share, the local host will likely not work, the user must use the gradio link.
- Upon visiting the webui, users can go to the model tab and select a model to load from the dropdown, along with a transformer to use for the model and then press load. If using a GGML model, it is recommended to use the ctransformers transformer, and the ExLlamav2_HF transformer if using a GPTQ model. Not all transformers are compatible with all types of models. 
    - The times required to load the models, along with the token generation speed, heavily varies based on background usage.
    - Users can also download other models using the "Download model or LoRA" portion of the model tab, and can visit the Parameters tab to modify the parameters to use when loading in a model, such as max_new_tokens, which dictates how many tokens the model will generate up to each response, if desired.
- After loading a model, the user can go to the chat tab and select the type of inference they would like to do, where "chat" is for a normal chatbot, "instruct" is for code generation, "chat-instruct" is for a mix of both, and then begin chatting.
    - This is the main intended and tested feature. However, there are also many others the user can do, such as training the model.