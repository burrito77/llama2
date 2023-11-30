# Project Name

Short description of your project.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Commands](#commands)

## Project Overview

This is a dockerized version of https://github.com/oobabooga/text-generation-webui, which is a webui for LLM's. You can build the docker container yourself or use the provided pre-built containers.

## Prerequisites

The docker container can be ran on a laptop with 16GB RAM, where a gpu is optional. Actual minimal requirements are unknown, but users will also need at least 10GB storage to use the smallest pre-built container

## Installation

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