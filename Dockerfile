FROM ubuntu:22.04 as app_base
# Pre-reqs
RUN apt-get update && apt-get install --no-install-recommends -y \
    git vim build-essential python3-dev python3-venv python3-pip
# Instantiate venv and pre-activate
RUN pip3 install virtualenv
RUN virtualenv /venv
# Credit, Itamar Turner-Trauring: https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV VIRTUAL_ENV=/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip3 install --upgrade pip setuptools && \
    pip3 install torch torchvision

# Copy and enable all scripts
COPY ./scripts /scripts
RUN chmod +x /scripts/*
ARG LCL_SRC_DIR="text-generation-webui"
COPY ${LCL_SRC_DIR} /src

ENV LLAMA_CUBLAS=1
# Copy source to app
RUN cp -ar /src /app
# Install oobabooga/text-generation-webui
RUN --mount=type=cache,target=/root/.cache/pip pip3 install -r /app/requirements_cpu_only.txt
# Install extensions
RUN --mount=type=cache,target=/root/.cache/pip \
    chmod +x /scripts/build_extensions.sh && . /scripts/build_extensions.sh

FROM ubuntu:22.04 as base
# Runtime pre-reqs
RUN apt-get update && apt-get install --no-install-recommends -y \
    python3-venv python3-dev git
# Copy app and src
COPY --from=app_base /app /app
COPY --from=app_base /src /src
# Copy and activate venv
COPY --from=app_base /venv /venv
ENV VIRTUAL_ENV=/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Finalise app setup
WORKDIR /app
EXPOSE 7860
EXPOSE 5000
EXPOSE 5005
# Required for Python print statements to appear in logs
ENV PYTHONUNBUFFERED=1
# Force variant layers to sync cache by setting --build-arg BUILD_DATE
ARG BUILD_DATE
ENV BUILD_DATE=$BUILD_DATE
RUN echo "$BUILD_DATE" > /build_date.txt
ARG VERSION_TAG
ENV VERSION_TAG=$VERSION_TAG
RUN echo "$VERSION_TAG" > /version_tag.txt
# Copy and enable all scripts
COPY ./scripts /scripts
RUN chmod +x /scripts/*
COPY ./config/models /app/models
ENTRYPOINT ["/scripts/docker-entrypoint.sh"]

FROM base AS llama-cpu
RUN echo "LLAMA-CPU" >> /variant.txt
RUN unset TORCH_CUDA_ARCH_LIST LLAMA_CUBLAS
RUN set "CMAKE_ARGS=-DLLAMA_OPENBLAS=on"
RUN set "FORCE_CMAKE=1"
RUN apt install gcc-11 g++-11 -y
RUN pip install llama-cpp-python --no-cache-dir
ENV EXTRA_LAUNCH_ARGS=""
CMD ["python3", "/app/server.py", "--cpu"]

FROM base AS default
RUN echo "DEFAULT" >> /variant.txt
ENV EXTRA_LAUNCH_ARGS=""
CMD ["python3", "/app/server.py"]