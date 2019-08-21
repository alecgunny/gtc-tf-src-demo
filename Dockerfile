ARG tag=19.07-py3
ARG release=v1.3.0

FROM nvcr.io/nvidia/tensorflow:$tag AS tf

FROM tf AS preproc-devel
ENV KAGGLE_CONFIG_DIR=/tmp/.kaggle
VOLUME $KAGGLE_CONFIG_DIR

RUN apt-get update && \
      apt-get install -y --no-install-recommends p7zip-full ffmpeg && \
      pip install kaggle && \
      rm -rf /var/lib/apt/lists/*

FROM preproc-devel AS preproc
COPY src/ /workspace

# build tensorrt inference server client image so that we can grab the
# model config proto for the tensorflow container
FROM ubuntu:16.04 AS wget
ARG release
RUN apt-get update && \
      apt-get install -y wget && \
      wget -O /tmp/clients.tar.gz https://github.com/NVIDIA/tensorrt-inference-server/releases/download/${release}/${release}_ubuntu1604.clients.tar.gz

FROM ubuntu:16.04 AS trtisclient
COPY --from=wget /tmp/clients.tar.gz /opt/tensorrtserver/clients/

RUN apt-get update && \
      apt-get install -y --no-install-recommends \
        curl \
        libcurl3-dev \
        libopencv-dev \
        libopencv-core-dev \
        python3 \
        python3-pip \
        python3-setuptools && \
      cd /opt/tensorrtserver/clients && \
      tar xzf clients.tar.gz && \
      pip3 install --upgrade \
        /opt/tensorrtserver/clients/python/tensorrtserver-*.whl \
       numpy \
       pillow && \
     rm -rf /var/lib/apt/lists/*

FROM tf
ENV MODELSTORE=/modelstore TENSORBOARD=/tensorboard
VOLUME $MODELSTORE $TENSORBOARD

COPY --from=trtisclient /usr/local/lib/python3.5/dist-packages/tensorrtserver/api/model_config_pb2.py /opt/model_config_pb2/
COPY src/ /workspace
ENV PYTHONPATH=$PYTHONPATH:/opt/model_config_pb2/
