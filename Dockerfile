FROM tensorflow/tensorflow:2.8.2-gpu

ARG DEBIAN_FRONTEND=noninteractive

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget

# Add new user to avoid running as root
RUN useradd -ms /bin/bash tensorflow
USER tensorflow
WORKDIR /home/tensorflow

# Copy this version of of the model garden into the image
COPY --chown=tensorflow protos/ /home/tensorflow/object_detection/protos

# Compile protobuf configs
RUN (protoc object_detection/protos/*.proto --python_out=.)

COPY packages/tf2/setup.py .
ENV PATH="/home/tensorflow/.local/bin:${PATH}"

RUN python -m pip install -U pip
RUN python -m pip install .

ENV TF_CPP_MIN_LOG_LEVEL 3

EXPOSE 5000

ENV TFHUB_CACHE_DIR=/app/.cache/tfhub_modules
ENV FLASK_ENV="docker"
ENV FLASK_APP=/app/app.py

COPY app/ /app
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

CMD ["gunicorn", "--reload", "--bind", "0.0.0.0:5000", "app:app"]