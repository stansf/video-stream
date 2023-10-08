FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# configure timezone, our app depends on it.
RUN /usr/bin/ln -sf /usr/share/zoneinfo/America/Toronto /etc/localtime

# install software
RUN apt update \
    && apt -y install python3-dev python3-pip ffmpeg \
    && apt clean
RUN pip3 install -U pip

WORKDIR /app
COPY stream-producer/stream_producer/requirements.txt /app
RUN pip install -r /app/requirements.txt

COPY stream-producer/stream_producer /app/stream-producer
