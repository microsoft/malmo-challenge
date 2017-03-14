# Copyright (c) 2017 Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================

FROM ubuntu:16.04

# Version variables
ENV MALMO_VERSION 0.21.0
ENV MALMOPY_VERSION 1.0.0RC1

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python-dev \
    python-pip \
    python-setuptools \
    cmake \
    zlib1g-dev \

    # install Malmo dependencies
    libpython2.7 \
    lua5.1 \
    libxerces-c3.1 \
    liblua5.1-0-dev \
    libav-tools \
    python-tk \
    python-imaging-tk \
    wget \
    unzip && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip install -U pip setuptools && pip install wheel && pip install chainer==1.21.0

# download and unpack Malmo
WORKDIR /root
RUN wget https://github.com/Microsoft/malmo/releases/download/$MALMO_VERSION/Malmo-$MALMO_VERSION-Linux-Ubuntu-16.04-64bit_withBoost.zip && \
    unzip Malmo-$MALMO_VERSION-Linux-Ubuntu-16.04-64bit_withBoost.zip && \
    rm Malmo-$MALMO_VERSION-Linux-Ubuntu-16.04-64bit_withBoost.zip && \
    mv Malmo-$MALMO_VERSION-Linux-Ubuntu-16.04-64bit_withBoost Malmo

ENV MALMO_XSD_PATH /root/Malmo/Schemas
ENV PYTHONPATH /root/Malmo/Python_Examples

WORKDIR /root
RUN wget https://malmoexperimentsgrp975.blob.core.windows.net/malmo-drop/malmopy-$MALMOPY_VERSION.zip -P malmopy && \
    cd malmopy && \
    unzip malmopy-$MALMOPY_VERSION.zip && \
    rm malmopy-$MALMOPY_VERSION.zip && \
    pip install -e '.[all]'