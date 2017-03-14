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

FROM microsoft/cntk:2.0.beta12.0-gpu-python2.7-cuda8.0-cudnn5.1

# Version variables
ENV MALMO_VERSION = 0.21.0
ENV MALMOPY_VERSION=1.0.0RC1

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    zlib1g-dev \
    python-dev \
    python-pip \

    # install Malmo dependencies
    libpython2.7 \
    openjdk-7-jdk \
    lua5.1 \
    libxerces-c3.1 \
    liblua5.1-0-dev \
    libav-tools \
    python-tk \
    python-imaging-tk \
    wget \
    unzip && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Set CNTK Python PATH at first position to be picked automatically
ENV PATH=/root/anaconda3/envs/cntk-py27/bin:$PATH

# Update pip
RUN /root/anaconda3/envs/cntk-py27/bin/pip install --upgrade pip

# download and unpack Malmo

WORKDIR /root
RUN wget https://github.com/Microsoft/malmo/releases/download/$(MALMO_VERSION)/Malmo-$(MALMO_VERSION)-Linux-Ubuntu-16.04-64bit_withBoost.zip
RUN unzip Malmo-$(MALMO_VERSION)-Linux-Ubuntu-16.04-64bit.zip
RUN mv Malmo-$(MALMO_VERSION)-Linux-Ubuntu-16.04-64bit Malmo
RUN rm Malmo-$(MALMO_VERSION)-Linux-Ubuntu-16.04-64bit.zip
ENV MALMO_XSD_PATH /root/Malmo/Schemas
ENV PYTHONPATH /root/Malmo/Python_Examples

WORKDIR /root
RUN wget https://malmoexperimentsgrp975.blob.core.windows.net/malmo-drop/malmopy-$MALMOPY_VERSION.zip -P malmopy && \
    cd malmopy && \
    unzip malmopy-$MALMOPY_VERSION.zip && \
    rm malmopy-$MALMOPY_VERSION.zip && \
    /root/anaconda3/envs/cntk-py27/bin/pip install -e '.[all]'
