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

FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ssh \
    git-all \
    python-pip \
    openjdk-8-jdk \
    openmpi-bin \

    # install Malmo dependencies
    wget \
    unzip \
    xvfb \
    cmake-qt-gui \
    libboost-all-dev \
    libpython2.7-dev \
    swig \
    xsdcxx \
    libxerces-c-dev \
    doxygen \
    xsltproc \
    python-tk \
    python-imaging-tk \
    zlib1g \
    zlib1g-dev && \
    update-ca-certificates -f

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/

# download and build Malmo
WORKDIR /root
ENV MALMO_XSD_PATH /root/Malmo/Schemas
ENV PYTHONPATH /root/Malmo/build/Malmo/src/PythonWrapper
RUN git clone https://github.com/Microsoft/Malmo && \
    wget https://raw.githubusercontent.com/bitfehler/xs3p/1b71310dd1e8b9e4087cf6120856c5f701bd336b/xs3p.xsl -P Malmo/Schemas && \
    mkdir Malmo/build && \
    cd Malmo/build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DINCLUDE_CSHARP=no -DINCLUDE_LUA=no .. && \
    make -j

# add and install malmopy, malmo challenge task and samples
WORKDIR /root
RUN git clone https://github.com/Microsoft/malmo-challenge.git
WORKDIR /root/malmo-challenge
RUN pip install wheel && pip install setuptools && pip install -e '.[all]'
