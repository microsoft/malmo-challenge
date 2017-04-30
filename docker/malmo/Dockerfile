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

# Install Malmo dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxerces-c3.1 \
    libav-tools \
    wget \
    unzip \
    xvfb \
    build-essential \
    git \
    cmake \
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
    zlib1g-dev \
    zlib1g \
    openjdk-8-jdk && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    update-ca-certificates -f

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64

WORKDIR /root
ENV MALMO_XSD_PATH /root/Malmo/Schemas
RUN git clone -b SockLogs https://github.com/Microsoft/Malmo && \
    wget https://raw.githubusercontent.com/bitfehler/xs3p/1b71310dd1e8b9e4087cf6120856c5f701bd336b/xs3p.xsl -P Malmo/Schemas && \
    mkdir Malmo/build && \
    cd Malmo/build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DINCLUDE_CSHARP=no -DINCLUDE_LUA=no .. && \
    make -j

# Precompile Malmo mod
WORKDIR /root/Malmo/Minecraft
RUN sed -i 's/gradlew runClient/gradlew --debug --stacktrace runClient/' launchClient.sh && \
    ./gradlew setupDecompWorkspace && \
    ./gradlew build

# Unlimited framerate settings
COPY options.txt /root/Malmo/Minecraft/run

COPY run.sh /root/
RUN chmod +x /root/run.sh

# Expose port
EXPOSE 10000

# Run Malmo
ENTRYPOINT ["/root/run.sh", "/root/Malmo/Minecraft/launchClient.sh"]

