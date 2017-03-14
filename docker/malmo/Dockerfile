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

ENV MALMO_VERSION 0.21.0

# Install Malmo dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-8-jdk \
    libxerces-c3.1 \
    libav-tools \
    wget \
    unzip \
    xvfb && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Download and unpack Malmo
WORKDIR /root
RUN wget https://github.com/Microsoft/malmo/releases/download/$MALMO_VERSION/Malmo-$MALMO_VERSION-Linux-Ubuntu-16.04-64bit_withBoost.zip && \
    unzip Malmo-$MALMO_VERSION-Linux-Ubuntu-16.04-64bit_withBoost.zip && \
    rm Malmo-$MALMO_VERSION-Linux-Ubuntu-16.04-64bit_withBoost.zip && \
    mv Malmo-$MALMO_VERSION-Linux-Ubuntu-16.04-64bit_withBoost Malmo
ENV MALMO_XSD_PATH /root/Malmo/Schemas

# Precompile Malmo mod
RUN mkdir ~/.gradle && echo 'org.gradle.daemon=true\n' > ~/.gradle/gradle.properties
WORKDIR /root/Malmo/Minecraft
RUN ./gradlew setupDecompWorkspace
RUN ./gradlew build

# Unlimited framerate settings
COPY options.txt /root/Malmo/Minecraft/run

COPY run.sh /root/
RUN chmod +x /root/run.sh

# Expose port
EXPOSE 10000

# Run Malmo
ENTRYPOINT ["/root/run.sh", "/root/Malmo/Minecraft/launchClient.sh"]