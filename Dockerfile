FROM malmopy-cntk-cpu-py27:latest

ADD ai_challenge ai_challenge
RUN pip install chainerrl
WORKDIR /root/malmo-challenge/ai_challenge