# pytorch-2302-py3-llama:v1.0.0
FROM nvcr.io/nvidia/pytorch:23.02-py3

COPY requirements.txt /tmp/requirements.txt

RUN pip3 install -r /tmp/requirements.txt
