FROM pytorch/pytorch:latest

WORKDIR /work

RUN pip install matplotlib tqdm gif sklearn

ENV PATH=/work/unet-reconnection/bin:$PATH
ENV PYTHONPATH=/work/unet-reconnection:$PYTHONPATH