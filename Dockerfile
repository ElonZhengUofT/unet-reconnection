FROM pytorch/pytorch:latest

WORKDIR /work

RUN pip install matplotlib tqdm gif

USER dev

ENV PATH=/work/2d-reconnection/bin:$PATH
ENV PYTHONPATH=/work/2d-reconnection