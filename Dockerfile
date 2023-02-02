FROM nvcr.io/nvidia/pytorch:22.11-py3

WORKDIR /work

RUN pip install matplotlib tqdm gif scikit-learn

ENV PATH=/work/unet-reconnection/bin:$PATH
ENV PYTHONPATH=/work/unet-reconnection:$PYTHONPATH