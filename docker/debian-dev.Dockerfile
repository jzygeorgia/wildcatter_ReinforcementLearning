ARG NAME=wildcatter
ARG TAG=dev
ARG VIRTUAL_ENV=/opt/venv

FROM python:3.9-slim-bullseye as builder
ARG VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN apt-get update
RUN apt-get install -y  \
    swig \
    build-essential

RUN python -m venv $VIRTUAL_ENV
RUN pip -q install pip --upgrade
RUN pip install \
    numpy \
    matplotlib \
    jupyterlab

COPY . wildcatter
WORKDIR /wildcatter
RUN pip install .

FROM python:3.9-slim-bullseye
ARG VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV
