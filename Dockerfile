FROM python:3.7-slim-buster
LABEL key="DATAPresetner"
ENV PYTHONUNBUFFERED 1
RUN apt-get update && apt-get install -y \
  gcc \
  libpq-dev
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
RUN mkdir /datarepresentationwebapp
WORKDIR /datarepresentationwebapp
COPY datarepresentationwebapp/datarepresentationwebapp /datarepresentationwebapp
RUN adduser user
USER user