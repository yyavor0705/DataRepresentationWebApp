FROM python:3.7-alpine
LABEL key="DATAPresetner"

ENV PYTHONUNBUFFERED 1

RUN apk add --update --no-cache postgresql-client
RUN apk add --update --no-cache --virtual .tmp-build-deps \
      gcc g++ libc-dev linux-headers postgresql-dev freetype-dev libpng-dev

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

RUN apk del .tmp-build-deps

RUN mkdir /datarepresentationwebapp
WORKDIR /datarepresentationwebapp
COPY datarepresentationwebapp/datarepresentationwebapp /datarepresentationwebapp

RUN adduser -D user
USER user