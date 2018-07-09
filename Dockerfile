FROM ubuntu:17.10

RUN apt-get update
RUN apt-get install -y python3-pip
RUN apt-get install -y git
RUN apt-get upgrade -y

COPY . /app
WORKDIR /app

RUN pip3 install -r ./app/requirements.txt

