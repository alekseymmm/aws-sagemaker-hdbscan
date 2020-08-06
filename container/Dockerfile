FROM python:slim

LABEL maintainer="am <alekseymmm@mail.ru>"
LABEL dlc_major_version="1"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
 && pip3 install --no-cache-dir -U \
    numpy \
    pandas \
    hdbscan
    
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY hdbscan /opt/program
WORKDIR /opt/
ENTRYPOINT ["train.py"]