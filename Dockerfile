FROM nvcr.io/nvidia/pytorch:24.05-py3-igpu

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/