FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

ENV MODEL_NAME=facebook/esm2_t6_8M_UR50D

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:80", "app:app"]
