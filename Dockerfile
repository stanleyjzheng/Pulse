FROM nvidia/cuda:11.2.0-base
FROM python:3.8

COPY . .

RUN echo $PWD

RUN apt-get update && apt-get install -y \
    libgl1

RUN echo $(ls)

RUN python3 -m pip install -r requirements.txt

CMD ["streamlit", "run", "app.py"]