FROM nvcr.io/nvidia/pytorch:20.12-py3

COPY . .

RUN apt-get update && apt-get install -y \
    libgl1

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "app.py"]