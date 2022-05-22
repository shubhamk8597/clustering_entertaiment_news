FROM python:3.8-slim

EXPOSE 5000

ENV FLASK_APP=flask_api.py

WORKDIR /clustering_entertainment_news

COPY ./kmeans_trained_model ./kmeans_trained_model

COPY ./requirements.txt ./requirements.txt

COPY ./flask_api.py ./flask_api.py

RUN pip install -r requirements.txt

CMD ["flask", "run", "--host", "0.0.0.0" ]