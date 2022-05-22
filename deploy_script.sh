#!/usr/bin/env bash

echo "Deploying Locally.."
echo "Stoping and removing previous containers"
docker stop clustering-entertainment-news || true && docker rm clustering-entertainment-news || true
echo "Creating docker image..."
docker build -t clustering_entertainment_news .
echo "Built Sucessfully"
echo "Running locally at port 5000"
docker run --name clustering-entertainment-news -m 0.5g --cpus=0.5 -v /home/shubham/SVK/clustering_entertaiment_news/logs:/clustering_entertainment_news/logs -p 5000:5000 clustering_entertainment_news
