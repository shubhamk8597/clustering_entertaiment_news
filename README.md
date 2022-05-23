# Clustering Entertainment News

In this repo we collect data and train kmeans model for clustering the entertainment news.
This is the description for each directory and file.

Directories 

1) Data - This directory contains the dataset from different sources of entertainment news. I have also stored here the combined dataset and also the data set after the cleaning process as csvs.
2) kmeans_trained_model - This model contains the vectorizer and the trained kmeans model.
3) logs - This directory will have the logs for individual runs of flask_api.py
4) scraped_Data_for_test - This contains the scraped data from google news that we label for testing further

Files

1) clustering_entertainment_news.ipynb - Jupyter notebook that has steps from data preparation to model evaluation.
2) deploy_script.sh - Deployment script to deploy flaks_api locally in docker.
3) Dockerfile - To build the docker container.
4) flask_api.py - The flask_api to run our model locally

Steps to build container and run on your machine

1) Clone the repo. 
2) Make changes to the deploy_Script.sh . Specify the correct path to the logs directory for mounting in docker run command.
3) Run ./deploy_script.sh. This will build the docker container and run it.
4) You can now access the flask api on your browser with docker_cointainer_ip:Exposed_port. For me its 172.17.0.2:5000/apidocs/
5) Click on GET --> Try it out --> paste the text you want to classify in the content tab.
6) Check logs for more details
