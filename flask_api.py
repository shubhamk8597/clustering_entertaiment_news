from flask import Flask, request
import pickle
from flasgger import Swagger
from flask import jsonify
import logging
import warnings
import pytz
from datetime import datetime
warnings.filterwarnings("ignore")

# Setting up logging
CET = pytz.timezone('Europe/Berlin')
LOG_DIR = 'logs/'
LOG_FILENAME = f'logfile_api_runs_{str(datetime.now(CET))}.log'
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=LOG_DIR+LOG_FILENAME, format='%(asctime)s %(levelname)-8s %(message)s',level=logging.DEBUG,datefmt='%Y-%m-%d %H:%M:%S')
logging.basicConfig(filename=LOG_DIR+LOG_FILENAME, format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')

# Importing the trained models
try:
    vectorizer = pickle.load(open("kmeans_trained_model/vectorizer.pkl", "rb"))
    kmeans_model = pickle.load(open("kmeans_trained_model/kmeans_entertainment_clustering.sav", 'rb'))
except Exception as ex:
    logging.error(f'Error loading models with --> {ex}')

# Flask implementation
app = Flask(__name__)
Swagger(app)


@app.route('/', methods=["Get"])
def predict_interested_news():
    """Let's Classify the Entertainment News
       ---
       parameters:
         - name: content
           in: query
           type: string
           required: true
       responses:
           200:
                 description: prediction values - 0 and 1 are New Movies Music TV Shows, 2 is Kardashians, 3 is Game of Thrones and 4 is Trump Related entertaiment
    """
    try:
        content = request.args.get('content')
        content = str(content)
        Y = vectorizer.transform([content])
        prediction = kmeans_model.predict(Y)
        if prediction[0] == 0:
            cluster = "New Music, TV Shows and Movies"
        elif prediction[0] == 1:
            cluster = "New Music, TV Shows and Movies"
        elif prediction[0] == 2:
            cluster = "Kardashians"
        elif prediction[0] == 3:
            cluster = "Game of Thrones"
        elif prediction[0] == 4:
            cluster = "Trump Related Entertainment"
        else:
            cluster = "No CLuster Assigned"
        result = {"content": content, "prediction": int(prediction[0]), "cluster": cluster}
        result_json = jsonify(result)
        logging.info(f'Sucessfully Processed Request{result_json} ')
    except Exception as ex:
        logging.error(f'Error while Processing Request --> {ex}')
    return result_json

if __name__ == "__main__":
    app.run()
