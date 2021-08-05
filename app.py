from flask import Flask, request, redirect, url_for
import tensorflow as tf
#from tensorflow.python.tf.keras.backend import set_session
import numpy as np
import logging.config
from classificador_inseto import model, predict, predict_processing, Init_Model
VERSAO = "0.1"
app = Flask(__name__)

# Load log configuration and create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('RestAPI')

# INIT MODEL
Init_Model()


@app.route('/')
def home():
    return redirect(url_for('versao'))


@app.route('/api/version')
def versao():
    return {"Versao da API": VERSAO,
            "tensorflow": tf.__version__,
            "CUDA": tf.test.is_built_with_cuda()}


@app.route('/predict', methods=['POST'])
def predic_method():
    try:

        data = request.form
        image_file = request.files['image']

        image_bytes = np.fromstring(image_file.read(), np.uint8)
        resultado = predict(image_bytes)
        print(resultado)
        return {'predicted': resultado}
    except Exception as e:
        logger.exception("ERRO ao processar predict ")
        return ({'score': None, 'result': None, 'status': 'ERRO AO DECODIFICAR JSON!'}, 400)


# def modelo(image):
#     try:
#         with graph.as_default():
#             set_session(sess)
#             predicted = model.predict(image, batch_size=1, verbose=1)[0]*100
#         #predicted = np.round(predicted)
#         return predicted
#     except Exception as e:
#         logger.exception(
#             "ERRO ao processar modelo! ")
#         return None


if __name__ == "__main__":
    app.run(debug=True)
