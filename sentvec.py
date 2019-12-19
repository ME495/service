import os
import sys
sys.path.append('/data02/chengjian19/models/research/skip_thoughts')
import numpy as np
import os.path
import scipy.spatial.distance as sd
from skip_thoughts import configuration
from skip_thoughts import encoder_manager
from flask import Flask, request
import pickle

app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

VOCAB_FILE = "/data02/chengjian19/skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/vocab.txt"
EMBEDDING_MATRIX_FILE = "/data02/chengjian19/skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/embeddings.npy"
CHECKPOINT_PATH = "/data02/chengjian19/skip_thoughts/pretrained/skip_thoughts_bi_2017_02_16/model.ckpt-500008"

encoder = encoder_manager.EncoderManager()
encoder.load_model(configuration.model_config(bidirectional_encoder=True),
                   vocabulary_file=VOCAB_FILE,
                   embedding_matrix_file=EMBEDDING_MATRIX_FILE,
                   checkpoint_path=CHECKPOINT_PATH)

@app.route('/', methods=['POST'])
def sentenc2vector():
    sentence = request.form['sentence'].encode()
    vector = encoder.encode([sentence])
    print vector.shape
    data = ','.join(list(map(str, vector[0].tolist())))
    return data, 200

if __name__ == '__main__':
    app.run(
      host='12.12.12.2',
      port= 7005,
      debug=True
    )