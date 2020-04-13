from flask import Flask,render_template,url_for,request,jsonify
from inference import convert_to_vector
from keras import models
from tensorflow.python.keras.backend import set_session
from keras_multi_head import MultiHeadAttention
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
import pickle
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load pickle
word2idx = pickle.load(open("model/word2idx.pkl","rb"))
char2idx = pickle.load(open("model/char2idx.pkl","rb"))
label2idx = pickle.load(open("model/label2idx.pkl","rb"))
idx2label = {v:k for k,v in label2idx.items()}

# maximum character per word
maxwordlength = 15

# Load Model
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)
model = models.load_model("model/ner_warung.h5",
                          custom_objects = {'CRF':CRF,'crf_loss':crf_loss,
                                  'crf_accuracy':crf_accuracy,'MultiHeadAttention':MultiHeadAttention})
model.summary()
print('Loaded the model')
    
## Predict data
def predict(dataset):
    predLabels = []
    
    for i, data in enumerate(dataset):
        tokens, char= data
        tokens = np.asarray([tokens])
        char = np.asarray([char])       
        pred = model.predict([tokens, char], verbose = False)[0]    
        predLabels.append([np.argmax(i) for i in pred])
        
    return predLabels
    
    
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/prediction',methods=["POST"])
def prediction():
    if request.method == 'POST':
        # get text from web app
        rawtext = request.form['rawtext']
        rawtext = rawtext.rstrip().splitlines()
        global sess
        global graph
        with graph.as_default():
            set_session(sess)
            result = []
            for line in rawtext:
                # convert to vector
                tmp = convert_to_vector(line, word2idx, char2idx, maxwordlength)
                # predict
                ans = predict([tmp])[0]
                ans = [idx2label[i] for i in ans]
                # create result format
                res = []
                res.append('order : '+str(line))
                res.append('entity : '+str(ans))
                result.append(res)

    return render_template("index.html",results = result, num_of_results = len(rawtext))



@app.route('/result',methods=['POST','GET'])
def result():
    data = request.get_json(force=True)
    data = data['text'].rstrip().splitlines()
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        result = []
        for line in data:
            # convert to vector
            tmp = convert_to_vector(line, word2idx, char2idx, maxwordlength)
            # predict
            ans = predict([tmp])[0]
            ans = [idx2label[i] for i in ans]
            # create result format
            res = []
            res.append('order : '+str(line))
            res.append('entity : '+str(ans))
            result.append(res)

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)