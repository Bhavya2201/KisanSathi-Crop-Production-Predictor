import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import gzip, pickletools

app = Flask(__name__)
#model = pickle.load(open('pickle_dump1.pkl', 'rb'))

with gzip.open('pickle_dump1.pkl', 'rb') as f:
    p = pickle.Unpickler(f)
    clf = p.load()
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = clf.predict(final_features) 

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Production prediction is {} tonnes/hectare'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
