from flask import Flask,render_template,request
import pandas as pd
import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle
from review_generator import review_generator
from cosine import cos_similarity 
import joblib
pkl_path = './static/pkls/'
db_path = './static/db/'
negative_model = tf.keras.models.load_model('./static/pkls/negative_model.h5')
positive_model = tf.keras.models.load_model('./static/pkls/positive_model.h5')
with open('./static/pkls/negative_tokenizer.pkl', 'rb') as f:
    negative_tokenizer = pickle.load(f)
with open('./static/pkls/positive_tokenizer.pkl', 'rb') as f:
    positive_tokenizer = pickle.load(f)

sample_df = pd.read_csv('./static/data/csvs/sample.csv')
app = Flask(__name__)

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/dark', methods=['GET', 'POST'])
def dark_page():
    adverses = ['긍정', '부정']
    if request.method == 'POST':
        adverse = request.form['dropdown']
        query = request.form['query']
        length = request.form['length']
        length = int(length)
        if adverse == '부정':
            model = negative_model
            tokenizer = negative_tokenizer
            result = review_generator(model, tokenizer, query, length)
            return render_template('dark_result.html', result=result, adverse=adverse, query=query)
        else:
            model = positive_model
            tokenizer = positive_tokenizer
            result = review_generator(model, tokenizer, query, length)
            return render_template('dark_result.html', result=result, adverse=adverse, query=query)
    return render_template('dark.html', adverses=adverses)

@app.route('/light', methods=['GET', 'POST'])
def light_page():
    date = datetime.datetime.now()
    locations = ['대구', '광주']
    if request.method == 'POST':
        location = request.form['dropdown_loc']
        cos_sim = joblib.load(pkl_path+location+'.pkl')
        data_path = db_path+location+'.csv'
        df = pd.read_csv(data_path)
        return render_template('light_result.html', location=location,sample_df = sample_df, date=date, df=df, data_path=data_path, cos_sim=cos_sim)
    return render_template('light.html', locations=locations)


if __name__ == '__main__':
    app.run(debug=False)