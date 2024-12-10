from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization

# Load your trained model
model = tf.keras.models.load_model('history.keras')

# Initialize TextVectorization layer with the same settings used during training
MAX_FEATURES = 2000000
vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode='int')

# Load dataset to adapt vectorizer (use the same dataset or equivalent as during training)
import pandas as pd
df = pd.read_csv("C:\\Users\\sampa\\Downloads\\DTL\\train.csv")
X = df['comment_text']
vectorizer.adapt(X.values)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['text']
        # Preprocess the input text using the vectorizer
        processed_text = vectorizer(tf.constant([input_text]))  # Vectorize the input text

        # Make prediction
        prediction = model.predict(processed_text)[0]

        # Labels corresponding to the columns in your dataset
        labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

        # Filter labels with predictions greater than 0.5
        results = [label for label, pred in zip(labels, prediction) if pred > 0.5]

        return render_template('result.html', text=input_text, results=results)

if __name__ == '__main__':
    app.run(debug=True)
