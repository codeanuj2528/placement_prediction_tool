from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get CGPA and IQ from the form
    cgpa = float(request.form['cgpa'])
    iq = float(request.form['iq'])
    
    # Make prediction
    features = np.array([[cgpa, iq]])
    prediction = model.predict(features)[0]
    
    # Assuming the model returns a probability
    probability = prediction * 100  # Convert to percentage
    
    return render_template('result.html', prediction=probability)

if __name__ == '__main__':
    app.run(debug=True)