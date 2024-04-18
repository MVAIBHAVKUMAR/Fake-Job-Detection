import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load the saved model and vectorizer
with open("C:\\Users\\mvaib\\OneDrive - rupee\\Desktop\\Fake Job Detection\\job_fraud_model.pkl", 'rb') as file:
    model = pickle.load(file)
with open("C:\\Users\\mvaib\\OneDrive - rupee\\Desktop\\Fake Job Detection\\job_fraud_vectorizer.pkl", 'rb') as file:
    vectorizer = pickle.load(file)

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# API route to receive a job description and return a prediction
@app.route('/predict', methods=['POST'])




def predict_fraud():
    job_description = request.json['description']  # Extract the job description

    # Preprocess the input text
    input_features = vectorizer.transform([job_description])

    # Make a prediction using the loaded model
    prediction = model.predict(input_features)[0]

    # Convert numerical prediction to text
    if prediction == 1:
        result = 'Fraudulent Job'
    else:
        result = 'Real Job'

    # Return the result as a JSON response
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True) 
