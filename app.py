from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained RandomForestClassifier model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request payload
        data = request.get_json()
        
        # Extract features
        # Note: If 'Gender' was label-encoded during training (e.g., Male=0, Female=1), 
        # ensure the API client sends it as the matching numeric value.
        gender = data.get('Gender')
        age = data.get('Age')
        salary = data.get('EstimatedSalary')
        
        # Create a DataFrame to maintain the feature names required by the model
        features = pd.DataFrame(
            [[gender, age, salary]], 
            columns=['Gender', 'Age', 'EstimatedSalary']
        )
        
        # Make the prediction
        prediction = model.predict(features)
        
        # Return the prediction result
        return jsonify({'prediction': int(prediction[0])})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/', methods=['GET'])
def health_check():
    return "Model API is up and running!"

if __name__ == '__main__':
    # AWS Elastic Beanstalk often defaults to port 8080
    app.run(host='0.0.0.0', port=8080)
