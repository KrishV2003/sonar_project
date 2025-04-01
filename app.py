from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and the saved scaler
model = joblib.load('best_svm_model.pkl')
scaler = joblib.load('scaler.pkl')  # Load the scaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        f0 = float(request.form['f0'])
        f1 = float(request.form['f1'])

        # Create a full feature array with 60 features
        input_data = np.zeros(60)
        input_data[0] = f0  # Assign f0 to the first feature
        input_data[1] = f1  # Assign f1 to the second feature

        # Scale the input data using the loaded scaler
        input_data_scaled = scaler.transform(input_data.reshape(1, -1))

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Prepare result
        result = "The object is a Rock" if prediction[0] == 'R' else "The object is a Mine"
        
        return render_template('result.html', prediction=result)
    
    except ValueError as e:
        return render_template('index.html', error="Invalid input. Please enter valid numbers.")
    except Exception as e:
        return render_template('index.html', error="An error occurred: " + str(e))


if __name__ == '__main__':
    app.run(debug=True)
