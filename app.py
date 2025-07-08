from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load(open('model.pkl', 'rb'))

# Home route
@app.route('/')
def home():
    return render_template('main.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    pl = float(request.form['pl'])
    pw = float(request.form['pw'])

    prediction = model.predict(np.array([[pl, pw]]))
    flower = ['Setosa', 'Versicolor', 'Virginica'][prediction[0]]

    return render_template('main.html', result=f"The flower is: {flower}")

if __name__ == '__main__':
    app.run(debug=True)
