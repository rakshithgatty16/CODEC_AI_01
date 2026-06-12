from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('weather_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    humidity = float(request.form['humidity'])
    windspeed = float(request.form['windspeed'])
    pressure = float(request.form['pressure'])

    prediction = model.predict(
        [[humidity, windspeed, pressure]]
    )

    return render_template(
        'index.html',
        prediction_text=f'Predicted Temperature: {prediction[0]:.2f} °C'
    )

if __name__ == '__main__':
    app.run(debug=True)