from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Model aur Vectorizer load kar
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        data = [news]
        vec_data = vectorizer.transform(data)
        prediction = model.predict(vec_data)
        output = "REAL News ✅" if prediction[0] == 1 else "FAKE News ❌"
        return render_template('index.html', prediction_text=f'Result: {output}', news_text=news)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')