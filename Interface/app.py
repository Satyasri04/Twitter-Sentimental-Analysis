from flask import Flask, render_template, request
from sentiment_analysis import test_analizer


app = Flask(__name__)
@app.route('/')

def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    input_text = request.form['input_text']
    if input_text:
        result = test_analizer(input_text)
    else:
        result = "Please enter some text for analysis."
    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)