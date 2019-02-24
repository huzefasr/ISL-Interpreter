from flask import Flask,redirect,render_template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_gesture')
def add_gesture():
    return render_template('add_gesture.html')

@app.route('/train_model')
def train_model():
    return render_template('train_model.html')

@app.route('/recognize_gesture')
def recognize_gesture():
    return render_template('recognize_gesture.html')

if __name__ == '__main__':
    app.run(debug=True)
