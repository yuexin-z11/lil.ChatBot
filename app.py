# import necessary libraries
from flask import Flask, render_template, jsonify, request
import processor

# initialize flask application
app = Flask(__name__)
# for flask to maintain sessions securely
app.config['SECRET_KEY'] = secrets.token_hex(24)

# defines a route for root 
@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())

# route for charbot for get and post reqests
@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():

    if request.method == 'POST':
        the_question = request.form['question']

        response = processor.chatbot_response(the_question)

    return jsonify({"response": response })



if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8888', debug=True)