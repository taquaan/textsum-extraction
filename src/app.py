from flask import (Flask, request, jsonify, render_template)
from summaryText import summary_text
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
   data = request.get_json()
   original_text = data['text']
   summarized_text = summary_text(original_text)
   return jsonify({'summarized_text': summarized_text})
   

if __name__ == '__main__':
  app.run(debug=True)