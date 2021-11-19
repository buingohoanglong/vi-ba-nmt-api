from flask import Flask, jsonify, request
from translate import translate as vi_ba_translate

app = Flask(__name__)


@app.route("/translate", methods=['POST'])
def translate():
    data = request.json
    vi = data['text']
    selected_model = data['model']
    ba = vi_ba_translate(vi, selected_model)

    response = {
        'IsSuccessed': True,
        'Message': 'Success',
        'ResultObj': ba
    }

    return jsonify(response)
    
