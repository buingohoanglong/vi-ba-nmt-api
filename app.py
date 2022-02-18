from flask import Flask, jsonify, request
from translate import translate as vi_ba_translate
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret'
# app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app)


@app.route("/translate", methods=['POST'])
def translate():
    data = request.json
    vi = data['text'].split('\n')
    print(vi)
    selected_model = data['model']
    ba = vi_ba_translate(vi, selected_model)
    for idx, value in enumerate(ba):
        ba[idx] = value if vi[idx].strip() != '' else ''

    response = jsonify({
        'IsSuccessed': True,
        'Message': 'Success',
        'ResultObj': {
            'src': vi,
            'tgt': ba
        }
    })
    print(ba)

    # response.headers.add('Access-Control-Allow-Origin', '*')

    return response


@app.errorhandler(Exception)
def handle_error(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    return jsonify(error=str(e)), code
    
