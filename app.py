from flask import Flask, request, jsonify
from transformers import AutoModel, AutoTokenizer
import torch

app = Flask(__name__)

# Load the model and tokenizer
model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

@app.route('/predict/', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data['input_text']
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model(**inputs)
    return jsonify({"embeddings": outputs.last_hidden_state.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
