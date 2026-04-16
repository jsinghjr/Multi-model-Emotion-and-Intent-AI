from flask import Flask, request, jsonify
from predict import predict_emotion

app = Flask(__name__)

@app.route("/text", methods=["POST"])
def analyze_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    result = predict_emotion(data["text"])
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)