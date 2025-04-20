import os
from flask import Flask, request, jsonify
from gradio_client import Client, handle_file
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Flask app
app = Flask(__name__)

# Initialize Hugging Face client for handwriting recognition
client = Client("Hammedalmodel/handwritten_to_text")

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Flask API is running successfully!"}), 200

@app.route('/extract_text', methods=['POST'])
def extract_text():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    result = client.predict(image=handle_file(image), api_name="/predict")
    
    return jsonify({"extracted_text": result}), 200

@app.route('/ask_gemini', methods=['POST'])
def ask_gemini():
    data = request.get_json()
    
    if not data or 'text' not in data or 'question' not in data:
        return jsonify({"error": "Invalid input"}), 400
    
    prompt = f"Given the following text:\n\n{data['text']}\n\nAnswer the question: {data['question']}"
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    return jsonify({"answer": response.text}), 200

if __name__ == '__main__':
    app.run(debug=True)
