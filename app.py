import os
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai

app = Flask(__name__, static_url_path='', static_folder='.')

# In-memory storage for our files
uploaded_context = ""

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    global uploaded_context
    if 'files' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    files = request.files.getlist('files')
    extracted_text = ""
    
    for file in files:
        filename = file.filename
        ext = filename.split('.')[-1].lower()
        
        try:
            if ext in ['txt', 'md', 'csv']:
                # Read text-based formats
                content = file.read().decode('utf-8')
                extracted_text += f"\n--- Content of {filename} ---\n{content}\n"
            elif ext in ['xlsx', 'xls']:
                # Read Excel formats using Pandas
                df = pd.read_excel(file)
                # Convert first 100 rows to string to prevent hitting token limits fast
                content = df.head(100).to_string() 
                extracted_text += f"\n--- Content of {filename} ---\n{content}\n"
            else:
                extracted_text += f"\n[Unsupported file format: {filename}. Please use txt, md, csv, or xlsx]\n"
        except Exception as e:
            print(f"Error reading {filename}: {str(e)}")
            continue

    if extracted_text:
        # Append to our active global context
        uploaded_context += extracted_text
        return jsonify({"message": f"Successfully processed."}), 200
    else:
        return jsonify({"error": "No valid data extracted"}), 400


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    api_key = data.get('api_key')
    query = data.get('query')
    temperature = float(data.get('temperature', 0.7))
    
    if not api_key:
        return jsonify({"error": "No API key provided."}), 400
        
    try:
        # Configure Gemini 
        genai.configure(api_key=api_key)
        
        # System instructions force behavior
        system_instruction = (
            "You are a strict data analysis chatbot. "
            "Answer the user's question USING ONLY the provided Document Context below. "
            "If the answer is not in the context, do not use outside knowledge. Simply reply: 'I cannot find the answer in the provided documents.'"
        )
        
        # Merge our documents directly into the prompt
        context_prompt = f"Document Context:\n{uploaded_context}\n\nUser Question: {query}"
        
        # Initialize Gemini 2.5 Flash
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_instruction,
            generation_config={"temperature": temperature}
        )
        
        response = model.generate_content(context_prompt)
        return jsonify({"response": response.text})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Indigo Logic Backend...")
    app.run(port=5000, debug=True)
