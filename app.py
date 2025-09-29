import os
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import openai
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd

# --- Configuration ---
# It's better to load these from environment variables in a real app
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY_CHAT') # Use a separate secret for the chat app
GOOGLE_SHEET_ID = '1LGeqJTaX6IfjHz2-H1YGzuRV3FM22P33Uqx2nwp5Zjs'
GOOGLE_SHEET_NAME = 'Sheet1'
GOOGLE_CREDENTIALS_JSON = os.getenv('GOOGLE_CREDENTIALS_JSON_CHAT')

# Initialize Flask App
app = Flask(__name__, static_folder='.', static_url_path='')

# --- Google Sheets Connection ---
def get_sheet_data():
    """Fetches all data from the Google Sheet and returns a pandas DataFrame."""
    try:
        creds_json = json.loads(GOOGLE_CREDENTIALS_JSON)
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)
        client = gspread.authorize(creds)
        
        sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
        records = sheet.get_all_records()
        return pd.DataFrame(records)
    except Exception as e:
        print(f"Error fetching Google Sheet data: {e}")
        return pd.DataFrame() # Return empty dataframe on error

# --- API Endpoints ---

@app.route('/')
def index():
    """Serves the main HTML file."""
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles the chat request, queries the sheet, and returns an AI-generated answer."""
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # 1. Get the latest data from the Google Sheet
        df = get_sheet_data()
        if df.empty:
            return jsonify({"answer": "Sorry, I could not retrieve any data to answer your question."})

        # 2. Convert dataframe to a string format for the prompt
        # Using markdown format for clarity
        data_string = df.to_markdown(index=False)
        
        # 3. Query OpenAI
        openai.api_key = OPENAI_API_KEY
        
        system_prompt = f"""
        You are a helpful assistant and an expert in customer support data analysis.
        Your task is to answer the user's question based ONLY on the provided call data.
        Do not make up information. If the answer is not in the data, say so.
        Analyze the data to provide insights, summaries, counts, or trends as requested.

        Here is the call data:
        {data_string}
        """

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ]
        )
        
        ai_answer = response.choices[0].message.content
        return jsonify({"answer": ai_answer})

    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == '__main__':
    # Use Gunicorn or another WSGI server in production
    app.run(host='0.0.0.0', port=8080)
