import os
import json
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from pinecone import Pinecone

# --- Configuration ---
# Ensure these are set in your Render environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY_CHAT')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT') # e.g., 'gcp-starter' or 'us-east-1'
PINECONE_INDEX_NAME = 'call-insights' # The name of your index in Pinecone

# --- Initialize Clients ---
# It's best practice to initialize clients once when the app starts
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    print("Successfully connected to OpenAI and Pinecone.")
except Exception as e:
    print(f"ERROR: Could not initialize clients: {e}")
    openai_client = None
    pinecone_index = None

# Initialize Flask App
app = Flask(__name__, static_folder='.', static_url_path='')

# --- API Endpoints ---

@app.route('/')
def index():
    """Serves the main HTML file."""
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat requests using Pinecone for semantic search."""
    if not openai_client or not pinecone_index:
        return jsonify({"error": "Backend services are not configured."}), 503

    user_question = request.json.get('question')
    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # 1. Create an embedding for the user's query
        embedding_model = "text-embedding-3-small"
        response = openai_client.embeddings.create(input=[user_question], model=embedding_model)
        query_embedding = response.data[0].embedding

        # 2. Query Pinecone to find the top 5 most relevant calls
        query_results = pinecone_index.query(
            vector=query_embedding,
            top_k=5,  # Fetch the 5 most semantically similar records
            include_metadata=True
        )

        # 3. Build a concise context from the search results
        context = "Relevant call records:\n"
        if not query_results['matches']:
            context = "No relevant call records found for this query."
        else:
            for match in query_results['matches']:
                metadata = match.get('metadata', {})
                context += (
                    f"- Call Date: {metadata.get('date', 'N/A')}\n"
                    f"  Sentiment: {metadata.get('sentiment', 'N/A')}\n"
                    f"  Summary: {metadata.get('summary', 'N/A')}\n\n"
                )

        # 4. Create the final prompt for the LLM
        system_prompt = f"""
        You are an expert customer support data analyst.
        Answer the user's question based ONLY on the context from the relevant call records provided below.
        Do not make up information. If the answer isn't in the context, state that clearly.

        Context:
        {context}
        """

        # 5. Call the LLM with the new, focused prompt
        chat_response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview", # A powerful model for analysis
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ]
        )
        
        ai_answer = chat_response.choices[0].message.content
        return jsonify({"answer": ai_answer})

    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == '__main__':
    # This is for local development. Render will use Gunicorn.
    app.run(host='0.0.0.0', port=8080)
