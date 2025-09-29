import os
import json
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from pinecone import Pinecone

# --- Configuration ---
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX_NAME = 'call-insights'

# Initialize Flask App
app = Flask(__name__, static_folder='.', static_url_path='')

# --- Initialize Clients ---
# Initialize client variables to None outside the try block
openai_client = None
pc = None
index = None

try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # This is the Pinecone index object
    index = pc.Index(PINECONE_INDEX_NAME)
    print("Successfully initialized OpenAI and Pinecone clients.")
except Exception as e:
    print(f"ERROR: Could not initialize clients during startup: {e}")

# --- API Endpoints ---

# FIX: Renamed the function from 'index' to 'serve_index_page' to avoid name collision
@app.route('/')
def serve_index_page():
    """Serves the main HTML file."""
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    # Check if clients were successfully initialized before using them
    if not openai_client or not index:
        # Return a user-facing error in the expected format for the frontend
        return jsonify({"answer": "Sorry, the connection to the AI services failed. Please check the server configuration and logs on Render."})

    try:
        user_query = request.json.get('question')
        if not user_query:
            return jsonify({"error": "No question provided"}), 400

        # Use the same model and dimension as the indexing script
        EMBEDDING_MODEL = "text-embedding-3-large"
        PINECONE_DIMENSION = 1024
        
        # 1. Create an embedding for the user's query
        response = openai_client.embeddings.create(
            input=[user_query], 
            model=EMBEDDING_MODEL,
            dimensions=PINECONE_DIMENSION
        )
        query_embedding = response.data[0].embedding

        # 2. Query Pinecone to find the top 5 most relevant calls
        query_results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )

        # 3. Build a concise context from the results
        if not query_results['matches']:
            return jsonify({"answer": "No relevant call records found for this query."})

        context = "Based on the call records, here is the relevant information:\n"
        for match in query_results['matches']:
            metadata = match['metadata']
            context += (
                f"- Call on {metadata.get('date', 'N/A')}: "
                f"Summary: {metadata.get('summary', 'N/A')} "
                f"Sentiment: {metadata.get('sentiment', 'N/A')}\n"
            )

        # 4. Create the final prompt for the LLM
        final_prompt = f"""
        Context from relevant calls:
        {context}

        User's Question: {user_query}

        Based only on the context provided, answer the user's question.
        """
        
        # 5. Call the LLM with the new, smaller prompt
        chat_response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant analyzing call center data."},
                {"role": "user", "content": final_prompt}
            ]
        )
        
        ai_answer = chat_response.choices[0].message.content
        return jsonify({"answer": ai_answer})

    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        # Return a user-facing error in the expected format
        return jsonify({"answer": "Sorry, an unexpected error occurred while processing your request."})

if __name__ == '__main__':
    # Use Gunicorn or another WSGI server in production
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

