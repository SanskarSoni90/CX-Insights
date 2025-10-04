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
openai_client = None
pc = None
index = None

try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    print("Successfully initialized OpenAI and Pinecone clients.")
except Exception as e:
    print(f"ERROR: Could not initialize clients during startup: {e}")

# --- API Endpoints ---

@app.route('/')
def serve_index_page():
    """Serves the main HTML file."""
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if not openai_client or not index:
        return jsonify({"answer": "Sorry, the connection to the AI services failed. Please check the server configuration and logs on Render."})

    try:
        user_query = request.json.get('question')
        if not user_query:
            return jsonify({"error": "No question provided"}), 400

        # Check if user is asking for total count/stats
        count_keywords = ['how many', 'total', 'count', 'number of calls', 'all calls']
        is_count_query = any(keyword in user_query.lower() for keyword in count_keywords)
        
        EMBEDDING_MODEL = "text-embedding-3-large"
        PINECONE_DIMENSION = 1024
        
        # 1. Create an embedding for the user's query
        response = openai_client.embeddings.create(
            input=[user_query], 
            model=EMBEDDING_MODEL,
            dimensions=PINECONE_DIMENSION
        )
        query_embedding = response.data[0].embedding

        # 2. Query Pinecone to find the most relevant calls
        # Use more results for count queries, fewer for specific inquiries
        top_k = 200 if is_count_query else 50
        
        query_results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # For count queries, also get total record count from Pinecone
        if is_count_query:
            # Get index stats to show total records
            stats = index.describe_index_stats()
            total_records = stats.get('total_vector_count', 'Unknown')
            context = f"Total records in database: {total_records}\n\nSample of relevant calls:\n"
        else:
            context = "Call Records:\n"
        # 3. Build a concise context from the results
        if not query_results['matches']:
            return jsonify({"answer": "No relevant call records found for this query."})
        
        for i, match in enumerate(query_results['matches'], 1):
            metadata = match['metadata']
            context += (
                f"{i}. Date: {metadata.get('date', 'N/A')} | "
                f"Summary: {metadata.get('summary', 'N/A')} | "
                f"Sentiment: {metadata.get('sentiment', 'N/A')}\n"
            )

        # 4. Create the refined prompt for data-heavy, professional responses
        final_prompt = f"""Context:
{context}

Question: {user_query}

RESPONSE GUIDELINES:
• Be direct and concise - no fluff or introductory phrases
• Lead with numbers and data points immediately
• Use bullet points for clarity
• Maximum 4-5 sentences or bullet points
• Quote specific calls when relevant (e.g., "Call 3: [quote]")
• If data is insufficient, state it in one sentence

Format:
[Direct answer with key metric] + [2-3 supporting data points] + [Brief insight/recommendation if applicable]

Example:
Q: "What are main complaints?"
A: "Billing issues dominate (3/5 calls, 60%). Call 2: 'Double charged for subscription.' Call 4: 'Unexpected renewal fee.' Recommend: Audit billing system for duplicate charges and improve renewal notifications."

Your answer:"""
        
        # 5. Call the LLM with the refined prompt
        chat_response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a data analyst. Provide short, metric-heavy answers with no fluff."},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.3,  # Lower temperature for more focused, consistent responses
            max_tokens=300  # Limit response length to keep it concise
        )
        
        ai_answer = chat_response.choices[0].message.content
        return jsonify({"answer": ai_answer})

    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return jsonify({"answer": "Sorry, an unexpected error occurred while processing your request."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
