import os
import json
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from pinecone import Pinecone

# --- Configuration ---
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
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

# --- Helper Functions ---

def classify_query_intent(query):
    """Use LLM to classify query intent for better routing."""
    classification_prompt = f"""Classify this query into ONE category:

Query: "{query}"

Categories:
1. AGGREGATE - Questions about totals, counts, percentages, trends, distributions, averages
   Examples: "How many calls?", "What percentage?", "Show trends", "Average satisfaction"

2. SPECIFIC - Questions about particular calls, detailed transcripts, specific customer issues
   Examples: "Tell me about call X", "What did the customer say?", "Show me the transcript"

3. COMPARISON - Questions comparing data across time periods or categories
   Examples: "Compare this week vs last week", "Billing vs technical issues"

4. SEARCH - Looking for calls matching certain criteria
   Examples: "Find all negative calls", "Show calls about billing", "Urgent unresolved issues"

Respond with ONLY the category name (AGGREGATE, SPECIFIC, COMPARISON, or SEARCH)."""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a query classifier. Respond with only the category name."},
                {"role": "user", "content": classification_prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        return response.choices[0].message.content.strip().upper()
    except:
        # Fallback to keyword matching
        query_lower = query.lower()
        if any(kw in query_lower for kw in ['how many', 'total', 'count', 'percentage', 'average', 'trend']):
            return 'AGGREGATE'
        elif any(kw in query_lower for kw in ['specific', 'particular', 'transcript', 'tell me about']):
            return 'SPECIFIC'
        elif any(kw in query_lower for kw in ['compare', 'vs', 'versus', 'difference between']):
            return 'COMPARISON'
        else:
            return 'SEARCH'


def query_with_strategy(query_embedding, intent):
    """Query Pinecone with strategy based on intent."""
    
    if intent == 'AGGREGATE':
        # Use aggregate namespace + sample of regular calls
        agg_results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            namespace="aggregates"
        )
        
        sample_results = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )
        
        return {'aggregates': agg_results, 'samples': sample_results}
    
    elif intent == 'SPECIFIC':
        # Detailed retrieval of few calls
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        return {'specific': results}
    
    elif intent == 'COMPARISON':
        # Get aggregates + more samples for comparison
        agg_results = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True,
            namespace="aggregates"
        )
        
        sample_results = index.query(
            vector=query_embedding,
            top_k=30,
            include_metadata=True
        )
        
        return {'aggregates': agg_results, 'samples': sample_results}
    
    else:  # SEARCH
        # Standard search with moderate results
        results = index.query(
            vector=query_embedding,
            top_k=20,
            include_metadata=True
        )
        return {'search': results}


def build_context(query_results, intent):
    """Build optimized context based on intent."""
    
    if intent == 'AGGREGATE':
        context = "=== PRE-COMPUTED STATISTICS ===\n"
        
        # Add aggregate summaries
        if 'aggregates' in query_results and query_results['aggregates']['matches']:
            for match in query_results['aggregates']['matches']:
                metadata = match.get('metadata', {})
                context += f"\n{metadata.get('summary', '')}\n"
        
        # Add compact sample examples
        context += "\n=== SAMPLE EXAMPLES ===\n"
        if 'samples' in query_results and query_results['samples']['matches']:
            for i, match in enumerate(query_results['samples']['matches'][:5], 1):
                m = match.get('metadata', {})
                context += f"{i}. [{m.get('date', 'N/A')}] {m.get('sentiment', 'N/A')} - {m.get('primary_issue', 'N/A')}\n"
        
        return context
    
    elif intent == 'SPECIFIC':
        context = "=== DETAILED CALL INFORMATION ===\n\n"
        
        if 'specific' in query_results:
            for i, match in enumerate(query_results['specific']['matches'], 1):
                m = match.get('metadata', {})
                context += f"""Call #{i} (Relevance: {match.get('score', 0):.2f})
Date: {m.get('date', 'N/A')}
Issue: {m.get('primary_issue', 'N/A')}
Category: {m.get('issue_category', 'N/A')}
Sentiment: {m.get('sentiment', 'N/A')}
Urgency: {m.get('urgency', 'N/A')}
Resolution: {m.get('resolution_status', 'N/A')}
Summary: {m.get('summary', 'N/A')}
Transcript Snippet: {m.get('transcript_snippet', 'N/A')}
Action Items: {m.get('action_items', 'N/A')}

"""
        return context
    
    elif intent == 'COMPARISON':
        context = "=== AGGREGATE DATA FOR COMPARISON ===\n"
        
        if 'aggregates' in query_results and query_results['aggregates']['matches']:
            for match in query_results['aggregates']['matches']:
                metadata = match.get('metadata', {})
                context += f"\n{metadata.get('summary', '')}\n"
        
        context += "\n=== SAMPLE DATA POINTS ===\n"
        if 'samples' in query_results and query_results['samples']['matches']:
            for i, match in enumerate(query_results['samples']['matches'][:15], 1):
                m = match.get('metadata', {})
                context += f"{i}. [{m.get('date', 'N/A')}] {m.get('issue_category', 'N/A')} | {m.get('sentiment', 'N/A')} | Satisfaction: {m.get('customer_satisfaction', 'N/A')}\n"
        
        return context
    
    else:  # SEARCH
        context = f"=== SEARCH RESULTS ({len(query_results.get('search', {}).get('matches', []))} calls found) ===\n\n"
        
        if 'search' in query_results:
            for i, match in enumerate(query_results['search']['matches'], 1):
                m = match.get('metadata', {})
                context += f"""{i}. {m.get('date', 'N/A')} | {m.get('sentiment', 'N/A')}
   Issue: {m.get('primary_issue', 'N/A')}
   Summary: {m.get('summary', 'N/A')[:150]}...
   
"""
        return context


# --- API Endpoints ---

@app.route('/')
def serve_index_page():
    """Serves the main HTML file."""
    return send_from_directory('.', 'index.html')


@app.route('/chat', methods=['POST'])
def chat():
    if not openai_client or not index:
        return jsonify({"answer": "Sorry, the connection to the AI services failed. Please check the server configuration."})

    try:
        user_query = request.json.get('question')
        if not user_query:
            return jsonify({"error": "No question provided"}), 400

        print(f"\n{'='*60}")
        print(f"Query: {user_query}")
        
        # Step 1: Classify query intent
        intent = classify_query_intent(user_query)
        print(f"Intent: {intent}")
        
        # Step 2: Create embedding
        EMBEDDING_MODEL = "text-embedding-3-large"
        PINECONE_DIMENSION = 1024
        
        response = openai_client.embeddings.create(
            input=[user_query], 
            model=EMBEDDING_MODEL,
            dimensions=PINECONE_DIMENSION
        )
        query_embedding = response.data[0].embedding
        
        # Step 3: Query with appropriate strategy
        query_results = query_with_strategy(query_embedding, intent)
        
        # Check if any results found
        has_results = False
        for key, value in query_results.items():
            if isinstance(value, dict) and value.get('matches'):
                has_results = True
                break
        
        if not has_results:
            return jsonify({"answer": "No relevant call records found for this query."})
        
        # Step 4: Build optimized context
        context = build_context(query_results, intent)
        print(f"Context length: {len(context)} characters")
        
        # Step 5: Create intent-specific prompt
        system_prompts = {
            'AGGREGATE': "You are a data analyst. Provide metric-heavy answers with statistics and percentages. Be concise.",
            'SPECIFIC': "You are a call center analyst. Provide detailed information about specific calls. Include relevant quotes.",
            'COMPARISON': "You are a comparative analyst. Highlight differences, trends, and insights across data segments.",
            'SEARCH': "You are a search result summarizer. Present findings clearly with key details from matching calls."
        }
        
        final_prompt = f"""Context:
{context}

Question: {user_query}

Instructions:
• Answer directly based on the data provided
• Use specific numbers and metrics
• Keep response under 200 words
• If comparing, highlight key differences
• Quote relevant details when applicable

Answer:"""
        
        # Step 6: Get LLM response
        chat_response = openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompts.get(intent, system_prompts['SEARCH'])},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.3,
            max_tokens=400
        )
        
        ai_answer = chat_response.choices[0].message.content
        print(f"Response generated successfully")
        print(f"{'='*60}\n")
        
        return jsonify({"answer": ai_answer})

    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"answer": "Sorry, an unexpected error occurred while processing your request."})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
