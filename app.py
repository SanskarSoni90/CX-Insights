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
        intent = response.choices[0].message.content.strip().upper()
        print(f"LLM classified intent as: {intent}")
        return intent
    except Exception as e:
        print(f"ERROR in LLM classification: {e}")
        # Fallback to keyword matching
        query_lower = query.lower()
        if any(kw in query_lower for kw in ['how many', 'total', 'count', 'percentage', 'average', 'trend']):
            intent = 'AGGREGATE'
        elif any(kw in query_lower for kw in ['specific', 'particular', 'transcript', 'tell me about']):
            intent = 'SPECIFIC'
        elif any(kw in query_lower for kw in ['compare', 'vs', 'versus', 'difference between']):
            intent = 'COMPARISON'
        else:
            intent = 'SEARCH'
        print(f"Fallback classified intent as: {intent}")
        return intent


def query_with_strategy(query_embedding, intent):
    """Query Pinecone with strategy based on intent."""
    
    try:
        if intent == 'AGGREGATE':
            # Use aggregate namespace + sample of regular calls
            print("Querying aggregates namespace...")
            agg_results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True,
                namespace="aggregates"
            )
            print(f"Found {len(agg_results.get('matches', []))} aggregate matches")
            
            print("Querying default namespace for samples...")
            sample_results = index.query(
                vector=query_embedding,
                top_k=10,
                include_metadata=True,
                namespace=""  # FIXED: Explicitly specify default namespace
            )
            print(f"Found {len(sample_results.get('matches', []))} sample matches")
            
            return {'aggregates': agg_results, 'samples': sample_results}
        
        elif intent == 'SPECIFIC':
            # Detailed retrieval of few calls
            print("Querying for specific calls...")
            results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True,
                namespace=""  # FIXED: Explicitly specify default namespace
            )
            print(f"Found {len(results.get('matches', []))} specific matches")
            return {'specific': results}
        
        elif intent == 'COMPARISON':
            # Get aggregates + more samples for comparison
            print("Querying aggregates for comparison...")
            agg_results = index.query(
                vector=query_embedding,
                top_k=10,
                include_metadata=True,
                namespace="aggregates"
            )
            print(f"Found {len(agg_results.get('matches', []))} aggregate matches")
            
            print("Querying default namespace for comparison samples...")
            sample_results = index.query(
                vector=query_embedding,
                top_k=30,
                include_metadata=True,
                namespace=""  # FIXED: Explicitly specify default namespace
            )
            print(f"Found {len(sample_results.get('matches', []))} sample matches")
            
            return {'aggregates': agg_results, 'samples': sample_results}
        
        else:  # SEARCH
            # Standard search with moderate results
            print("Querying for search results...")
            results = index.query(
                vector=query_embedding,
                top_k=20,
                include_metadata=True,
                namespace=""  # FIXED: Explicitly specify default namespace
            )
            print(f"Found {len(results.get('matches', []))} search matches")
            return {'search': results}
    
    except Exception as e:
        print(f"ERROR in query_with_strategy: {e}")
        import traceback
        traceback.print_exc()
        return {}


def build_context(query_results, intent):
    """Build optimized context based on intent."""
    
    # Helper function to extract matches from Pinecone response
    def get_matches(result):
        if hasattr(result, 'matches'):
            return result.matches
        elif isinstance(result, dict) and 'matches' in result:
            return result['matches']
        return []
    
    if intent == 'AGGREGATE':
        context = "=== PRE-COMPUTED STATISTICS ===\n"
        
        # Add aggregate summaries
        agg_matches = get_matches(query_results.get('aggregates', {}))
        if agg_matches:
            for match in agg_matches:
                metadata = match.get('metadata', {}) if isinstance(match, dict) else match.metadata
                context += f"\n{metadata.get('summary', '')}\n"
        else:
            context += "\nNo aggregate data available.\n"
        
        # Add compact sample examples
        context += "\n=== SAMPLE EXAMPLES ===\n"
        sample_matches = get_matches(query_results.get('samples', {}))
        if sample_matches:
            for i, match in enumerate(sample_matches[:5], 1):
                m = match.get('metadata', {}) if isinstance(match, dict) else match.metadata
                context += f"{i}. [{m.get('date', 'N/A')}] {m.get('sentiment', 'N/A')} - {m.get('primary_issue', 'N/A')}\n"
        else:
            context += "No sample data available.\n"
        
        return context
    
    elif intent == 'SPECIFIC':
        context = "=== DETAILED CALL INFORMATION ===\n\n"
        
        specific_matches = get_matches(query_results.get('specific', {}))
        if specific_matches:
            for i, match in enumerate(specific_matches, 1):
                score = match.get('score', 0) if isinstance(match, dict) else match.score
                m = match.get('metadata', {}) if isinstance(match, dict) else match.metadata
                
                # Build context with only available fields
                context += f"Call #{i} (Relevance: {score:.2f})\n"
                context += f"Date: {m.get('date', 'N/A')}\n"
                
                # Add optional fields only if they exist
                if m.get('primary_issue'):
                    context += f"Issue: {m.get('primary_issue')}\n"
                if m.get('issue_category'):
                    context += f"Category: {m.get('issue_category')}\n"
                    
                context += f"Sentiment: {m.get('sentiment', 'N/A')}\n"
                
                if m.get('urgency'):
                    context += f"Urgency: {m.get('urgency')}\n"
                if m.get('resolution_status'):
                    context += f"Resolution: {m.get('resolution_status')}\n"
                if m.get('customer_satisfaction'):
                    context += f"Satisfaction: {m.get('customer_satisfaction')}/5\n"
                    
                context += f"Summary: {m.get('summary', 'N/A')}\n"
                
                if m.get('transcript_snippet'):
                    context += f"Transcript Snippet: {m.get('transcript_snippet')}\n"
                if m.get('action_items'):
                    context += f"Action Items: {m.get('action_items')}\n"
                
                context += "\n"
        else:
            context += "No specific call data found.\n"
        
        return context
    
    elif intent == 'COMPARISON':
        context = "=== AGGREGATE DATA FOR COMPARISON ===\n"
        
        agg_matches = get_matches(query_results.get('aggregates', {}))
        if agg_matches:
            for match in agg_matches:
                metadata = match.get('metadata', {}) if isinstance(match, dict) else match.metadata
                context += f"\n{metadata.get('summary', '')}\n"
        else:
            context += "\nNo aggregate data available for comparison.\n"
        
        context += "\n=== SAMPLE DATA POINTS ===\n"
        sample_matches = get_matches(query_results.get('samples', {}))
        if sample_matches:
            for i, match in enumerate(sample_matches[:15], 1):
                m = match.get('metadata', {}) if isinstance(match, dict) else match.metadata
                context += f"{i}. [{m.get('date', 'N/A')}] {m.get('issue_category', 'N/A')} | {m.get('sentiment', 'N/A')} | Satisfaction: {m.get('customer_satisfaction', 'N/A')}\n"
        else:
            context += "No sample data available for comparison.\n"
        
        return context
    
    else:  # SEARCH
        search_matches = get_matches(query_results.get('search', {}))
        context = f"=== SEARCH RESULTS ({len(search_matches)} calls found) ===\n\n"
        
        if search_matches:
            for i, match in enumerate(search_matches, 1):
                m = match.get('metadata', {}) if isinstance(match, dict) else match.metadata
                summary = m.get('summary', 'N/A')
                summary_preview = summary[:150] + "..." if len(summary) > 150 else summary
                context += f"""{i}. {m.get('date', 'N/A')} | {m.get('sentiment', 'N/A')}
   Issue: {m.get('primary_issue', 'N/A')}
   Summary: {summary_preview}
   
"""
        else:
            context += "No matching calls found.\n"
        
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
        
        print("Creating embedding...")
        response = openai_client.embeddings.create(
            input=[user_query], 
            model=EMBEDDING_MODEL,
            dimensions=PINECONE_DIMENSION
        )
        query_embedding = response.data[0].embedding
        print("Embedding created successfully")
        
        # Step 3: Query with appropriate strategy
        query_results = query_with_strategy(query_embedding, intent)
        
        # FIXED: Better result checking - handle both dict and Pinecone response objects
        has_results = False
        total_matches = 0
        for key, value in query_results.items():
            matches = None
            # Handle Pinecone response object
            if hasattr(value, 'matches'):
                matches = value.matches
            # Handle dictionary
            elif isinstance(value, dict) and 'matches' in value:
                matches = value['matches']
            
            if matches and len(matches) > 0:
                has_results = True
                total_matches += len(matches)
                print(f"  {key}: {len(matches)} matches")
        
        print(f"Total matches found: {total_matches}")
        
        if not has_results:
            return jsonify({"answer": "No relevant call records found for this query. This could mean:\n• No calls match your search criteria\n• The data hasn't been indexed yet\n• Try rephrasing your question"})
        
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
        print("Generating AI response...")
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
        print(f"ERROR in /chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"answer": f"Sorry, an unexpected error occurred: {str(e)}\n\nPlease check the server logs for more details."})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
