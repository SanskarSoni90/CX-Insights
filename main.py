import os
import requests
import time
import json
from datetime import datetime, timedelta, timezone
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import openai

# --- Configuration ---
# Load secrets from environment variables
EXOTEL_ACCOUNT_SID = os.getenv('EXOTEL_ACCOUNT_SID')
EXOTEL_API_KEY = os.getenv('EXOTEL_API_KEY')
EXOTEL_API_TOKEN = os.getenv('EXOTEL_API_TOKEN')
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_SHEET_ID = '1LGeqJTaX6IfjHz2-H1YGzuRV3FM22P33Uqx2nwp5Zjs'
GOOGLE_SHEET_NAME = 'Sheet1'
# The GOOGLE_CREDENTIALS_JSON is a multi-line secret, it needs to be loaded carefully
GOOGLE_CREDENTIALS_JSON = os.getenv('GOOGLE_CREDENTIALS_JSON')

# File to store processed call SIDs
PROCESSED_CALLS_FILE = 'processed_calls.txt'

# --- Helper Functions ---

def get_processed_calls():
    """Reads the set of already processed call SIDs from a file."""
    if not os.path.exists(PROCESSED_CALLS_FILE):
        return set()
    with open(PROCESSED_CALLS_FILE, 'r') as f:
        return set(line.strip() for line in f)

def add_processed_call(sid):
    """Adds a new call SID to the processed calls file."""
    with open(PROCESSED_CALLS_FILE, 'a') as f:
        f.write(sid + '\n')

def fetch_recent_exotel_calls():
    """Fetches calls from Exotel from the last 15 minutes."""
    print("Fetching recent calls from Exotel...")
    try:
        # Calculate time range
        start_time = datetime.now(timezone.utc) - timedelta(minutes=15)
        start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        # List of possible Exotel API endpoints to try
        endpoints_to_try = [
            f"https://api.exotel.com/v1/Accounts/{EXOTEL_ACCOUNT_SID}/Calls.json",
            f"https://api.in.exotel.com/v1/Accounts/{EXOTEL_ACCOUNT_SID}/Calls.json", 
            f"https://twilix.exotel.com/v1/Accounts/{EXOTEL_ACCOUNT_SID}/Calls.json",
            f"https://{EXOTEL_ACCOUNT_SID}.exotel.com/v1/Accounts/{EXOTEL_ACCOUNT_SID}/Calls.json",
            f"https://api.sg.exotel.com/v1/Accounts/{EXOTEL_ACCOUNT_SID}/Calls.json"
        ]
        
        params = {
            'DateCreated': f'gte:{start_time_str}',
            'PageSize': 20
        }
        
        last_error = None
        
        for url in endpoints_to_try:
            try:
                print(f"Trying endpoint: {url}")
                response = requests.get(url, auth=(EXOTEL_API_KEY, EXOTEL_API_TOKEN), params=params, timeout=30)
                
                if response.status_code == 200:
                    print(f"✅ SUCCESS! Working endpoint: {url}")
                    data = response.json()
                    print(f"Found {len(data.get('Calls', []))} calls in the last 15 minutes.")
                    
                    # Save the working endpoint for future reference
                    print(f"🔥 IMPORTANT: Use this URL in your code: {url}")
                    return data.get('Calls', [])
                    
                elif response.status_code == 401:
                    print(f"❌ Authentication failed for {url} - check your API credentials")
                    print(f"Response: {response.text}")
                elif response.status_code == 404:
                    print(f"❌ Account not found on {url} - wrong endpoint for your account")
                else:
                    print(f"❌ {url} returned status {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError as e:
                print(f"❌ Connection failed for {url}: {str(e)}")
                last_error = e
            except Exception as e:
                print(f"❌ Error with {url}: {str(e)}")
                last_error = e
        
        # If we get here, no endpoint worked
        print("🚫 No working Exotel endpoint found!")
        print("Please check:")
        print("1. Your EXOTEL_ACCOUNT_SID is correct")
        print("2. Your EXOTEL_API_KEY is correct") 
        print("3. Your EXOTEL_API_TOKEN is correct")
        print("4. Your Exotel account is active and has API access")
        
        if last_error:
            raise last_error
        return []
        
    except Exception as e:
        print(f"Final error fetching calls from Exotel: {e}")
        return []


def test_exotel_connection():
    """Test function to verify Exotel API connectivity and credentials."""
    print("=== Testing Exotel API Connection ===")
    print(f"Account SID: {EXOTEL_ACCOUNT_SID}")
    print(f"API Key: {EXOTEL_API_KEY[:10]}...") # Only show first 10 chars for security
    print(f"API Token: {'*' * len(EXOTEL_API_TOKEN)}")
    
    # Try to get account details first
    endpoints_to_try = [
        f"https://api.exotel.com/v1/Accounts/{EXOTEL_ACCOUNT_SID}.json",
        f"https://api.in.exotel.com/v1/Accounts/{EXOTEL_ACCOUNT_SID}.json",
        f"https://twilix.exotel.com/v1/Accounts/{EXOTEL_ACCOUNT_SID}.json"
    ]
    
    for url in endpoints_to_try:
        try:
            print(f"Testing account endpoint: {url}")
            response = requests.get(url, auth=(EXOTEL_API_KEY, EXOTEL_API_TOKEN), timeout=10)
            
            if response.status_code == 200:
                print(f"✅ Account endpoint working: {url}")
                account_info = response.json()
                print(f"Account Name: {account_info.get('account', {}).get('name', 'N/A')}")
                print(f"Account Status: {account_info.get('account', {}).get('status', 'N/A')}")
                return True
            else:
                print(f"❌ Status {response.status_code}: {response.text[:200]}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return False

def transcribe_audio(audio_url):
    """Submits audio for transcription to AssemblyAI and waits for the result."""
    print(f"Transcribing audio from: {audio_url}")
    try:
        headers = {'authorization': ASSEMBLYAI_API_KEY}
        json_data = {'audio_url': audio_url}
        
        # Submit transcription request
        upload_response = requests.post('https://api.assemblyai.com/v2/transcript', json=json_data, headers=headers)
        upload_response.raise_for_status()
        transcript_id = upload_response.json()['id']
        print(f"Transcription submitted. ID: {transcript_id}")

        # Poll for completion
        while True:
            poll_url = f'https://api.assemblyai.com/v2/transcript/{transcript_id}'
            poll_response = requests.get(poll_url, headers=headers)
            poll_response.raise_for_status()
            transcript_data = poll_response.json()
            if transcript_data['status'] == 'completed':
                print("Transcription complete.")
                return transcript_data['text']
            elif transcript_data['status'] == 'failed':
                print(f"Transcription failed: {transcript_data.get('error')}")
                return None
            print("Transcription in progress, waiting...")
            time.sleep(5)
    except requests.exceptions.RequestException as e:
        print(f"Error during transcription: {e}")
        return None

def analyze_transcript_with_openai(transcript):
    """Analyzes the transcript using OpenAI to extract insights."""
    print("Analyzing transcript with OpenAI...")
    if not transcript:
        return {}
    
    # Initialize the OpenAI client
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    system_prompt = """
    You are an expert call center analyst. Analyze the following call transcript and extract the specified information.
    Respond ONLY with a valid JSON object. Do not include any explanatory text before or after the JSON.
    The JSON object should have the following keys:
    "primary_issue", "issue_category", "sentiment", "urgency", "resolution_status", 
    "customer_satisfaction", "key_topics", "action_items", "agent_performance", 
    "summary", "transcript_snippet"
    
    - sentiment: "Positive", "Negative", or "Neutral".
    - urgency: "High", "Medium", or "Low".
    - resolution_status: "Resolved", "Unresolved", or "Needs Follow-up".
    - customer_satisfaction: A rating from 1 to 5.
    - key_topics: A comma-separated string of main topics.
    - action_items: A comma-separated string of action items.
    - transcript_snippet: A relevant 1-2 sentence snippet from the transcript.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript}
            ],
            response_format={"type": "json_object"}
        )
        analysis = json.loads(response.choices[0].message.content)
        print("Analysis complete.")
        return analysis
    except Exception as e:
        print(f"Error analyzing transcript with OpenAI: {e}")
        return {}

def update_google_sheet(data_row):
    """Appends a new row of data to the Google Sheet."""
    print("Updating Google Sheet...")
    try:
        # Load credentials from the environment variable
        creds_json = json.loads(GOOGLE_CREDENTIALS_JSON)
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)
        client = gspread.authorize(creds)
        
        sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
        
        # The order must match the columns in the sheet
        headers = [
            'call_id', 'date', 'duration', 'direction', 'primary_issue', 'issue_category',
            'sentiment', 'urgency', 'resolution_status', 'customer_satisfaction',
            'key_topics', 'action_items', 'agent_performance', 'summary',
            'transcript_snippet', 'transcript'
        ]
        
        # Ensure all headers have a value, default to ""
        row_to_insert = [data_row.get(h, "") for h in headers]

        sheet.append_row(row_to_insert)
        print("Google Sheet updated successfully.")
    except Exception as e:
        print(f"Error updating Google Sheet: {e}")

# --- Main Logic ---
def main():
    """Test version of main function to debug Exotel connection."""
    print("=== DEBUGGING EXOTEL CONNECTION ===")
    
    # Test connection first
    if test_exotel_connection():
        print("✅ Account connection successful, testing calls endpoint...")
        calls = fetch_recent_exotel_calls()
        print(f"Retrieved {len(calls)} calls")
    else:
        print("❌ Account connection failed")
    
    print("=== DEBUG COMPLETE ===")
