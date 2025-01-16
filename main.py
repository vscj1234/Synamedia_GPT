from flask import Flask, render_template, request, jsonify, session 
import openai 
import os 
import faiss 
import numpy as np 
from dotenv import load_dotenv 
import pickle 
import uuid 
from typing import List, Dict, Union 
import logging 
from threading import Lock 
import time 
import requests 
 
# Configure logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 
logger = logging.getLogger(__name__) 
 
# Load environment variables 
load_dotenv() 
 
# Initialize Flask app 
app = Flask(__name__) 
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-here") 
app.config["SESSION_TYPE"] = "filesystem" 
 
# Configure OpenAI 
openai.api_key = os.getenv("OPENAI_API_KEY") 
 
# Constants 
EMBEDDINGS_CACHE_FILE = "embeddings_cache.pkl" 
EMBEDDING_MODEL = "text-embedding-ada-002" 
CHAT_MODEL = "gpt-4o-mini" 
MAX_CONTEXT_SECTIONS = 3 
CACHE_SAVE_INTERVAL = 300  # Save cache every 5 minutes 
 
# Zammad API configuration 
ZAMMAD_URL = "https://cloudjune.zammad.com" 
ZAMMAD_API_TOKEN = "zYUYZ8f0bl46IfYuFStAytjLq4Kqpj4dH9xMjXGwFsTuMzo-Dw8rpp-RZsoFgH6u" 
 
class EmbeddingManager: 
    def __init__(self): 
        self.embedding_cache: Dict[str, List[float]] = {} 
        self.faiss_index: Union[faiss.IndexFlatL2, None] = None 
        self.sections: List[str] = [] 
        self.cache_lock = Lock() 
        self.last_save_time = time.time() 
        self.load_cached_embeddings() 
 
    def load_cached_embeddings(self) -> None: 
        """Load embeddings from cache file if it exists.""" 
        try: 
            if os.path.exists(EMBEDDINGS_CACHE_FILE): 
                with open(EMBEDDINGS_CACHE_FILE, 'rb') as f: 
                    cached_data = pickle.load(f) 
                    if isinstance(cached_data, dict): 
                        self.faiss_index = cached_data.get("index") 
                        self.sections = cached_data.get("sections", []) 
                        self.embedding_cache = cached_data.get("cache", {}) 
                    else: 
                        logger.warning("Unexpected cache format. Initializing new FAISS index.") 
                        self.initialize_new_index() 
                logger.info("Embeddings loaded successfully") 
            else: 
                logger.info("No cache file found. Initializing new FAISS index.") 
                self.initialize_new_index() 
        except Exception as e: 
            logger.error(f"Error loading embeddings cache: {e}") 
            self.initialize_new_index() 
 
    def initialize_new_index(self): 
        """Initialize a new FAISS index.""" 
        self.faiss_index = faiss.IndexFlatL2(1536)  # OpenAI's text-embedding-ada-002 uses 1536 dimensions 
        self.sections = [] 
        self.embedding_cache = {} 
 
    def get_embedding(self, text: str) -> List[float]: 
        """Get embedding for text, using cache if available.""" 
        if not text or not isinstance(text, str): 
            raise ValueError("Invalid input for embedding") 
 
        with self.cache_lock: 
            if text in self.embedding_cache: 
                return self.embedding_cache[text] 
 
            try: 
                response = openai.Embedding.create(input=[text], model=EMBEDDING_MODEL) 
                embedding = response["data"][0]["embedding"] 
                self.embedding_cache[text] = embedding 
                self.sections.append(text) 
                if self.faiss_index is not None: 
                    self.faiss_index.add(np.array([embedding], dtype=np.float32)) 
                self.save_embeddings_if_needed() 
                return embedding 
            except Exception as e: 
                logger.error(f"Error generating embedding: {e}") 
                raise 
 
    def save_embeddings_if_needed(self): 
        """Save the current embeddings to the cache file if enough time has passed.""" 
        current_time = time.time() 
        if current_time - self.last_save_time > CACHE_SAVE_INTERVAL: 
            self.save_embeddings() 
            self.last_save_time = current_time 
 
    def save_embeddings(self): 
        """Save the current embeddings to the cache file.""" 
        try: 
            with open(EMBEDDINGS_CACHE_FILE, 'wb') as f: 
                pickle.dump({ 
                    "index": self.faiss_index, 
                    "sections": self.sections, 
                    "cache": self.embedding_cache 
                }, f) 
            logger.info("Embeddings saved successfully") 
        except Exception as e: 
            logger.error(f"Error saving embeddings: {e}") 
 
class QueryManager: 
    def __init__(self): 
        self.user_queries: Dict[str, List[str]] = {} 
 
    def add_query(self, user_id: str, query: str) -> None: 
        """Add a query to the user's history.""" 
        if user_id not in self.user_queries: 
            self.user_queries[user_id] = [] 
        self.user_queries[user_id].append(query) 
 
    def get_previous_query(self, user_id: str) -> Union[str, None]: 
        """Get the user's previous query.""" 
        queries = self.user_queries.get(user_id, []) 
        return queries[-2] if len(queries) > 1 else None 
 
# Initialize managers 
embedding_manager = EmbeddingManager() 
query_manager = QueryManager() 
 
def create_zammad_ticket() -> str: 
    """ 
    Create a Zammad ticket using the provided credentials and return a message. 
    """ 
    ticket_data = { 
        "title": "User Issue Ticket from API", 
        "group": "Users",  # Replace with the appropriate group name in your Zammad instance 
        "customer": "rutvija27@gmail.com",  # Replace with an actual email of a user in your Zammad system 
        "article": { 
            "subject": "Test Ticket Creation", 
            "body": "This is a test ticket created via the Zammad API.", 
            "type": "note", 
        }, 
        "priority_id": 2,  # Priority ID (1 = low, 2 = normal, 3 = high, etc.) 
        "state_id": 1,     # State ID (1 = new, 2 = open, etc.) 
    } 
 
    headers = { 
        "Authorization": f"Token token={ZAMMAD_API_TOKEN}", 
        "Content-Type": "application/json", 
    } 
 
    response = requests.post(f"{ZAMMAD_URL}/api/v1/tickets", json=ticket_data, headers=headers) 
 
    if response.status_code == 201:  # 201 Created 
        return "Ticket created successfully!" 
    elif response.status_code == 401:  # Unauthorized 
        return "Invalid API token. Please check your credentials." 
    else: 
        return f"Failed to create ticket. HTTP {response.status_code}: {response.text}" 
 
def generate_response(context: str, query: str) -> str: 
    """ 
    Generate a response using GPT-4. 
    This prompt includes a step encouraging the AI to create a Zammad ticket if the user requests it. 
    """ 
    prompt = f""" 
You are Synamedia's AI assistant, specialized in video delivery, processing, and monetization solutions. 
Your goal is to provide accurate and concise answers based on Synamedia's products and services. 
 
Context: {context} 
 
User Query: {query} 
 
Response Guidelines: 
Limit your response to a maximum of 2 lines. 
Be clear, precise, and directly address the user's question. 
Avoid unnecessary details or repetition. 

Troubleshooting Guidelines:
Structured Responses:
Address user issues with step-by-step solutions. Tailor steps to the specific problem described, ensuring each action is clear and easy to follow.
 
Dynamic Guidance:
Offer follow-up solutions if initial suggestions do not resolve the issue. Reassure users and maintain a positive tone throughout.
 
Example Interaction 1:
 
User: "I’m unable to stream the content."
Bot: "Let’s resolve this together. Please try the following steps one at a time:
Ensure your internet connection is stable and meets the minimum bandwidth requirement.
Restart your streaming device (smart TV, phone, laptop, etc.).
Clear the app or browser cache.
Let me know if you still experience issues after trying these steps."
 
User: "I’m still experiencing buffering."
Bot: "Thanks for trying those steps! Here are a few additional solutions:
Test your internet speed at [link to tool, if applicable]. Ensure it meets the requirements for streaming.
If you’re using Wi-Fi, try moving closer to your router or switching to a wired connection.
Check if other apps or devices are consuming bandwidth and pause them temporarily.
Let me know if the issue persists." 
 
User: "That worked. Thanks!"
Bot: "I’m glad we could resolve the issue! Let me know if there’s anything else I can assist you with."

Example Interaction 2:
 
User: "Advertising Metrics Not Loading"
Bot: "Check the user’s internet connection and reload the dashboard."

User: "Checked, it is all good"
Bot: "Ensure that the user’s subscription plan includes analytics features and "Clear the browser or app cache and cookies."

User: "done, still not working"
Bot: "Check the user’s internet connection and reload the dashboard. If it doesn't work let me know if I can create a ticket for you"

User: "No luck, yes please"
Bot: "Ticket successfully created"

Example Interaction 3:
 
User: "Tv not working"
Bot: "Please check your TV power connection and ensure it’s turned on. If that doesn’t resolve the issue, try restarting the TV and any connected devices."

User: "still no luck"
Bot: "Sorry to hear that, do you want me to raise a ticket?"

User: "Yes please"
Bot: "Ticket successfully created"
 
(If the user asks for a ticket creation, instruct the system to create a Zammad ticket.) 
(When the troubleshooting steps exahust and user still face issues, instruct the system to create a Zammad ticket.) 
""" 
 
    try: 
        response = openai.ChatCompletion.create( 
            model=CHAT_MODEL, 
            messages=[ 
                {"role": "system", "content": prompt}, 
                {"role": "user", "content": query} 
            ] 
        ) 
        return response["choices"][0]["message"]["content"] 
    except Exception as e: 
        logger.error(f"Error generating GPT response: {e}") 
        return "I apologize, but I'm having trouble generating a response. Please try again." 
 
@app.route("/") 
def index(): 
    """Render the main page.""" 
    return render_template("index.html") 
 
@app.route("/ask", methods=["POST"]) 
def ask(): 
    """Handle user queries.""" 
    try: 
        data = request.get_json() 
        user_query = data.get("query", "").strip() 
 
        if not user_query: 
            return jsonify({"response": "I couldn't understand your question. Could you please rephrase it?"}) 
 
        # Manage session 
        user_id = session.get("user_id") 
        if not user_id: 
            user_id = str(uuid.uuid4()) 
            session["user_id"] = user_id 
 
        # Add query to history 
        query_manager.add_query(user_id, user_query) 
 
        # Check for previous question query 
        if user_query.lower() == "what was my previous question?": 
            previous_query = query_manager.get_previous_query(user_id) 
            if previous_query: 
                return jsonify({"response": f"Your previous question was: {previous_query}"}) 
            return jsonify({"response": "You haven't asked any previous questions yet."}) 
 
        # If user explicitly wants to create a ticket 
        if "create a ticket" in user_query.lower(): 
            ticket_creation_result = create_zammad_ticket() 
            return jsonify({"response": ticket_creation_result}) 
 
        # Get relevant content for GPT 
        try: 
            query_embedding = embedding_manager.get_embedding(user_query) 
            if embedding_manager.faiss_index is None: 
                return jsonify({"response": "I'm currently initializing. Please try again in a moment."}) 
 
            distances, indices = embedding_manager.faiss_index.search( 
                np.array([query_embedding], dtype=np.float32), 
                k=MAX_CONTEXT_SECTIONS 
            ) 
 
            relevant_sections = [ 
                embedding_manager.sections[i] 
                for i in indices[0] 
                if i != -1 and i < len(embedding_manager.sections) 
            ] 
 
            if not relevant_sections: 
                return jsonify({"response": "I don't have enough information to answer that question accurately."}) 
 
            context = " ".join(relevant_sections) 
            response_text = generate_response(context, user_query) 
            return jsonify({"response": response_text}) 
 
        except Exception as e: 
            logger.error(f"Error processing query: {e}") 
            return jsonify({"response": "I encountered an error while processing your question. Please try again."}) 
 
    except Exception as e: 
        logger.error(f"Error in ask endpoint: {e}") 
        return jsonify({"response": "An unexpected error occurred. Please try again."}) 
 
if __name__ == "__main__": 
    app.run(debug=True)
