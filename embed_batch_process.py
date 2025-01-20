import os
import openai
import faiss
import numpy as np
from dotenv import load_dotenv
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Path to the scraped data folder
SCRAPED_DATA_FOLDER = r'E:\Synamedia Bot\scraped_data'

# Path to store cached embeddings
EMBEDDINGS_CACHE_FILE = r'E:\Synamedia Bot\embeddings_cache.pkl'

def get_embedding(text, model="text-embedding-ada-002"):
    if isinstance(text, str) and len(text) > 0:
        response = openai.Embedding.create(input=[text], model=model)
        embedding = response['data'][0]['embedding']
        return embedding
    else:
        raise ValueError(f"Invalid input for embedding: {text}")

def store_multiple_embeddings_in_faiss(text_sections):
    if not text_sections:
        raise ValueError("No valid text sections to embed.")
    
    dimension = 1536  # OpenAI's text-embedding-ada-002 uses 1536 dimensions
    index = faiss.IndexFlatL2(dimension)
    
    embeddings = []
    embedding_cache = {}
    
    batch_size = 10  # Adjust this based on your needs and API limits
    for i in range(0, len(text_sections), batch_size):
        batch = text_sections[i:i + batch_size]
        try:
            logger.info(f"Processing batch {i // batch_size + 1}")
            response = openai.Embedding.create(input=batch, model="text-embedding-ada-002")
            embeddings_batch = [data['embedding'] for data in response['data']]
            embeddings.extend(embeddings_batch)
            
            # Update the embedding cache
            for text, embedding in zip(batch, embeddings_batch):
                embedding_cache[text] = embedding
            
            logger.info(f"Batch {i // batch_size + 1} processed successfully.")
        except Exception as e: 
            logger.error(f"Error processing batch starting at index {i}: {e}")
    
    if embeddings:
        embeddings_np = np.array(embeddings, dtype=np.float32)
        index.add(embeddings_np)
        logger.info(f"Total embeddings generated: {len(embeddings)}")
    else:
        raise ValueError("No valid embeddings generated.")
    
    return index, text_sections, embedding_cache

def load_scraped_text_from_folder(folder_path):
    text_sections = []
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.readlines()
                    text_sections.extend([line.strip() for line in content if line.strip()])
    except Exception as e:
        logger.error(f"Error loading text files from folder: {e}")
    
    return text_sections

def save_embeddings_to_cache(faiss_index, text_sections, embedding_cache):
    with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
        pickle.dump({
            'index': faiss_index,
            'sections': text_sections,
            'cache': embedding_cache
        }, f)

if __name__ == '__main__':
    website_text_sections = load_scraped_text_from_folder(SCRAPED_DATA_FOLDER)

    if website_text_sections:
        try:
            faiss_index, sections, embedding_cache = store_multiple_embeddings_in_faiss(website_text_sections)
            save_embeddings_to_cache(faiss_index, sections, embedding_cache)
            logger.info("Embeddings created and cached successfully.")
        except ValueError as e:
            logger.error(f"Error creating embeddings: {e}")
    else:
        logger.error("Failed to process any text files.")
