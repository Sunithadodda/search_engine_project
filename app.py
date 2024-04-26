from flask import Flask, render_template, request # type: ignore
import chromadb # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore

app = Flask(__name__)

# Initialize ChromaDB client and collection
client = chromadb.PersistentClient(path="vectordb")
collection = client.get_collection(name="Data")

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the functions for encoding query and printing movie names
def encode_input_query(query_point, model):
    """
    Encode the query point using the provided model.
    """
    return model.encode(query_point).tolist()

def print_movie_info(result):
    """
    Extract movie names and subtitle IDs from the query results.
    """
    movie_info = [(meta['name'], meta['subtitle_id']) for sublist in result['metadatas'] for meta in sublist if 'name' in meta and 'subtitle_id' in meta]
    return movie_info

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    # Get the subtitle snippet from the form
    prompt = request.form['prompt']
    
    # Encode the query point
    doc_vector = encode_input_query(prompt, model)
    
    # Query the collection
    results = collection.query(
        query_embeddings=doc_vector,
        n_results=10,
    )

    # Extract movie names and subtitle IDs
    movie_information = print_movie_info(results)
    
    # Render the template with the movie names and subtitle IDs
    return render_template('result.html', movie_info=movie_information)

if __name__ == '__main__':
    app.run(debug=True)
