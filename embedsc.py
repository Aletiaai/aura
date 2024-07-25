import re
import numpy as np
import torch
import pypdf
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

# Load the model and tokenizer
model_name = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = pypdf.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def chunk_text(text):
    # Split the input text into individual sentences.
    single_sentences_list = _split_sentences(text)

    # Combine adjacent sentences to form a context window around each sentence.
    combined_sentences = _combine_sentences(single_sentences_list)
    
    # Convert the combined sentences into vector representations using a neural network model.
    embeddings = convert_to_vector(combined_sentences)
    
    # Calculate the cosine distances between consecutive combined sentence embeddings to measure similarity.
    distances = _calculate_cosine_distances(embeddings)
    
    # Determine the threshold distance for identifying breakpoints based on the 80th percentile of all distances.
    breakpoint_percentile_threshold = 80
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)

    # Find all indices where the distance exceeds the calculated threshold, indicating a potential chunk breakpoint.
    indices_above_thresh = [i for i, distance in enumerate(distances) if distance > breakpoint_distance_threshold]

    # Initialize the list of chunks and a variable to track the start of the next chunk.
    chunks = []
    start_index = 0

    # Loop through the identified breakpoints and create chunks accordingly.
    for index in indices_above_thresh:
        chunk = ' '.join(single_sentences_list[start_index:index+1])
        chunks.append(chunk)
        start_index = index + 1
    
    # If there are any sentences left after the last breakpoint, add them as the final chunk.
    if start_index < len(single_sentences_list):
        chunk = ' '.join(single_sentences_list[start_index:])
        chunks.append(chunk)
    
    # Return the list of text chunks.
    return chunks

def _split_sentences(text):
    # Use regular expressions to split the text into sentences based on punctuation followed by whitespace.
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return sentences

def _combine_sentences(sentences):
    # Create a buffer by combining each sentence with its previous and next sentence to provide a wider context.
    combined_sentences = []
    for i in range(len(sentences)):
        combined_sentence = sentences[i]
        if i > 0:
            combined_sentence = sentences[i-1] + ' ' + combined_sentence
        if i < len(sentences) - 1:
            combined_sentence += ' ' + sentences[i+1]
        combined_sentences.append(combined_sentence)
    return combined_sentences

def convert_to_vector(texts):
    # Try to generate embeddings for a list of texts using a pre-trained model and handle any exceptions.
    try:
        # Tokenize the input texts
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use the mean of the last hidden state as the sentence embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.numpy()
    except Exception as e:
        print("An error occurred:", e)
        return np.array([])  # Return an empty array in case of an error

def _calculate_cosine_distances(embeddings):
    # Calculate the cosine distance (1 - cosine similarity) between consecutive embeddings.
    distances = []
    for i in range(len(embeddings) - 1):
        similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        distance = 1 - similarity
        distances.append(distance)
    return distances

# Main Section
if __name__ == "__main__":
    pdf_path = "path/to/your/pdf/file.pdf"
    
    # Step 1: Extract text from PDF
    full_text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Chunk the extracted text
    chunks = chunk_text(full_text)
    
    # Step 3: Process or store the chunks as needed
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(chunk[:100] + "...")  # Print first 100 characters of each chunk
        print()

    print(f"Total number of chunks: {len(chunks)}")