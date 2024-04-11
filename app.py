import csv
import io
import math
import unicodedata
from flask import Flask, render_template, request, Response
from nltk.tokenize import sent_tokenize
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import numpy as np
import spacy
import calamancy

app = Flask(__name__)

def load_stopwords(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])
    return stopwords

# Assuming you have your Tagalog stopwords file 'stopwords_file.txt'
stopwords_file = 'stopwords_file.txt'
stopwords = load_stopwords(stopwords_file)

@app.route('/')
def index():
    return render_template('index.html')

def calculate_num_sentences(num_sentences):
    num_sentences = math.floor(num_sentences / 2)
    return max(1, num_sentences)

@app.route('/summarize', methods=['POST'])
def summarize():
    csv_file = request.files['file']
    csv_text = csv_file.stream.read().decode('latin-1')

    summaries = []

    if csv_file.filename.lower().endswith('.csv'):
        csv_data = csv.reader(io.StringIO(csv_text))
        for row in csv_data:
            if row:  # Check if the row is not empty
                text = row[0].strip()  # Trim leading and trailing whitespace
                num_sentences = calculate_num_sentences(len(sent_tokenize(text)))
                summary = summarize_dataset(text, num_sentences)
                summaries.append(summary)

    csv_output = convert_to_csv(summaries)
    return Response(csv_output, mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=summary.csv'})

def summarize_dataset(text, num_sentences):
    summary = summarize_text(text, num_sentences=num_sentences)
    return ' '.join(summary)

def preprocess_text(text):
    # Tokenize the text into sentences
    original_sentences = sent_tokenize(text)

    # Remove stopwords and punctuation
    preprocessed_sentences = []
    for sentence in original_sentences:
        words = sentence.split()
        filtered_words = [word.lower() for word in words if word.lower() not in stopwords and word.isalnum()]
        preprocessed_sentences.append(" ".join(filtered_words))

    return original_sentences, preprocessed_sentences

def normalize_scores(scores):
    scaler = MinMaxScaler()
    scores = np.array(scores).reshape(-1, 1)
    normalized_scores = scaler.fit_transform(scores)
    return normalized_scores.flatten()

# nlp = spacy.load("en_core_web_sm")
nlp = calamancy.load("tl_calamancy_lg-0.1.0")

def identify_named_entities(sentence):
    doc = nlp(sentence)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def preprocess_text_with_ner(text):
    original_sentences = sent_tokenize(text)
    preprocessed_sentences = []
    named_entities = []

    for sentence in original_sentences:
        # Remove stopwords and punctuation
        words = sentence.split()
        filtered_words = [word.lower() for word in words if word.lower() not in stopwords and word.isalnum()]
        preprocessed_sentence = " ".join(filtered_words)
        
        # Identify named entities in the sentence
        entities = identify_named_entities(sentence)
        named_entities.append(entities)

        preprocessed_sentences.append(preprocessed_sentence)

    return original_sentences, preprocessed_sentences, named_entities


def calculate_scores(preprocessed_sentences, named_entities, bm_k1, bm_b, textrank_alpha):
    # Calculate BM25 scores for sentences
    bm25 = BM25Okapi(preprocessed_sentences, k1=bm_k1, b=bm_b)
    bm25_scores = normalize_scores(bm25.get_scores(preprocessed_sentences))

    # Create a TF-IDF matrix for the sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)

    # Calculate TextRank scores using cosine similarity
    similarity_matrix = np.dot(tfidf_matrix, tfidf_matrix.T)
    textrank_graph = nx.from_numpy_array(similarity_matrix)
    textrank_scores = nx.pagerank(textrank_graph, alpha=textrank_alpha)
    textrank_scores = normalize_scores(list(textrank_scores.values()))

    entity_scores = []
    for entities in named_entities:
        entity_score = 0
        for entity, label in entities:
            # Adjust scores based on named entity labels (e.g., PERSON, ORGANIZATION)
            if label == "PER":
                entity_score += 0.2  # Increase score for sentences containing person names
            elif label == "ORG":
                entity_score += 0.1  # Increase score for sentences containing organization names
            elif label == "LOC":
                entity_score += 0.15  # Increase score for sentences containing location names
        entity_scores.append(entity_score)

    # Normalize entity scores
    normalized_entity_scores = normalize_scores(entity_scores)

    return bm25_scores, textrank_scores, normalized_entity_scores

# Modify the summarize_text function to include named entity scores
def summarize_text(text, num_sentences, bm25_weight=0.7, textrank_weight=0.3, entity_weight=0.1, bm_k1=1.25, bm_b=0.75, textrank_alpha=0.9):
    original_sentences, preprocessed_sentences, named_entities = preprocess_text_with_ner(text)

    if len(preprocessed_sentences) <= num_sentences:
        return original_sentences

    bm25_scores, textrank_scores, entity_scores = calculate_scores(preprocessed_sentences, named_entities, bm_k1, bm_b, textrank_alpha)

    # Combine the scores
    combined_scores = bm25_weight * bm25_scores + textrank_weight * textrank_scores + entity_weight * entity_scores

    # Sort sentences by combined score
    sorted_indices = np.argsort(combined_scores)[::-1][:num_sentences]

    # Select the top sentences to form the summary
    summary = [original_sentences[i] for i in sorted_indices]

    return summary

def clean_text(text):
    # Define characters to exclude from removal
    allowed_characters = set(',.!?')  # Add any other characters you want to preserve
    
    # Remove non-ASCII characters and characters not in the allowed set
    cleaned_text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in allowed_characters)
    
    # Normalize Unicode characters to remove accents and diacritics
    cleaned_text = unicodedata.normalize('NFKD', cleaned_text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    return cleaned_text.strip()

def convert_to_csv(summaries):
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
    for summary in summaries:
        # Clean and normalize the summary text
        cleaned_summary = clean_text(summary)
        # Write the cleaned summary to CSV
        writer.writerow([cleaned_summary])
    csv_data = output.getvalue()
    output.close()
    return csv_data

if __name__ == '__main__':
    app.run(debug=True)
