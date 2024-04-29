import re
import nltk
import numpy as np
from tkinter import *
from tkinter import scrolledtext, filedialog
from nltk.tokenize import sent_tokenize, word_tokenize  # Added sent_tokenize
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import csv
import math

def preprocess_text(text, custom_stopwords_file='stopwords_file.txt'):
    # Tokenize the text into sentences
    original_sentences = sent_tokenize(text)

    # Read custom stopwords from the file
    custom_stopwords = []
    with open(custom_stopwords_file, 'r', encoding='utf-8') as f:
        custom_stopwords = [line.strip() for line in f]

    # Remove custom stopwords and keep track of correspondence with original sentences
    sentence_mapping = []
    for sentence in original_sentences:
        words = sentence.split()
        filtered_words = [word for word in words if word.lower() not in custom_stopwords]
        sentence_mapping.append((sentence, " ".join(filtered_words)))

    return sentence_mapping


def calculate_textrank_score(sentences):
    # Create a TF-IDF matrix for the sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([filtered for _, filtered in sentences])

    # Calculate TextRank scores using cosine similarity
    similarity_matrix = np.dot(tfidf_matrix, tfidf_matrix.T)
    textrank_graph = nx.from_numpy_array(similarity_matrix)
    textrank_scores = nx.pagerank(textrank_graph)

    return textrank_scores


def get_num_sentences(sentence_count):
    num_sentences = math.floor(sentence_count / 2)
    return max(1, num_sentences)

def summarize_text(text):
    sentence_count = len(sent_tokenize(text))
    num_sentences = get_num_sentences(sentence_count)
    # Preprocess the text
    sentence_mapping = preprocess_text(text)

    if len(sentence_mapping) <= num_sentences:
        summary = [original for original, _ in sentence_mapping]
    else:
        # Calculate TextRank scores for sentences
        textrank_scores = calculate_textrank_score(sentence_mapping)

        # Sort sentences by raw TextRank score
        sorted_sentence_mapping = sorted(sentence_mapping, key=lambda x: textrank_scores[sentence_mapping.index(x)], reverse=True)

        # Select the top sentences to form the summary
        summary = [original for original, _ in sorted_sentence_mapping[:num_sentences]]

    # Return the summary
    return ' '.join(summary)

def load_csv_and_summarize():
    input_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    output_file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if input_file_path and output_file_path:
        with open(input_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            with open(output_file_path, 'w', newline='', encoding='utf-8') as output_csv:
                csv_writer = csv.writer(output_csv)
                for row in csv_reader:
                    text = ' '.join(row)
                    summary = summarize_text(text)
                    csv_writer.writerow([summary])

# Create the main window
window = Tk()
window.title("TextRank BASELINE")

# Create button to load CSV and summarize
load_csv_button = Button(window, text="Load CSV, Summarize, and Save", command=load_csv_and_summarize)
load_csv_button.grid(column=0, row=0, pady=5)

# Start the GUI
window.mainloop()
