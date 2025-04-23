import spacy
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer

# Load dataset (CNN/Daily Mail)
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# Download NLTK tokenizer
nltk.download("punkt")
nltk.download('punkt_tab')

# Function to preprocess text
def preprocess_text(text):
    doc = nlp(text)
    return " ".join([sent.text for sent in doc.sents])

# Function for Extractive Summarization
def extractive_summary(text, num_sentences=3):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:num_sentences])

# Load Pre-trained Model & Tokenizer for Abstractive Summarization
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Function for Abstractive Summarization
def abstractive_summary(text):
    inputs = tokenizer.encode("summarize: "   + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids =  model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4,  early_stopping=True)
    return tokenizer.decode(summary_ids[0],skip_special_tokens=True)

# Function to Evaluate Summary using ROUGE Score
def evaluate_summary(original, summary):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(original, summary)
    return scores

# ---- EXECUTION ----
# Get a sample article
sample_article = dataset["test"][0]["article"]
preprocessed_text = preprocess_text(sample_article)

# Generate Extractive Summary
extractive_result = extractive_summary(preprocessed_text, num_sentences=3)

# Generate Abstractive Summary
abstractive_result = abstractive_summary(preprocessed_text)

# Evaluate Summary Quality
evaluation = evaluate_summary(preprocessed_text, abstractive_result)

# ---- OUTPUT ----
print("\nOriginal Text:\n", sample_article[:500])
print("\nExtractive Summary:\n", extractive_result)
print("\nAbstractive Summary:\n", abstractive_result)
print("\nEvaluation Scores:\n", evaluation)
