# 📝 Text Summarization using BART and CNN/DailyMail

A natural language processing project that performs both **extractive** and **abstractive summarization** using the CNN/Daily Mail news dataset. It utilizes a pre-trained BART model from Hugging Face for abstractive summarization and evaluates results using ROUGE metrics.

---

## 📌 Project Objectives

- Apply extractive and abstractive summarization techniques to news articles.
- Use SpaCy and NLTK for text preprocessing and sentence segmentation.
- Use the pre-trained BART model (`facebook/bart-large-cnn`) for abstractive summarization.
- Evaluate the summary quality using ROUGE scores.

---

## 🧾 Dataset

- **Name:** CNN / Daily Mail
- **Version:** 3.0.0
- **Source:** Hugging Face Datasets Library
- **Split Used:** `test[0]` sample article for demonstration

---

## 🧰 Tools & Libraries

- **Programming Language:** Python
- **Key Libraries:**
  - `transformers` (Hugging Face)
  - `datasets` (Hugging Face)
  - `spacy`, `nltk`
  - `torch` (PyTorch)
  - `rouge_score`

---

## 🔍 Text Summarization Methods

### 📌 1. Extractive Summarization
- Uses NLTK sentence tokenizer.
- Selects the first `n` sentences (default = 3).

```python
def extractive_summary(text, num_sentences=3):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:num_sentences])
📌 2. Abstractive Summarization
Uses the BART model (facebook/bart-large-cnn) via Hugging Face.

python
Copy
Edit
def abstractive_summary(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
📊 Evaluation Metrics
Summary quality is evaluated using ROUGE scores:

ROUGE-1

ROUGE-2

ROUGE-L

python
Copy
Edit
def evaluate_summary(original, summary):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    return scorer.score(original, summary)
▶️ How to Run
Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/text-summarization-cnn-dailymail.git
cd text-summarization-cnn-dailymail
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Download NLTK tokenizer data

python
Copy
Edit
import nltk
nltk.download('punkt')
Run the script

bash
Copy
Edit
python text_summarization.py
🗂️ Project Structure
bash
Copy
Edit
├── text_summarization.py         # Main Python script
├── README.md                     # Documentation
├── requirements.txt              # Python dependencies
💡 Sample Output
text
Copy
Edit
Original Text:
 CNN anchor Bob Costas introduces a clip showing Michelle Obama on the "Tonight Show." She appears in a comedy bit, with Will Ferrell and Jimmy Fallon. Costas comments on her jump rope moves: "Impressive."

Extractive Summary:
 CNN anchor Bob Costas introduces a clip showing Michelle Obama on the "Tonight Show." She appears in a comedy bit, with Will Ferrell and Jimmy Fallon. Costas comments on her jump rope moves: "Impressive."

Abstractive Summary:
 Michelle Obama made a surprise appearance on the "Tonight Show," showing off her jump rope moves. Bob Costas was impressed by her performance in a comedy bit with Jimmy Fallon and Will Ferrell.

Evaluation Scores:
 {'rouge1': ..., 'rouge2': ..., 'rougeL': ...}
👤 Author
Talha Saeed
📍 Data Scientist
🔗 GitHub Profile
