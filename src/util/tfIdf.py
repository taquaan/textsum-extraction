import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest

def tFIDSummary(text):
    try:
        nlp = spacy.load("en_core_web_sm")

        # Text preprocess
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]

        # TF-IDF Matrix
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Cosine similarity
        score = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[-1])[0]

        # Select the top sentences with highest score
        
    except Exception as e:
        print(f"Error occurred in preprocess_text: {e}")