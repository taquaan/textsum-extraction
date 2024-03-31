from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from heapq import nlargest
from nltk.tokenize import sent_tokenize

def tFID_summary(text, n):
    try:
        sentences = sent_tokenize(text)

        # Create the TF-IDF Matrix
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Cosine similarities between each sentences and document
        sent_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

        # Select the top n sentences with highest scores
        summary_sentences = nlargest(n, range(len(sent_scores)), key=sent_scores.__getitem__)

        summary_tfidf = ' '.join([sentences[i] for i in sorted(summary_sentences)])

        summary_sentences = summary_tfidf.split('. ')
        formatted_summary = '.\n'.join(summary_sentences)

        name = "TF-IDF METHOD"
        name_len = (60 - len(name)) // 2
        print("\n\n" + "=" * name_len + " " + name + " " + "=" * name_len)
        print("\nTF-IDF's Summarized text:")
        print(formatted_summary)

        return formatted_summary
    except KeyboardInterrupt:
        print(" - Program terminated by user.")
    except Exception as e:
        raise RuntimeError(f" - Error occurred in TF-IDF Summary: {e}")
    

