import numpy as np
from nltk.tokenize import sent_tokenize
import networkx as nx
from gensim.models import Word2Vec

def preprocess_text(text):
    try:
        sentences = sent_tokenize(text)
        return sentences
    except Exception as e:
        print(f"Error occurred in preprocess_text: {e}")
        return []

def generate_summary(sentences, sent_len):
    try:
        # Word Embedding using Word2Vec
        sentence_tokens = [sentence.split() for sentence in sentences]
        w2v = Word2Vec(sentence_tokens, vector_size=50, min_count=1, epochs=1000)
        word_embeddings = {word: w2v.wv[word] for word in w2v.wv.index_to_key}
        
        # Sentence Embedding
        sentence_embeddings = [np.mean([word_embeddings[word] for word in sentence.split() if word in word_embeddings], axis=0) for sentence in sentences]
        
        # Similarity Matrix
        similarity_matrix = np.zeros([len(sentences), len(sentences)])
        for i, row_embedding in enumerate(sentence_embeddings):
            for j, column_embedding in enumerate(sentence_embeddings):
                similarity_matrix[i][j] = 1 - np.dot(row_embedding, column_embedding) / (np.linalg.norm(row_embedding) * np.linalg.norm(column_embedding))
        
        # PageRank Algorithm
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        # Get top sentences based on PageRank scores
        top_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)[:sent_len]
        
        # Sort top sentences by their original order
        top_sentences = sorted(top_sentences, key=lambda x: sentences.index(x[1]))
        
        return top_sentences
    except Exception as e:
        print(f"Error occurred in generate_summary: {e}")
        return []

def textRankSummary(text, sent_len):
    try:
        name = "TEXT RANK ALGORITHM METHOD"
        name_len = (60 - len(name)) // 2
        print("\n" + "=" * name_len + " " + name + " " + "=" * name_len)

        # Preprocess text
        sentences = preprocess_text(text)
        if not sentences:
            print("Error: No sentences found.")
        
        # Generate summary using Text Rank algorithm
        top_sentences = generate_summary(sentences, sent_len)
        if top_sentences:
            print("\nText Rank's Summarized text:")
            summary_paragraph = " ".join(sentence for score, sentence in top_sentences)
            print(summary_paragraph)
            final_summary = summary_paragraph
        else:
            print("Unable to generate summary.")
        
        if final_summary:
            return final_summary
        else:
            return "Unable to generate summary using Text Rank algorithm."
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

