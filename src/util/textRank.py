import numpy as np
import re
import spacy
import networkx as nx
from gensim.models import Word2Vec
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# Required downloads before start:
# !python -m spacy download en_core_web_sm

def preprocess_text(text):
    try:
        sentences = re.split(r'\.\s|\?\s|\!\s', text)
        return sentences
    except Exception as e:
        print(f"Error occurred in preprocess_text: {e}")
        return []

def extract_named_entities(sentences):
    try:
        nlp = spacy.load("en_core_web_sm")
        sentence_entities = []
        for sentence in sentences:
            doc = nlp(sentence)
            entities = [ent.text for ent in doc.ents]
            sentence_entities.append(entities)
        return sentence_entities
    except Exception as e:
        print(f"Error occurred in extract_named_entities: {e}")
        return []

def generate_summary(sentences):
    try:
        # Word Embedding using Word2Vec
        sentence_tokens = [sentence.split() for sentence in sentences]
        w2v = Word2Vec(sentence_tokens, vector_size=50, min_count=1, epochs=1000)
        # Precompute word embeddings
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
        # Get top sentences
        top_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)[:5]
        return top_sentences
    except Exception as e:
        print(f"Error occurred in generate_summary: {e}")
        return []

def perform_topic_modeling(sentences):
    try:
        # Tokenize sentences
        sentence_tokens = [sentence.split() for sentence in sentences]
        # Create Dictionary
        dictionary = Dictionary(sentence_tokens)
        # Create Corpus
        corpus = [dictionary.doc2bow(tokens) for tokens in sentence_tokens]
        # Train LDA model
        lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
        # Get topics
        topics = lda_model.print_topics(num_words=5)
        return topics
    except Exception as e:
        print(f"Error occurred in perform_topic_modeling: {e}")
        return []

def textRankSummary(text):
    try:
        name = "TEXT RANK ALGORITHM"
        name_len = (60 - len(name)) // 2
        print("\n" + "=" * name_len + " " + name + " " + "=" * name_len)

        sentences = preprocess_text(text)
        if not sentences:
            print("Error: No sentences found.")
        
        sentence_entities = extract_named_entities(sentences)
        if not sentence_entities:
            print("Error: Failed to extract named entities.")
        
        print("\nText Rank's Named Entities:")
        for i, entities in enumerate(sentence_entities):
            print(f"Sentence {i+1}: {entities}")
        
        top_sentences = generate_summary(sentences)
        if top_sentences:
            print("\nText Rank's Summarized text:")
            summary_paragraph = ". ".join(sentence for score, sentence in top_sentences)
            print(summary_paragraph)
            final_summary = summary_paragraph
        else:
            print("Unable to generate summary.")
        
        topics = perform_topic_modeling(sentences)
        if topics:
            print("\nText Rank's Topics:")
            for topic in topics:
                print(topic)
        else:
            print("Unable to perform topic modeling.")
        if final_summary:
            return final_summary
        else:
            return "Unable to generate summary using Text Rank algorithm."
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

