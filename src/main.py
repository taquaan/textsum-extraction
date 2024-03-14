import numpy as np
import nltk
import re
from nltk import ne_chunk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from scipy import spatial
import networkx as nx

# Required download before start:
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

print("\nInput text:")
text = input()

demo='''Santiago is a Shepherd who has a recurring dream which is supposedly prophetic. Inspired on learning this, he undertakes a journey to Egypt to discover the meaning of life and fulfill his destiny. During the course of his travels, he learns of his true purpose and meets many characters, including an “Alchemist”, that teach him valuable lessons about achieving his dreams. Santiago sets his sights on obtaining a certain kind of “treasure” for which he travels to Egypt. The key message is, “when you want something, all the universe conspires in helping you to achieve it.” Towards the final arc, Santiago gets robbed by bandits who end up revealing that the “treasure” he was looking for is buried in the place where his journey began. The end.'''

# Text preprocess
sentences = sent_tokenize(text)

sentences_clean=[re.sub(r'[^\w\s]','',sentence.lower()) for sentence in sentences]

stop_words = stopwords.words('english')

sentence_tokens=[[words for words in sentence.split(' ') if words not in stop_words] for sentence in sentences_clean]

# NER
sentence_entities = []
for sentence in sentences:
    entities = []
    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence))):
        if hasattr(chunk, 'label'):
            entities.append(' '.join(c[0] for c in chunk))
    sentence_entities.append(entities)

w2v = Word2Vec(sentence_tokens, vector_size=1, min_count=1, epochs=1000)

sentence_embeddings = [[w2v.wv[word] for word in words if word in w2v.wv] for words in sentence_tokens]

max_len = max(len(tokens) for tokens in sentence_tokens)
sentence_embeddings = [np.pad(embedding, ((0, max_len - len(embedding)), (0, 0)), 'constant') for embedding in sentence_embeddings]

# Similarity Matrix
similarity_matrix = np.zeros([len(sentence_tokens), len(sentence_tokens)])

for i, row_embedding in enumerate(sentence_embeddings):
    for j, column_embedding in enumerate(sentence_embeddings):
        similarity_matrix[i][j] = 1 - spatial.distance.cosine(row_embedding.mean(axis=0), column_embedding.mean(axis=0))

# PageRank
nx_graph = nx.from_numpy_array(similarity_matrix)
scores = nx.pagerank(nx_graph)

top_sentence = {sentence: scores[index] for index, sentence in enumerate(sentences)}
top = dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:4])

print("\nSummarized text:")
for sent in sentences:
    if sent in top.keys():
        print(sent)
