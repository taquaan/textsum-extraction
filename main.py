import numpy as np
import nltk
import re
from nltk.corpus import stopwords

#Download nltk.punkt and nltk.stopwords before start
# nltk.download('punkt')
# nltk.download('stopwords')

text='''Santiago is a Shepherd who has a recurring dream which is supposedly prophetic. Inspired on learning this, he undertakes a journey to Egypt to discover the meaning of life and fulfill his destiny. During the course of his travels, he learns of his true purpose and meets many characters, including an “Alchemist”, that teach him valuable lessons about achieving his dreams. Santiago sets his sights on obtaining a certain kind of “treasure” for which he travels to Egypt. The key message is, “when you want something, all the universe conspires in helping you to achieve it.” Towards the final arc, Santiago gets robbed by bandits who end up revealing that the “treasure” he was looking for is buried in the place where his journey began. The end.'''

# Remove punctuations and stopwords
def word_preprocess(text):
    sentences = nltk.sent_tokenize(text)
    clean = [re.sub(r'[^\w\s]','',sentence.lower()) for sentence in sentences]
    stop_words = stopwords.words('english')
    tokens =[[words for words in sentence.split(' ') if words not in stop_words] for sentence in clean]
    return tokens

tokens = word_preprocess(text)
print(tokens)