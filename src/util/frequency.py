import nltk

# Required downloading punkt and stopwords
# nltk.download('punkt')
# nltk.download('stopwords')

from collections import Counter
from nltk.corpus import stopwords

from nltk.tokenize import sent_tokenize, word_tokenize

def frequency_summary(text, n):
    try:
        sentences = sent_tokenize(text)

        stop_words = set(stopwords.words('english'))

        # Get all important words
        words = [word.lower() for word in word_tokenize(text) if word.lower() not in stop_words and word.isalnum()]

        # Count the words
        word_freq = Counter(words)

        sent_scores = {}

        for sentence in sentences:
            # Get all the word in a sentence
            sentence_words = [word.lower() for word in word_tokenize(sentence) if word.lower() not in stop_words and word.isalnum()]
            # Get the frequency of words in sentence based on word_freq
            sentence_score = sum([word_freq[word] for word in sentence_words])
            if len(sentence_words) < 20:
                sent_scores[sentence] = sentence_score

        summary_sentences = sorted(sent_scores, key=sent_scores.get, reverse=True)[:n]
        summary = ' '.join(summary_sentences)

        summary_sentences = summary.split('. ')
        formatted_summary = '.\n'.join(summary_sentences)

        name = "FREQUENCY-BASED METHOD"
        name_len = (60 - len(name)) // 2
        print("\n\n" + "=" * name_len + " " + name + " " + "=" * name_len)
        print("Summarized text:")
        print(formatted_summary)

        return formatted_summary
    except KeyboardInterrupt:
        print(" - Program terminated by user.")
    except Exception as e:
        raise RuntimeError(f" - Error occurred in TF-IDF Summary: {e}")
