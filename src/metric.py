from format.text_format import text_format
from summarizer import Summarizer
from rouge import Rouge
from util.textRank import textRankSummary
from util.frequency import frequency_summary
from util.tfIdf import tFID_summary

def evaluate_using_rouge(original_text, sum_len):
    try:
        text = text_format(original_text)
        rouge = Rouge()
        model = Summarizer()

        reference_summary = model(text, num_sentences=sum_len)

        tr_summary = textRankSummary(text, sum_len)
        frq_summary = frequency_summary(text, sum_len)
        tfidf_summary = tFID_summary(text, sum_len)

        rouge_scores = {
            'Text Rank': rouge.get_scores(reference_summary, tr_summary)[0]['rouge-1']['f'],
            'Frequency': rouge.get_scores(reference_summary, frq_summary)[0]['rouge-1']['f'],
            'TF-IDF': rouge.get_scores(reference_summary, tfidf_summary)[0]['rouge-1']['f']
        }

        name = "ROUGE EVALUATION:"
        name_len = (60 - len(name)) // 2
        print("\n\n" + "=" * name_len + " " + name + " " + "=" * name_len)

        print("\nROUGE's reference summary:")
        print(reference_summary)

        print("\nROUGE scores:")
        for method, score in rouge_scores.items():
            print(f" - {method}: {score}")

        max_score = max(rouge_scores.values())
        max_method = [method for method, score in rouge_scores.items() if score == max_score]

        name = "FINAL RESULT"
        name_len = (60 - len(name)) // 2
        print("\n\n" + "=" * name_len + " " + name + " " + "=" * name_len)
        if max_method[0] == "Frequency":
            print("\nFrequency-based summary have the highest ROUGE score:")
            print(frq_summary)
            return frq_summary
        elif max_method[0] == "TF-IDF":
            print("\nTF-IDF method have the highest ROUGE score:")
            print(tfidf_summary)
            return tfidf_summary
        elif max_method[0] == "Text Rank":
            print("\nText Rank method have the highest ROUGE score:")
            print(tr_summary)
            return tr_summary
    except KeyboardInterrupt:
        print(" - Program terminated by user.")
    except Exception as e:
        raise RuntimeError(f" - Error occurred in Evaluating process: {e}")
