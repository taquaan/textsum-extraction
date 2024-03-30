from metric import evaluate_using_rouge

def summary_text(text):
    # Define the length of the summarized text
    summary_len = 5
    
    # Run the summarize function
    summary = evaluate_using_rouge(text, summary_len)
    return summary