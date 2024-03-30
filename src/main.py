from metric import evaluate_using_rouge

def main():
    print("\nInput:")
    text = input()
    summary_len = 5

    evaluate_using_rouge(text, summary_len)
if __name__ == "__main__":
    main()