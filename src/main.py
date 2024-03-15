from util.textRank import textRankSummary

def main():
    print("\nInput:")
    text = input()
    textRank = textRankSummary(text)

if __name__ == "__main__":
    main()