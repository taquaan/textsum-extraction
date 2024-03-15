from util.textRank import textRankSummary
from util.tfIdf import tFIDSummary

def main():
    print("\nInput:")
    text = input()
    # textRank = textRankSummary(text)
    tfIdf = tFIDSummary(text)

if __name__ == "__main__":
    main()