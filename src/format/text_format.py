import re

# abbreviations.py
english_abbreviations = [
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Jr.",
    "Sr.",
    "Prof.",
    "St.",
    "Ave.",
    "Blvd.",
    "Dept.",
    "Corp.",
    "Ltd.",
    "Inc.",
    "e.g.",
    "etc.",
    "i.e.",
    "Rev.",
    "Co.",
    "Fig."
]

def text_format(text): 
    # Split the text into sentences using regular expressions
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    
    # Add "\n" to the end of each sentence, except for exceptions
    modified_text = ""
    for sentence in sentences:
        if not any(sentence.endswith(exc) for exc in english_abbreviations):
            sentence += "\n"
        modified_text += sentence
    
    return modified_text