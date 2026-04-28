import argparse
import pandas as pd
import re
import string
import nltk
import emoji
import os
from textblob import Word
from symspellpy import SymSpell, Verbosity

# Setup NLTK
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class BankFivePreprocessor:
    def __init__(self):
        # Initialize SymSpell for spelling correction
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

        # Load dictionary directly from symspellpy package
        import symspellpy
        dictionary_path = os.path.join(
            os.path.dirname(symspellpy.__file__),
            "frequency_dictionary_en_82_765.txt"
        )
        if not os.path.exists(dictionary_path):
            raise FileNotFoundError(f"SymSpell dictionary not found at {dictionary_path}")
        
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    def remove_noise(self, text):
        """Removes Trustpilot repetitions and ellipses artifacts."""
        if "…." in text:
            parts = text.split("….")
            text = parts[-1]  # Keep the actual review body
        
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        text = text.replace("exclamation mark", "!")
        return text.strip()

    def clean_basic(self, text):
        """Basic cleaning: Emojis to text, punctuation, and whitespace."""
        text = emoji.demojize(text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def fix_spelling(self, text):
        """Uses SymSpell to correct common typos."""
        suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
        return suggestions[0].term if suggestions else text

    def lemmatize_text(self, text):
        """Uses TextBlob to convert words to their base dictionary form."""
        words = text.split()
        lemmatized = [Word(w).lemmatize() for w in words]
        return " ".join(lemmatized)

    def extract_subjects(self, text):
        """Categorization tagging based on banking keywords."""
        text = text.lower()
        categories = {
            "Loans": r"loan|mortgage|heloc|bridge|rates|refinancing",
            "Customer Service": r"teller|staff|personnel|manager|helpful|polite|employee|service",
            "Digital Banking": r"app|online|web|site|atm|debit|card|face id"
        }
        tags = [category for category, pattern in categories.items() if re.search(pattern, text)]
        return ", ".join(tags) if tags else "General"


def main():
    parser = argparse.ArgumentParser(description="Robust Preprocessing Pipeline for BankFive Data")
    
    parser.add_argument("--input", type=str, default="bankfive_sentiment_data.csv", help="Input CSV file")
    parser.add_argument("--output", type=str, default="refined_bank_data.csv", help="Output CSV file")
    
    parser.add_argument("--lower", action="store_true", help="Convert text to lowercase")
    parser.add_argument("--remove_noise", action="store_true", help="Remove Trustpilot ellipses and URLs")
    parser.add_argument("--fix_spelling", action="store_true", help="Apply SymSpell correction")
    parser.add_argument("--lemmatize", action="store_true", help="Apply TextBlob lemmatization")
    parser.add_argument("--tag_subjects", action="store_true", help="Extract Categorization Tags via Regex")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    df = pd.read_csv(args.input)
    pipeline = BankFivePreprocessor()

    print(f"Processing {len(df)} records...")

    if args.remove_noise:
        print("-> Removing noise...")
        df['text'] = df['text'].apply(pipeline.remove_noise)
        
    if args.tag_subjects:
        print("-> Tagging subjects...")
        df['subject_tag'] = df['text'].apply(pipeline.extract_subjects)

    if args.lower:
        print("-> Lowercasing...")
        df['text'] = df['text'].str.lower()

    if args.fix_spelling:
        print("-> Fixing spelling (SymSpell)...")
        df['text'] = df['text'].apply(pipeline.clean_basic)
        df['text'] = df['text'].apply(pipeline.fix_spelling)

    if args.lemmatize:
        print("-> Lemmatizing (TextBlob)...")
        df['text'] = df['text'].apply(pipeline.lemmatize_text)

    df.to_csv(args.output, index=False)
    print(f"Success! Refined data saved to: {args.output}")
    if 'subject_tag' in df.columns:
        print(df[['subject_tag', 'text']].head())
    else:
        print(df[['text']].head())


if __name__ == "__main__":
    main()