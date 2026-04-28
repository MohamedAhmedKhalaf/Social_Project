import pandas as pd
import numpy as np
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

# Ignore sklearn zero-division warnings for clean terminal output
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('sentiwordnet', quiet=True)
nltk.download('wordnet', quiet=True)

# ==========================================
# 1. TEXT REPRESENTATION
# ==========================================

def get_bow_representation(texts):
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(texts)
    return bow_matrix, vectorizer

def load_glove_model(glove_file):
    print("Loading GloVe Model...")
    glove_dict = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_dict[word] = vector
    print(f"Loaded {len(glove_dict)} words from GloVe.")
    return glove_dict

def get_glove_representation(texts, glove_dict, vector_size=100):
    features =[]
    for text in texts:
        tokens = str(text).lower().split()
        vectors = [glove_dict[w] for w in tokens if w in glove_dict]
        if vectors:
            features.append(np.mean(vectors, axis=0))
        else:
            features.append(np.zeros(vector_size))
    return np.array(features)

# ==========================================
# 2. LEXICAL-BASED MODELLING
# ==========================================

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'): return wn.ADJ
    elif treebank_tag.startswith('V'): return wn.VERB
    elif treebank_tag.startswith('N'): return wn.NOUN
    elif treebank_tag.startswith('R'): return wn.ADV
    else: return None

def sentiwordnet_predict(text):
    sentiment_score = 0.0
    tokens = word_tokenize(str(text))
    tagged = pos_tag(tokens)
    for word, tag in tagged:
        wn_tag = get_wordnet_pos(tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB): continue
        synsets = list(swn.senti_synsets(word, wn_tag))
        if not synsets: continue
        synset = synsets[0]
        sentiment_score += (synset.pos_score() - synset.neg_score())
        
    if sentiment_score > 0: return "Positive"
    elif sentiment_score < 0: return "Negative"
    else: return "Neutral"

def load_bing_liu_dict(pos_file, neg_file):
    with open(pos_file, 'r', encoding='latin-1') as f:
        pos_words = set([line.strip() for line in f if not line.startswith(';') and line.strip()])
    with open(neg_file, 'r', encoding='latin-1') as f:
        neg_words = set([line.strip() for line in f if not line.startswith(';') and line.strip()])
    return pos_words, neg_words

def bing_liu_predict(text, pos_words, neg_words):
    negations = {"not", "no", "never", "didn't", "don't", "doesn't", "wasn't", "weren't", "isn't", "aren't", "cannot", "cant", "couldn't", "wouldn't", "shouldn't", "lack", "none"}
    score = 0
    tokens = str(text).lower().split()
    for i, word in enumerate(tokens):
        multiplier = -1 if i > 0 and tokens[i-1] in negations else 1
        if word in pos_words: score += (1 * multiplier)
        elif word in neg_words: score += (-1 * multiplier)
            
    if score > 0: return "Positive"
    elif score < 0: return "Negative"
    else: return "Neutral"

# ==========================================
# 3. METRICS HELPER FUNCTION
# ==========================================
def calculate_metrics(y_true, y_pred):
    """Calculates all requested classification metrics"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4)

# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def run_ml_pipeline():
    # 1. Load Master Labeled Data
    LABELED_CSV_FILE = "full_dataset.csv" 
    print(f"Loading master dataset from {LABELED_CSV_FILE}...")
    try:
        master_df = pd.read_csv(LABELED_CSV_FILE)
        # Drop rows where ground truth is completely missing just in case
        master_df = master_df.dropna(subset=['ground_truth']).reset_index(drop=True)
        ground_truth_labels = master_df['ground_truth'].values
    except FileNotFoundError:
        print(f"❌ Error: '{LABELED_CSV_FILE}' not found.")
        return

    # 2. Load Dictionary and GloVe
    pos_words, neg_words = load_bing_liu_dict('positive-words.txt', 'negative-words.txt')
    glove_dict = load_glove_model('glove.6B.100d.txt')
    
    # 3. Define datasets & Short names for columns
    datasets = {
        "Clean Basic": ("full_balanced_basic_clean.csv", "Clean"),
        "Lemmatized": ("full_lemmatized_clean.csv", "Lemma"),
        "Refined Full": ("full_dataset.csv", "Refined")
    }
    
    all_metrics =[]
    
    # Iterate through each preprocessing scheme
    for desc, (file_path, short_name) in datasets.items():
        print(f"\n{'='*50}\nEvaluating Preprocessing Scheme: {desc}\n{'='*50}")
        try:
            df = pd.read_csv(file_path)
            # Ensure index alignment with master dataframe
            df = df.iloc[:len(master_df)].reset_index(drop=True) 
        except FileNotFoundError:
            print(f"File {file_path} not found. Skipping...")
            continue
            
        X_text = df['text'].values
        y = ground_truth_labels
        
        # --- A. LEXICAL MODELS ---
        print("--> Generating Lexical Predictions...")
        swn_preds =[sentiwordnet_predict(text) for text in X_text]
        bing_preds =[bing_liu_predict(text, pos_words, neg_words) for text in X_text]
        
        # Save to Master DataFrame
        master_df[f'Pred_SWN_{short_name}'] = swn_preds
        master_df[f'Pred_Bing_{short_name}'] = bing_preds
        
        # Calculate & Store Metrics
        metrics_swn = calculate_metrics(y, swn_preds)
        metrics_bing = calculate_metrics(y, bing_preds)
        
        all_metrics.append({"Scheme": desc, "Representation": "Lexical", "Model": "SentiWordNet", "Accuracy": metrics_swn[0], "Precision": metrics_swn[1], "Recall": metrics_swn[2], "F1_Score": metrics_swn[3]})
        all_metrics.append({"Scheme": desc, "Representation": "Lexical", "Model": "Bing Liu Custom", "Accuracy": metrics_bing[0], "Precision": metrics_bing[1], "Recall": metrics_bing[2], "F1_Score": metrics_bing[3]})

        # --- B. TEXT REPRESENTATIONS ---
        X_bow, vectorizer = get_bow_representation(X_text)
        X_glove = get_glove_representation(X_text, glove_dict)
        
        # Train/Test Split (Important: We calculate ML metrics ONLY on test set to be statistically accurate)
        idx_train, idx_test, y_train, y_test = train_test_split(
            np.arange(len(X_text)), y, test_size=0.2, random_state=42
        )
        
        representations = {
            "Bag-Of-Words": (X_bow, "BoW"),
            "GloVe 100d": (X_glove, "GloVe")
        }
        
        # --- C. MACHINE LEARNING MODELS ---
        for rep_name, (X_features, rep_short) in representations.items():
            print(f"--> Training ML Models using {rep_name}...")
            
            # Split features
            X_train = X_features[idx_train]
            X_test = X_features[idx_test]
            
            # Convert sparse BoW to array for Naive Bayes/DT compatibility if needed
            if hasattr(X_train, "toarray"):
                X_train_arr = X_train.toarray()
                X_test_arr = X_test.toarray()
                X_full_arr = X_features.toarray()
            else:
                X_train_arr = X_train
                X_test_arr = X_test
                X_full_arr = X_features
            
            # -- 1. Naive Bayes --
            nb_clf = MultinomialNB() if rep_name == "Bag-Of-Words" else GaussianNB()
            nb_clf.fit(X_train_arr, y_train)
            
            # Metrics evaluated on unseen test data ONLY
            nb_test_preds = nb_clf.predict(X_test_arr)
            nb_metrics = calculate_metrics(y_test, nb_test_preds)
            
            # Full dataset prediction (to attach to the master CSV)
            master_df[f'Pred_NB_{rep_short}_{short_name}'] = nb_clf.predict(X_full_arr)
            
            all_metrics.append({"Scheme": desc, "Representation": rep_name, "Model": "Naive Bayes", "Accuracy": nb_metrics[0], "Precision": nb_metrics[1], "Recall": nb_metrics[2], "F1_Score": nb_metrics[3]})

            # -- 2. Decision Tree --
            dt_clf = DecisionTreeClassifier(random_state=42)
            dt_clf.fit(X_train_arr, y_train)
            
            # Metrics evaluated on unseen test data ONLY
            dt_test_preds = dt_clf.predict(X_test_arr)
            dt_metrics = calculate_metrics(y_test, dt_test_preds)
            
            # Full dataset prediction (to attach to the master CSV)
            master_df[f'Pred_DT_{rep_short}_{short_name}'] = dt_clf.predict(X_full_arr)
            
            all_metrics.append({"Scheme": desc, "Representation": rep_name, "Model": "Decision Tree", "Accuracy": dt_metrics[0], "Precision": dt_metrics[1], "Recall": dt_metrics[2], "F1_Score": dt_metrics[3]})

    # ==========================================
    # 5. EXPORT RESULTS
    # ==========================================
    
    # 1. Export the Master Dataset with all predictions
    master_output_filename = "full_dataset_with_predictions.csv"
    master_df.to_csv(master_output_filename, index=False)
    
    # 2. Export the highly detailed metrics CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_output_filename = "detailed_model_metrics.csv"
    metrics_df.to_csv(metrics_output_filename, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Success! Processing complete.")
    print(f"📄 Generated {master_output_filename} (contains original text + 18 model predictions)")
    print(f"📊 Generated {metrics_output_filename} (Accuracy, Precision, Recall, F1 for all models)")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_ml_pipeline()