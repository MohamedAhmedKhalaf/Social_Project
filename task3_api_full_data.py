import google.generativeai as genai
import pandas as pd
import numpy as np
import time
from statsmodels.stats.inter_rater import fleiss_kappa

# 1. API Configuration
# Replace with your actual Gemini API key
API_KEY = "YOUR_API_KEY_HERE" 
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# 2. Load the data
# Assuming 'refined_bank_data.csv' has columns: source, rating, text, subject_tag
df = pd.read_csv("refined_bank_data.csv")

def get_gemini_sentiment(text, version=1):
    """Function to call Gemini with two different prompt styles"""
    if version == 1:
        # Prompt Style: Standard Sentiment Analysis
        prompt = f"Label the sentiment of this bank review as Positive, Negative, or Neutral. Output ONLY the word. Text: {text}"
    else:
        # Prompt Style: Customer Satisfaction Perspective
        prompt = f"Is the customer in this review happy or unhappy? Answer ONLY with Positive, Negative, or Neutral. Text: {text}"
    
    try:
        response = model.generate_content(prompt)
        label = response.text.strip().capitalize()
        # Clean response to ensure it's valid
        if "Positive" in label: return "Positive"
        if "Negative" in label: return "Negative"
        return "Neutral"
    except Exception as e:
        print(f"API Error: {e}")
        return "Neutral"

# 3. Running the Annotation
print(f"Starting API Annotation for {len(df)} records. This will take some time due to Rate Limits...")
ann1_list = []
ann2_list = []

for i, text in enumerate(df['text']):
    # Get two different labels for every record
    l1 = get_gemini_sentiment(text, version=1)
    l2 = get_gemini_sentiment(text, version=2)
    
    ann1_list.append(l1)
    ann2_list.append(l2)
    
    # Progress Check
    if (i + 1) % 10 == 0:
        print(f"Annotated {i+1}/{len(df)} records...")
    
    # RATE LIMIT: Free tier allows 15 requests per minute. 
    # Since we do 2 labels per row, we wait 8 seconds per row to be safe.
    time.sleep(8)

# 4. Finalizing Data
df['ann_1'] = ann1_list
df['ann_2'] = ann2_list

# Inter-Annotator Agreement Calculation (Fleiss Kappa)
cat_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
agreement_matrix = np.zeros((len(df), 3))
for i in range(len(df)):
    agreement_matrix[i, cat_map[ann1_list[i]]] += 1
    agreement_matrix[i, cat_map[ann2_list[i]]] += 1

try:
    kappa = fleiss_kappa(agreement_matrix)
    print(f"\nFinal Fleiss Kappa Score: {kappa:.4f}")
except Exception as e:
    print(f"\nCould not calculate Fleiss Kappa: {e}")

# Define Ground Truth (Taking Annotator 1 as the gold standard)
df['ground_truth'] = df['ann_1']

# Reorder columns to exactly match your desired output format
final_columns = ['source', 'rating', 'text', 'subject_tag', 'ann_1', 'ann_2', 'ground_truth']

# Keep only the columns that exist (prevents errors if a column name is slightly different)
existing_columns = [col for col in final_columns if col in df.columns]
df = df[existing_columns]

# 5. Save the file exactly as needed for the next steps
df.to_csv("full_dataset.csv", index=False)
print("File successfully saved as: full_dataset.csv")