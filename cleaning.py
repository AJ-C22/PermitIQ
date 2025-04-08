import pandas as pd
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords') 
nltk.download('wordnet')   

df = pd.read_csv("permit_data.csv", nrows=100000)

print("Original Columns:", df.columns)

df = df[["PMPERMITID", "PERMITTYPE", "DESCRIPTION"]].copy() # Use .copy() to avoid SettingWithCopyWarning

df['DESCRIPTION'] = df['DESCRIPTION'].fillna('')
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace('"', '')
df['DESCRIPTION'] = df['DESCRIPTION'].str.lower()
df['DESCRIPTION'] = df['DESCRIPTION'].str.split('===', n=1, expand=True)[0]
df['DESCRIPTION'] = df['DESCRIPTION'].str.strip()

# --- Further Cleaning for Classification ---
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(r'[^\w\s]', '', regex=True)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(r'\d+', '', regex=True)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(r'\s+', ' ', regex=True).str.strip()


# --- Advanced Cleaning (Requires NLTK/spaCy) ---
# Example using NLTK (uncomment and adapt if needed)
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()
#
# def clean_text_advanced(text):
#     words = text.split()
#     words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
#     return ' '.join(words)
#
# df['DESCRIPTION'] = df['DESCRIPTION'].apply(clean_text_advanced)


print("Cleaned Columns:", df.columns) # Should still be the same columns

# Ensure PMPERMITID is string if it's an identifier not used numerically
df['PMPERMITID'] = df['PMPERMITID'].astype(str)

df.to_csv("permit_data_cleaned.csv", index=False)

print("Sample cleaned descriptions:")
print(df['DESCRIPTION'].head())