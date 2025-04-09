import pandas as pd
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords') 
nltk.download('wordnet')   

df = pd.read_csv("permit_data.csv", nrows=100000)

print("Original Columns:", df.columns)

df = df[["PERMITTYPE", "DESCRIPTION"]].copy() # Use .copy() to avoid SettingWithCopyWarning

df['DESCRIPTION'] = df['DESCRIPTION'].fillna('')
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace('"', '')
#df['DESCRIPTION'] = df['DESCRIPTION'].str.lower()
df['DESCRIPTION'] = df['DESCRIPTION'].str.split('===', n=1, expand=True)[0]

# --- Further Cleaning for Classification ---
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(r'[^\w\s]', '', regex=True)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(r'\d+', '', regex=True)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(r'\s+', ' ', regex=True).str.strip()

print(f"Rows before filtering: {len(df)}")
# Condition for keeping rows: PERMITTYPE is not blank AND DESCRIPTION is not blank
rows_to_keep = (df['PERMITTYPE'] != '') & (df['DESCRIPTION'] != '')
df = df[rows_to_keep]
print(f"Rows after filtering: {len(df)}")

df.to_csv("permit_data_cleaned.csv", index=False)