import pandas as pd
import re
import numpy as np # Import numpy if not already
import nltk
import ssl

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords') 
nltk.download('wordnet')   

# --- Function to disable SSL verification for NLTK download (Reuse from previous versions) ---
def download_nltk_resource_no_ssl(resource_id, resource_name):
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    print(f"Attempting to download NLTK resource: {resource_name} (SSL verification disabled)")
    download_successful = False
    try:
        nltk.download(resource_id, quiet=True)
        print(f"Successfully downloaded {resource_name}.")
        if resource_id == 'stopwords':
             _ = nltk.corpus.stopwords.words('english')
        elif resource_id == 'wordnet':
             _ = nltk.stem.WordNetLemmatizer().lemmatize('test')
        print(f"NLTK resource {resource_name} loaded successfully after download.")
        download_successful = True
    except Exception as e:
        print(f"Error during/after NLTK resource download {resource_name}: {e}")
        print("Skipping advanced cleaning steps that require this resource.")
    finally:
        ssl._create_default_https_context = ssl.create_default_context
    return download_successful


# --- Initialize NLTK components (Reuse robust initialization from previous versions) ---
print("Initializing NLTK resources...")
stop_words = None
lemmatizer = None
nltk_resources_available = True
try:
    stop_words = set(nltk.corpus.stopwords.words('english'))
    print("NLTK stopwords loaded from local cache.")
except LookupError:
    print("NLTK stopwords not found locally.")
    if not download_nltk_resource_no_ssl('stopwords', 'Stopwords Corpus'):
        nltk_resources_available = False
    else:
         try:
             stop_words = set(nltk.corpus.stopwords.words('english'))
         except Exception as e:
              print(f"Failed to load stopwords even after download attempt: {e}")
              nltk_resources_available = False

if nltk_resources_available:
    try:
        lemmatizer = nltk.stem.WordNetLemmatizer()
        _ = lemmatizer.lemmatize('test') # Verify by using it
        print("NLTK WordNet loaded from local cache.")
    except LookupError:
        print("NLTK WordNet not found locally.")
        if not download_nltk_resource_no_ssl('wordnet', 'WordNet Corpus'):
            nltk_resources_available = False
        else:
            try:
                lemmatizer = nltk.stem.WordNetLemmatizer()
                _ = lemmatizer.lemmatize('test') # Verify again
            except Exception as e:
                print(f"Failed to load WordNet even after download attempt: {e}")
                nltk_resources_available = False
    except Exception as e:
         print(f"Error initializing lemmatizer: {e}")
         nltk_resources_available = False
else:
     print("Skipping further NLTK initialization checks as stopwords failed.")


# --- General Cleaning ---
df = pd.read_csv("permit_data.csv") 

print("Original Columns:", df.columns)

df = df[["PERMITTYPE", "DESCRIPTION"]].copy() 

# --- Basic Cleaning ---
df['PERMITTYPE'] = df['PERMITTYPE'].fillna('').astype(str).str.strip().str.lower()
df['DESCRIPTION'] = df['DESCRIPTION'].fillna('')
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace('"', '')
df['DESCRIPTION'] = df['DESCRIPTION'].str.lower()
df['DESCRIPTION'] = df['DESCRIPTION'].str.split('===', n=1, expand=True)[0].fillna('')
df['DESCRIPTION'] = df['DESCRIPTION'].str.strip()

# --- Extract "PURPOSE:" text (Reuse safer version) ---
print("Attempting to extract text after 'purpose [optional spaces]:[optional spaces]' where found...")
purpose_regex = r'purpose\s*:\s*'
contains_purpose = df['DESCRIPTION'].str.contains(purpose_regex, na=False, regex=True)
num_found = contains_purpose.sum()

if num_found > 0:
    print(f"Found {num_found} rows containing the purpose pattern. Extracting relevant text...")
    split_result = df.loc[contains_purpose, 'DESCRIPTION'].str.split(purpose_regex, n=1, expand=True, regex=True)
    if 1 in split_result.columns:
        extracted_purpose = split_result[1].str.strip().fillna('')
        df.loc[contains_purpose, 'DESCRIPTION'] = extracted_purpose
        print("Update complete.")
    else:
        print("Warning: Split operation did not produce text after the pattern. Setting description to empty for these rows.")
        df.loc[contains_purpose, 'DESCRIPTION'] = ''
else:
    print("No rows containing the purpose pattern found. Skipping extraction.")

# --- Further Cleaning (Regex, Specific Strings, Colon Removal) ---
print("Applying further cleaning (punctuation, numbers, specific strings)...")
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(r'[^\w\s]', '', regex=True).fillna('')
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(r'\d+', '', regex=True).fillna('')
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace("cutapplicationid", '', regex=False)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace("cpuc compliance", '', regex=False)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(r'\s+', ' ', regex=True).str.strip()
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(" x ", '', regex=False)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(" e ", '', regex=False)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(" n ", '', regex=False)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(" f ", '', regex=False)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(" kw ", '', regex=False)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(" nd ", '', regex=False)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(" pv ", '', regex=False)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(" b ", '', regex=False)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(" s ", '', regex=False)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(" mh ", '', regex=False)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(" hm ", '', regex=False)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(" h ", '', regex=False)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(" ti ", '', regex=False)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(" tc ", '', regex=False)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(" k ", '', regex=False)


# --- Filter 1: Remove rows with blank PERMITTYPE or DESCRIPTION ---
print(f"Rows before filtering blanks: {len(df)}")
rows_to_keep_initial = (df['PERMITTYPE'] != '') & (df['DESCRIPTION'] != '')
df = df[rows_to_keep_initial].copy()
print(f"Rows after filtering blanks: {len(df)}")

# --- [NEW STEP] Advanced Cleaning (NLTK Stopwords and Lemmatization) ---
if nltk_resources_available and not df.empty: 
    print("Applying NLTK cleaning (stopwords, lemmatization)...")

    def clean_text_advanced(text):
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
             return ''
        global lemmatizer, stop_words 
        if lemmatizer is None or stop_words is None:
            return text 
        try:
            words = text.split()
            words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
            return ' '.join(words)
        except Exception as e:
            return '' #

    df['DESCRIPTION'] = df['DESCRIPTION'].apply(clean_text_advanced)
    print("Finished NLTK cleaning.")
elif df.empty:
     print("Skipping NLTK cleaning because DataFrame is empty.")
else:
    print("Skipping NLTK advanced cleaning due to missing resources or initialization errors.")


# --- Final Output ---
print("Saving cleaned data...")
df.to_csv("permit_data_cleaned.csv", index=False)

