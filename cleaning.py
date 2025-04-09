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
    # (Keep the function definition from previous versions - includes try/except for context and finally block)
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        # Temporarily set the unverified context for NLTK download
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
        # Restore original SSL context setting IMPORTANT
        ssl._create_default_https_context = ssl.create_default_context
    return download_successful


# --- Initialize NLTK components (Reuse robust initialization from previous versions) ---
# (This section attempts to load stopwords/wordnet and downloads if needed using the SSL fix)
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


# --- Start of your existing code (modified slightly) ---
df = pd.read_csv("permit_data.csv", nrows=100000) # Keep nrows for now

print("Original Columns:", df.columns)

df = df[["PERMITTYPE", "DESCRIPTION"]].copy() # Use .copy() to avoid SettingWithCopyWarning

# --- Basic Cleaning ---
df['PERMITTYPE'] = df['PERMITTYPE'].fillna('').astype(str).str.strip().str.lower() # Clean PERMITTYPE
df['DESCRIPTION'] = df['DESCRIPTION'].fillna('')
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace('"', '')
df['DESCRIPTION'] = df['DESCRIPTION'].str.lower()
# Handle potential None after split
df['DESCRIPTION'] = df['DESCRIPTION'].str.split('===', n=1, expand=True)[0].fillna('')
df['DESCRIPTION'] = df['DESCRIPTION'].str.strip()

# --- Extract "PURPOSE:" text (Reuse safer version) ---
# (This block should be here, before regex cleaning removes colons)
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
# Remove all non-word/space chars (including colons now)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(r'[^\w\s]', '', regex=True).fillna('')
# Remove numbers
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(r'\d+', '', regex=True).fillna('')
# Remove specific unwanted strings
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace("cutapplicationid", '', regex=False)
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace("cpuc compliance", '', regex=False)
# Consolidate whitespace and strip
df['DESCRIPTION'] = df['DESCRIPTION'].str.replace(r'\s+', ' ', regex=True).str.strip()

# --- Filter 1: Remove rows with blank PERMITTYPE or DESCRIPTION ---
print(f"Rows before filtering blanks: {len(df)}")
rows_to_keep_initial = (df['PERMITTYPE'] != '') & (df['DESCRIPTION'] != '')
# Use .copy() here to prepare for NLTK step if you modify df later
df = df[rows_to_keep_initial].copy()
print(f"Rows after filtering blanks: {len(df)}")

# --- [NEW STEP] Advanced Cleaning (NLTK Stopwords and Lemmatization) ---
if nltk_resources_available and not df.empty: # Check flag and if df is not empty
    print("Applying NLTK cleaning (stopwords, lemmatization)...")

    # Define the cleaning function (ensure lemmatizer/stopwords are checked)
    def clean_text_advanced(text):
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
             return ''
        # Use the globally initialized variables
        global lemmatizer, stop_words # Refer to the variables loaded earlier
        if lemmatizer is None or stop_words is None:
            # This case should ideally be handled by the nltk_resources_available flag,
            # but this is an extra safety check.
            return text # Return text unmodified if resources aren't ready
        try:
            words = text.split()
            # Lemmatize words that are not stopwords
            words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
            return ' '.join(words)
        except Exception as e:
            # print(f"Error processing text '{text[:50]}...': {e}") # Optional debug
            return '' # Return empty on error during apply

    # Apply the function to the DESCRIPTION column
    df['DESCRIPTION'] = df['DESCRIPTION'].apply(clean_text_advanced)
    print("Finished NLTK cleaning.")
elif df.empty:
     print("Skipping NLTK cleaning because DataFrame is empty.")
else:
    print("Skipping NLTK advanced cleaning due to missing resources or initialization errors.")


# --- Final Output ---
print("Saving cleaned data...")
try:
    if not df.empty:
        df.to_csv("permit_data_cleaned.csv", index=False)
        print(f"Cleaned data saved to permit_data_cleaned.csv")
        print("\nSample of final cleaned data:")
        print(df.head())
    else:
        print("Cleaned DataFrame is empty. No file saved.")
except Exception as e:
    print(f"Error saving file '{OUTPUT_FILE}': {e}")