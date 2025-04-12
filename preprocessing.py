import re
import nltk
import ssl
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv # Needed if function uses env vars, though not directly here

# --- NLTK Resource Download Function (with SSL fix) ---
# (Copied from st.py for self-contained functionality)
def download_nltk_resource_no_ssl(resource_id, resource_name):
    try: _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError: pass
    else: ssl._create_default_https_context = _create_unverified_https_context

    print(f"(Preprocessing) Attempting to download NLTK resource: {resource_name}")
    download_successful = False
    try:
        nltk.download(resource_id, quiet=True)
        # Verify download by trying to load
        if resource_id == 'stopwords': _ = nltk.corpus.stopwords.words('english')
        elif resource_id == 'wordnet': _ = nltk.stem.WordNetLemmatizer().lemmatize('test')
        print(f"(Preprocessing) Successfully downloaded/verified {resource_name}.")
        download_successful = True
    except Exception as e: print(f"(Preprocessing) Error downloading/verifying {resource_name}: {e}")
    finally: ssl._create_default_https_context = ssl.create_default_context # Restore
    return download_successful

# --- Initialize NLTK Components (Load once when module is imported) ---
print("(Preprocessing) Initializing NLTK resources...")
nltk_resources_available = True
stop_words = None
lemmatizer = None

try:
    stop_words = set(stopwords.words('english'))
    print("(Preprocessing) NLTK stopwords loaded.")
except LookupError:
    print("(Preprocessing) NLTK stopwords not found locally.")
    if not download_nltk_resource_no_ssl('stopwords', 'Stopwords Corpus'): nltk_resources_available = False
    else:
         try: stop_words = set(stopwords.words('english'))
         except Exception as e: print(f"(Preprocessing) Failed to load stopwords after download: {e}"); nltk_resources_available = False

if nltk_resources_available:
    try:
        lemmatizer = WordNetLemmatizer()
        _ = lemmatizer.lemmatize('test') # Verify
        print("(Preprocessing) NLTK WordNet loaded.")
    except LookupError:
        print("(Preprocessing) NLTK WordNet not found locally.")
        if not download_nltk_resource_no_ssl('wordnet', 'WordNet Corpus'): nltk_resources_available = False
        else:
            try: lemmatizer = WordNetLemmatizer(); _ = lemmatizer.lemmatize('test')
            except Exception as e: print(f"(Preprocessing) Failed to load WordNet after download: {e}"); nltk_resources_available = False
    except Exception as e: print(f"(Preprocessing) Error initializing lemmatizer: {e}"); nltk_resources_available = False
else: print("(Preprocessing) Skipping WordNet load due to previous error.")


# --- Core Cleaning Function ---
def clean_description(text):
    """
    Applies the same cleaning steps used during training to a single text string.
    """
    if not isinstance(text, str): # Handle potential non-string input
        return ""

    # 1. Basic Cleaning (mirroring cleaning.py steps before advanced processing)
    text = text.replace('"', '') # Remove quotes
    text = text.lower()           # Convert to lowercase
    # Note: Splitting '===' is specific to the bulk CSV format, likely not needed for user input.
    text = text.strip()           # Remove leading/trailing whitespace

    # 2. Regex Cleaning (remove punctuation, numbers, specific strings)
    #    Important: Mirror the final regex used in cleaning.py. Assuming we remove all punctuation now.
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    # Remove specific known strings (mirroring cleaning.py)
    text = text.replace("cutapplicationid", '')
    text = text.replace("cpuc compliance", '')
    # Normalize whitespace again after removals
    text = re.sub(r'\s+', ' ', text).strip()

    # 3. NLTK Processing (Stopwords and Lemmatization)
    if nltk_resources_available and stop_words and lemmatizer:
        try:
            words = text.split()
            # Lemmatize words that are not stopwords
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
            text = ' '.join(lemmatized_words)
        except Exception as e:
             print(f"(Preprocessing) Error during NLTK processing for text '{text[:50]}...': {e}")
             # Optionally return original text or empty string on error
             # return text # Or return ""
             pass # Keep text as it was before NLTK step on error
    elif not nltk_resources_available:
         print("(Preprocessing) Warning: Skipping NLTK steps as resources are unavailable.")


    # 4. Final whitespace strip
    text = text.strip()

    return text

# Example Usage (for testing the function directly)
if __name__ == "__main__":
    test_desc = '   "Install New Electrical Panel === And circuit Breakers Ref# cutapplicationid123!!!"   '
    cleaned = clean_description(test_desc)
    print(f"Original: '{test_desc}'")
    print(f"Cleaned:  '{cleaned}'")

    test_desc_2 = "Convert garage to ADU adding plumbing"
    cleaned_2 = clean_description(test_desc_2)
    print(f"Original: '{test_desc_2}'")
    print(f"Cleaned:  '{cleaned_2}'") 