import pandas as pd
import re

# Define chunk size (adjust based on your memory capacity)
chunk_size = 50000  # Process 50,000 rows at a time
input_file = "permit_data.csv"
output_file = "permit_data_cleaned.csv"

# --- Define the cleaning function ---
def clean_chunk(df_chunk):
    """Applies cleaning steps to a DataFrame chunk."""
    # Keep only necessary columns
    df_chunk = df_chunk[["PMPERMITID", "PERMITTYPE", "DESCRIPTION"]].copy()

    # --- Basic Cleaning ---
    df_chunk['DESCRIPTION'] = df_chunk['DESCRIPTION'].fillna('')
    df_chunk['DESCRIPTION'] = df_chunk['DESCRIPTION'].str.replace('"', '')
    df_chunk['DESCRIPTION'] = df_chunk['DESCRIPTION'].str.lower()
    df_chunk['DESCRIPTION'] = df_chunk['DESCRIPTION'].str.split('===', n=1, expand=True)[0]
    df_chunk['DESCRIPTION'] = df_chunk['DESCRIPTION'].str.strip()

    # --- Further Cleaning for Classification ---
    df_chunk['DESCRIPTION'] = df_chunk['DESCRIPTION'].str.replace(r'[^\w\s]', '', regex=True)
    df_chunk['DESCRIPTION'] = df_chunk['DESCRIPTION'].str.replace(r'\d+', '', regex=True)
    df_chunk['DESCRIPTION'] = df_chunk['DESCRIPTION'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # --- Advanced Cleaning (Requires NLTK/spaCy) ---
    # def clean_text_advanced(text):
    #     words = text.split()
    #     words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    #     return ' '.join(words)
    # df_chunk['DESCRIPTION'] = df_chunk['DESCRIPTION'].apply(clean_text_advanced)

    # Ensure PMPERMITID is string
    df_chunk['PMPERMITID'] = df_chunk['PMPERMITID'].astype(str)

    return df_chunk

# --- Process the file in chunks ---
print(f"Starting processing of {input_file} in chunks of {chunk_size}...")

first_chunk = True
# Create a TextFileReader object
reader = pd.read_csv(input_file, chunksize=chunk_size, iterator=True)

for i, chunk in enumerate(reader):
    print(f"Processing chunk {i+1}...")
    cleaned_chunk = clean_chunk(chunk)

    if first_chunk:
        # For the first chunk, write with header
        cleaned_chunk.to_csv(output_file, index=False, mode='w', header=True)
        first_chunk = False
        print("Columns in cleaned data:", cleaned_chunk.columns.tolist()) # Print columns once
    else:
        # For subsequent chunks, append without header
        cleaned_chunk.to_csv(output_file, index=False, mode='a', header=False)

print(f"Finished processing. Cleaned data saved to {output_file}")