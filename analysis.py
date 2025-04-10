import pandas as pd

df = pd.read_csv('permit_data_cleaned.csv')
# After cleaning, before saving:
all_words = df['DESCRIPTION'].str.split(expand=True).stack()
print("\nMost frequent words after NLTK cleaning:")
short_words = all_words[all_words.astype(str).str.len() <= 2]
print(short_words.value_counts().head(50))

print("\nPERMITTYPE value counts:")
print(df['PERMITTYPE'].value_counts())

