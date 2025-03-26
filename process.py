'''
Process data.
Includes basic tokenization and vectorization.
'''

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import time

DATAPATH = 'combined_emotion.csv'
OUTPATH = 'processed_data.npz'

def preprocess(text : str):
    '''
        Preprocess the text to prepare it for tokenization.
        Converts text to lower case. Removes puncuation.
    '''
    text = text.lower()                     # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)     # Remove punctuation
    return text 

# Initialize TFIDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=3000, stop_words='english', max_df=0.95, min_df=5
)

choice = input("Run tokenization + vectorization (y/n): ")
if choice == 'y':
    # Process data
    data = pd.read_csv(DATAPATH)
   
    # Apply preproccessing
    t1 = time.time()
    data['sentence'] = data['sentence'].apply(preprocess)
    t2 = time.time()
    print(f'Pre processing takes {(t2 - t1):.2f} seconds')

    # Do TF-IDF vectorization
    t1 = time.time()
    tfidfData = vectorizer.fit_transform(data['sentence'])
    t2 = time.time()
    print(f'Vecotrization takes {(t2 - t1):.2f} seconds')

    # Save processed data as a sparse matrix
    t1 = time.time()
    save_npz(OUTPATH, tfidfData)
    t2 = time.time()
    print(f'Saving takes {(t2 - t1):.2f} seconds')
    print('Saved processed data as a sparse matrix.')