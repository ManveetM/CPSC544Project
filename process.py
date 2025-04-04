'''
Process data.
Includes basic tokenization and vectorization.
'''

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
import time
import numpy as np
import nlpaug.augmenter.word as naw

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

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
    max_features=3000, stop_words='english', max_df=0.95, min_df=5, ngram_range=(1, 2)
)

# Initialize synonym augmenter
aug = naw.SynonymAug(aug_src='wordnet')

choice = input("Run tokenization + vectorization (y/n): ")
if choice == 'y':
    # Process data
    data = pd.read_csv(DATAPATH)

    # Data Augmentation for emotions that is less represented than others ('love', 'suprise')
    t1 = time.time()
    augmented_rows = []
    for i, row in data.iterrows():
        if row['emotion'] in ['love', 'suprise']:
            aug_sentence = aug.augment(row['sentence'])
            if isinstance(aug_sentence, list):
                aug_sentence = aug_sentence[0]
            augmented_rows.append({'sentence': aug_sentence, 'emotion': row['emotion']})

    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        data = pd.concat([data, aug_df], ignore_index=True)
        print(f"Data augmented: {len(aug_df)} new samples added.")
    t2 = time.time()
    print(f'Data augmentation process took {(t2 - t1):.2f} seconds')
   
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

    # Save augmented labels as an .npy file
    labels = data['emotion'].values
    np.save('labels.npy', labels)
    print('Saved augmented label as an .npy file.')

    # Save an augmented .csv file to be used in visualize.py
    data.to_csv("augmented_emotion.csv", index=False)
    print("Saved augmented dataset to augmented_emotion.csv")