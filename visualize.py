import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK resources are downloaded
#nltk.download('stopwords')
#nltk.download('punkt')

# Load dataset
df = pd.read_csv("combined_emotion.csv")

# Basic data analysis
# print(dataset.head)
#print(df.shape)
# print(dataset.columns)
#print(df.info())
# print(dataset.isnull().sum())
#print(dataset.describe())

# Check for emppty or whitespace only strings
# print((dataset['sentence'].str.strip() == '').sum())
# print((dataset['emotion'].str.strip() == '').sum())

# Samples per emotion
# emotionCount = df["emotion"].value_counts()
# plt.figure(figsize=(12,6))
# sns.barplot(x=emotionCount.index, y=emotionCount.values, palette="viridis")
# plt.xlabel("emotion", fontsize=12)
# plt.ylabel("Number of Samples", fontsize=12)
# plt.title("Distribution of Samples per Langauge", fontsize=14)
# plt.xticks(rotation=45)
# plt.show()

exit(0)

# Preprocess and tokenize
# Preprocessing function
def preprocess(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = word_tokenize(text)  # Tokenize words
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return words

# Apply preprocessing
df['processed'] = df['sentence'].apply(preprocess)

# Count word frequencies per emotion
emotion_word_counts = {}
for emotion, group in df.groupby('emotion'):
    words = [word for sentence in group['processed'] for word in sentence]  # Flatten word list
    emotion_word_counts[emotion] = Counter(words).most_common(10)  # Get top 10 words

# Used to generate one plot of common words for all emotion
# Convert to DataFrame for visualization
# plot_data = []
# for emotion, words in emotion_word_counts.items():
#     for word, count in words:
#         plot_data.append((emotion, word, count))

# plot_df = pd.DataFrame(plot_data, columns=['emotion', 'word', 'count'])

# # Plot using seaborn
# plt.figure(figsize=(12, 6))
# sns.barplot(data=plot_df, x='word', y='count', hue='emotion')
# plt.xticks(rotation=45)
# plt.title('Most Common Words Per Emotion')
# plt.show()

# Used to generate plots of common words for each emotion 
# Plot using subplots for each emotion
fig, axes = plt.subplots(len(emotion_word_counts), 1, figsize=(10, 12))

# Ensure axes is iterable even if there is only one emotion
if len(emotion_word_counts) == 1:
    axes = [axes]

for ax, (emotion, words) in zip(axes, emotion_word_counts.items()):
    sns.barplot(y=[word[0] for word in words], x=[word[1] for word in words], ax=ax)
    ax.set_title(f"Most Common Words in {emotion.capitalize()} Sentences")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Word")

plt.tight_layout()
plt.show()