import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

dataset = pd.read_csv("Language Detection.csv")
#print(dataset.head)
print(dataset.shape)
print(dataset.columns)
#print(dataset.info())
#print(dataset.isnull().sum())
#print(dataset.describe())

# Word Frequency Histogram
# langCount = dataset["Language"].value_counts()
# plt.figure(figsize=(12,6))
# sns.barplot(x=langCount.index, y=langCount.values, palette="viridis")
# plt.xlabel("Language", fontsize=12)
# plt.ylabel("Number of Samples", fontsize=12)
# plt.title("Distribution of Samples per Langauge", fontsize=14)
# plt.xticks(rotation=45)
# plt.show()

# Most common words per language
# language = "Turkish"
# textData = dataset[dataset["Language"] == language]["Text"].str.cat(sep=" ")
# words = textData.split()
# wordCount = Counter(words)
# commWords = pd.DataFrame(wordCount.most_common(15), columns=["Word","Frequency"])

# plt.figure(figsize=(12,8))
# sns.barplot(x="Frequency", y="Word", data=commWords, palette="coolwarm")
# plt.title(f"Top 15 most common words in {language}")
# plt.show()