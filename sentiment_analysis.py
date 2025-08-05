import pandas as pd
import numpy as np
import re
import transformers
from transformers import pipeline
import matplotlib.pyplot as plt

data = pd.read_csv('book_reviews_sample.csv')
transformer_pipeline = pipeline("sentiment-analysis")

# cleaning the data
data['review_clean'] = data['reviewText'].str.lower()
data['review_clean'] = data['review_clean'].apply(lambda x: re.sub(r"([^\w\s])", "", x))

# labeling data

transformer_label = []

for review in data['review_clean'].values:
    sentiment_list = transformer_pipeline(review)
    transformer_label.append([sent['label'] for sent in sentiment_list])

data['sentiment_label'] = transformer_label

#data displaying
data['sentiment_label'].value_counts().plot.bar()
plt.tight_layout()
plt.show()

# print(transformer_label[:10])

