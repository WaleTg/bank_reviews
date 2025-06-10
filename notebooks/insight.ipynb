import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd

# Load your dataset
file_path = r"C:\Users\HP\Desktop\Tenx\week2\data\clean\ethiopian_bank_reviews_cleaned.csv"
df = pd.read_csv(file_path)

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Apply sentiment analysis on the correct column: 'content'
df["sentiment_score"] = df["content"].apply(lambda t: sia.polarity_scores(str(t))["compound"])
def get_sentiment_label(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

df["sentiment_label"] = df["sentiment_score"].apply(get_sentiment_label)

import os
import seaborn as sns
import matplotlib.pyplot as plt

# Make sure output directory exists
os.makedirs("outputs", exist_ok=True)

# Plot and save
sns.countplot(data=df, x="sentiment_label", hue="bank")
plt.title("Sentiment Distribution by Bank")
plt.savefig("outputs/sentiment_dist.png")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from pathlib import Path

# Setup
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
try:
    df = pd.read_csv('data/processed/ethiopian_reviews_analyzed.csv')
except FileNotFoundError:
    print("Error: ethiopian_reviews_analyzed.csv not found")
    exit(1)

# Plot 1: Sentiment Distribution by Bank (Bar Chart)
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='sentiment_label', hue='bank')
plt.title('Sentiment Distribution by Bank')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.legend(title='Bank')
plt.savefig(output_dir / 'sentiment_distribution.png')
plt.close()

# Plot 2: Keyword Cloud for BOA
boa_reviews = ' '.join(df[df['bank'] == 'Bank of Abyssinia']['keywords'].apply(lambda x: ' '.join(x.split(','))))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(boa_reviews)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Keyword Cloud for Bank of Abyssinia')
plt.savefig(output_dir / 'keyword_cloud_boa.png')
plt.close()

# Plot 3: Rating Distribution by Bank
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='rating', hue='bank')
plt.title('Rating Distribution by Bank')
plt.xlabel('Rating (Stars)')
plt.ylabel('Number of Reviews')
plt.legend(title='Bank')
plt.savefig(output_dir / 'rating_distribution.png')
plt.close()

# Plot 4: Theme Distribution by Bank
theme_counts = df.assign(themes=df['themes'].str.split(',')).explode('themes')
theme_counts = theme_counts.groupby(['bank', 'themes']).size().unstack(fill_value=0)
theme_counts.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Theme Distribution by Bank')
plt.xlabel('Bank')
plt.ylabel('Number of Reviews')
plt.legend(title='Theme', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(output_dir / 'theme_distribution.png')
plt.close()

# Plot 5: Slow Transfer Complaints by Bank
transfer_issues = df[df['themes'].str.contains('Transaction Performance') & df['content'].str.contains('slow|load|transfer', case=False)]
transfer_counts = transfer_issues.groupby('bank').size()
plt.figure(figsize=(8, 6))
transfer_counts.plot(kind='bar')
plt.title('Slow Transfer Complaints by Bank')
plt.xlabel('Bank')
plt.ylabel('Number of Reviews')
plt.savefig(output_dir / 'slow_transfers.png')
plt.close()

print("Visualizations saved to outputs/")
