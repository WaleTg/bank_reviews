import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the cleaned data
file_path = r"C:\Users\HP\Desktop\Tenx\week2\data\clean\ethiopian_bank_reviews_cleaned.csv"
df = pd.read_csv(file_path)

# Standardize column names (if needed)
df.columns = df.columns.str.strip().str.lower()

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Apply sentiment scoring on the 'content' column
df["sentiment_score"] = df["content"].apply(lambda t: sia.polarity_scores(str(t))["compound"])



# Save or preview results
print(df[["bank", "content", "sentiment_score"]].head())
import pandas as pd, spacy, pathlib, collections
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")
file_path=r"C:\Users\HP\Desktop\Tenx\week2\data\clean\ethiopian_bank_reviews_cleaned.csv"
df  = pd.read_csv(file_path)

def top_terms(bank, k=40):
    docs = df.loc[df.bank==bank, "content"].tolist()
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=2000)
    X   = vec.fit_transform(docs)
    idx = X.sum(axis=0).A1.argsort()[-k:][::-1]
    return [vec.get_feature_names_out()[i] for i in idx]

themes = {}
for bank in df.bank.unique():
    terms = top_terms(bank)
    # quick rule-based grouping demo
    group = {"login":["login","log in","sign in","otp"],
             "speed":["slow","loading","speed","lag"],
             "bugs":["crash","error","bug","freeze"],
             "ui":  ["interface","ui","design"]}
    bank_themes = collections.defaultdict(list)
    for t in terms:
        for k,keys in group.items():
            if any(k2 in t for k2 in keys):
                bank_themes[k].append(t)
    themes[bank] = dict(bank_themes)

pathlib.Path("outputs").mkdir(exist_ok=True)
pd.Series(themes).to_json("outputs/themes.json")
