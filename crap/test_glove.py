import matplotlib.pyplot as plt
import seaborn as sns
import re
import pdb
import numpy as np
import pandas as pd
import spacy
import string
nlp = spacy.load("en_core_web_lg")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn import model_selection, metrics, svm
SEED=40


full_train_data = pd.read_csv('C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/test_data/train.csv')
full_submission_data = pd.read_csv('C:/Users/alexa/OneDrive/Desktop/Thesis/Bitbucket/thesisforex2/test_data/test.csv')
train_data_shape = full_train_data.shape[0]
df = pd.concat([full_train_data, full_submission_data])

def clean_text(text):
    # remove_URL
    url = re.compile(r'https?://\S+|www\.\S+')
    text =  url.sub(r'', text)

    # remove_html
    html = re.compile(r'<.*?>')
    text = html.sub(r'', text)

    # remove_emoji
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags = re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # remove_punct
    table = str.maketrans('', '', string.punctuation)
    text = text.translate(table)
    
    return text


df['text'] = df['text'].apply(lambda x : clean_text(x))
train_data = df.copy().iloc[:train_data_shape]
submission_data = df.copy().iloc[train_data_shape:]

X = train_data.copy().drop(['target'], axis=1)
y = train_data.copy()['target']
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.15, random_state=SEED)
X_sub = submission_data.copy().drop(['target'], axis=1)

class SpacyVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp):
        self.nlp = nlp
        self.dim = 300

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([self.nlp(text).vector for text in X])




column_preprocessor = ColumnTransformer(
    [
        ('text_glove', SpacyVectorTransformer(nlp), 'text'),
    ],
    remainder='drop',
    n_jobs=1
)

pipeline = Pipeline([
    ('column_preprocessor', column_preprocessor),
    ('svm', svm.SVC(kernel='rbf', C=1.2, gamma=0.2))
])
pdb.set_trace()
print("start pipeline fit")
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
print(metrics.accuracy_score(y_test, predictions))