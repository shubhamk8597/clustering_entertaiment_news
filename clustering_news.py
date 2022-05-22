import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import seaborn as sns

df_entertainment_news = pd.read_csv('data/combined_entertaiment_dataset.csv')

print(df_entertainment_news[df_entertainment_news['text'].duplicated(keep=False)].sort_values('text').head(8))


df_entertainment_news1 = df_entertainment_news.drop_duplicates('text')

df_entertainment_news = df_entertainment_news1.sample(frac = 0.5)
print(df_entertainment_news.head())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


import nltk
from nltk.corpus import stopwords


def preprocess_text(text: str, remove_stopwords: bool) -> str:
    """This utility function sanitizes a string by:
    - removing links
    - removing special characters
    - removing numbers
    - removing stopwords
    - transforming in lowercase
    - removing excessive whitespaces
    Args:
        text (str): the input text you want to clean
        remove_stopwords (bool): whether or not to remove stopwords
    Returns:
        str: the cleaned text
    """

    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove special chars and numbers
    text = re.sub("[^A-Za-z]+", " ", text)
    # remove stopwords
    if remove_stopwords:
        # 1. tokenize
        tokens = nltk.word_tokenize(text)
        # 2. check if stopword
        tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        # 3. join back together
        text = " ".join(tokens)
    # return text in lower case and stripped of whitespaces
    text = text.lower().strip()
    return text

df_entertainment_news['text']=df_entertainment_news['text'].apply(str)
df_entertainment_news['cleaned'] = df_entertainment_news['text'].apply(lambda x: preprocess_text(x, remove_stopwords=True))

# initialize the vectorizer

vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
# fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
X = vectorizer.fit_transform(df_entertainment_news['cleaned'])


from sklearn.cluster import KMeans

K = range(1, 15)
SSE = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    SSE.append(kmeans.inertia_)

import matplotlib.pyplot as plt

plt.plot(K, SSE, 'bx-')
plt.title('Elbow Method')
plt.xlabel('cluster numbers')
plt.show()

# initialize kmeans with 3 centroids
kmeans = KMeans(n_clusters=5, random_state=42)
# fit the model
kmeans.fit(X)
# store cluster labels in a variable
clusters = kmeans.labels_



from sklearn.decomposition import PCA

# initialize PCA with 2 components
pca = PCA(n_components=2, random_state=42)
# pass our X to the pca and store the reduced vectors into pca_vecs
pca_vecs = pca.fit_transform(X.toarray())
# save our two dimensions into x0 and x1
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]

df_entertainment_news['cluster'] = clusters
df_entertainment_news['x0'] = x0
df_entertainment_news['x1'] = x1


def get_top_keywords(n_terms):
    """This function returns the keywords for each centroid of the KMeans"""
    df = pd.DataFrame(X.todense()).groupby(clusters).mean()  # groups the TF-IDF vector by cluster
    terms = vectorizer.get_feature_names_out()  # access tf-idf terms
    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)[
                                          -n_terms:]]))  # for each row of the dataframe, find the n terms that have the highest tf idf score


print(get_top_keywords(30))

cluster_map = {0: "0", 1: "1", 2: "2", 3: "3",4: "4"}
# apply mapping
df_entertainment_news['cluster'] = df_entertainment_news['cluster'].map(cluster_map)

# set image size
plt.figure(figsize=(12, 7))
# set a title
plt.title("TF-IDF + KMeans clustering", fontdict={"fontsize": 18})
# set axes names
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
# create scatter plot with seaborn, where hue is the class used to group the data
sns.scatterplot(data=df_entertainment_news, x='x0', y='x1', hue='cluster', palette="viridis")
plt.show()

text = '''
Maye Musk, Elon Musk’s 74-year-old mom, stunned on the cover of Sports Illustrated Swimsuit
Maye Musk, the mother of billionaire Tesla owner Elon Musk, is quite the celebrity herself. The 74-year-old has a long career in modeling and this year she is making history as one of selected women to be featured in the cover of Sports Illustrated Swimsuit. Like Kim...'''
Y = vectorizer.transform([text])
prediction = kmeans.predict(Y)
print(prediction)

