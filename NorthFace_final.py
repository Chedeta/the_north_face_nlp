#!/usr/bin/env python
# coding: utf-8

# # Projet NLP - Clustering et LSA par l'étude d'un dataset de The North Face

!pip install spacy
import spacy
spacy.__version__
python -m spacy download en_core_web_sm

!pip install wordcloud
import wordcloud

import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from spacy.lang.en.stop_words import STOP_WORDS

import matplotlib.pyplot as plt
import seaborn as sns

import en_core_web_sm
nlp = en_core_web_sm.load()

df = pd.read_csv('sample-data.csv')
df.shape
df.head()


# # Partie 1 : Texte pre-processing
# Nettoyage du des balises HTML et des liens hypertextes + création d'une colonne description nettoyée ==> description_nohtml

import re
clean_tags = re.compile('<.*?>') 

def cleanhtml(raw_html):
    remhtml = re.sub(clean_tags, ' ', raw_html)
    remhtml = re.sub(r'http\S+', ' ', remhtml)
    return remhtml

df['description_nohtml']=''

for i in range(len(df)) :
    df['description_nohtml'][i] = cleanhtml(df.description[i])
df.head()


# Fonction qui permet de nettoyer le document

def cleaning(df):
    df =df.str.replace(r"\S*@\S*\s?",' ') #supprime adresse mail
    df = df.str.replace('[^\w\s]',' ') #garde les alphanumérique
    df = df.str.replace("#", " ") #hastags en texte
    df = df.str.replace('\d+', '') #enlève les chiffres
    df = df.str.replace(" +", " ") #remplace + par espace
    df = df.fillna('').apply(lambda x: x.lower())
    df = df.str.split(' ').map(lambda x: " ".join(s for s in x if len(s) > 2))
    return (df)
df['cleaned_nohtml'] = cleaning(df["description_nohtml"])


# Création des tokens et lemmas à l'aide de spacy

tokens = []
lemma = []
for doc in nlp.pipe(df['cleaned_nohtml'].astype('unicode').values):
    if doc.has_annotation:
        tokens.append([n.text for n in doc if n.text not in STOP_WORDS])
        lemma.append([n.lemma_ for n in doc if n.lemma_ not in STOP_WORDS])
    else: #si pas le même nombre d'entrées dans df, on ajoute un élément vide
        tokens.append(None)
        lemma.append(None)

df['desc_tokens'] = tokens
df['desc_lemma'] = lemma
df.head()


# Fonction qui lemmatize le texte avant vectorisation

def lemmatize_text(text):
      text = ''.join(ch for ch in text if ch.isalnum() or ch==" ")
      text = text.replace(" +"," ").lower().strip()
      return text
    
df['clean_lemma'] = df["desc_lemma"].apply(lambda x: lemmatize_text(str(x)))
df.head()


# Création du TF-IDF, cette méthode permet d'évaluer la pertinence d'un mot dans le document (dans notre cas, dans le lemma). 
# 
# Ici, il s'agit de mesurer le nombre de fois qu'un mot apparaît dans un document, en podérant par la fréquence d'apparition de ce mot dans les autres documents. 

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['clean_lemma'])

dense = X.toarray()
dense


# Création du dataset vectorisé

data = pd.DataFrame(dense, 
             columns=[x for x in vectorizer.get_feature_names_out()], 
             index=["doc_{}".format(x) for x in range(0, len(df))] )
data


# # Partie 2 : Clustering des produits, système de recommandation
# Utilisation du DBSCAN, fine tuning des paramètres
 
liste1 = []
liste2 = []
liste3 = []
for i in range(7120,7300, 2):
    db = DBSCAN(eps=i/10000, min_samples=6, metric="cosine", algorithm="auto") 
    db.fit(data)
    df['db_cluster']=db.labels_
    liste1.append(len(df[df['db_cluster'] == -1]))
    liste2.append(df['db_cluster'].nunique())
    liste3.append(i/1000)
fig, ax1 = plt.subplots() 

ax1.set_xlabel('epsilon - dbscan') 
ax1.set_ylabel('Nb dans cluster -1') 
ax1.plot(liste3,liste1, color = 'red') 
ax2 = ax1.twinx() 
ax2.plot(liste3,liste2, color = 'blue')  
plt.ylabel('Nb clusters') 
plt.show()
fig.clear(True)
liste1 = []
liste2 = []
liste3 = []

db = DBSCAN(eps=0.7165, min_samples=6, metric="cosine", algorithm="auto")
db.fit(data)
df['db_cluster']=db.labels_
print('On trouve {} clusters. '.format(df['db_cluster'].nunique()))


# Répartition des documents dans les onze clusters

df['db_cluster'].value_counts().sort_values(ascending=False)
df['product'] = df['description'].apply(lambda x:x.split(' - ')[0])
df.head()

import matplotlib.pyplot as plt
import wordcloud
for i in range(0,10):
    text_merge_clean_token = ''
    df_temp = df[df['db_cluster'] == i]
    for j in range(0, len(df_temp)):
        text_merge_clean_token += str(df_temp['clean_lemma'].iloc[j])
    print('CLUSTER n°' + str(i))
    wd = wordcloud.WordCloud(collocations=False, max_words=30)
    cloud = wd.generate(nlp(text_merge_clean_token).text)
    fig = plt.figure(
    figsize = (8, 5),
    facecolor = 'k',
    edgecolor = 'k')
    plt.imshow(cloud )
    plt.show()
    print('\n')


# Fonction de recommandation de produits similaires

def find_similar_items(item_id):
    item_id = int(item_id)
    db_cluster_target = df[df['id'] == item_id]['db_cluster'].tolist()[0]
    df_filtre = df[df['db_cluster'] == db_cluster_target]
    result = df_filtre['product'].sample(5).to_list()
    return result
find_similar_items(input())


# # Partie 3 : Topic Modeling
# LSA : décomposition des documents et recherche des topics pertinents à l'aide de TruncatedSVD

liste = []
for i in range(10,21):
    svd = TruncatedSVD(n_components=i, algorithm='randomized', n_iter=50)
    lsa = svd.fit_transform(X)
    var_explained = svd.explained_variance_ratio_.sum()
    liste.append(var_explained)
print('La variance max est de : '+str(max(liste))+' pour n_components = ' +str(10 + liste.index(max(liste))))

svd_model = TruncatedSVD(n_components=12, algorithm='randomized', random_state=122)
lsa = svd_model.fit_transform(X)

topic_encoded_df = pd.DataFrame(lsa, columns = ["topic " + str(i) for i in range(0,12)])
topic_encoded_df["documents"] = df['clean_lemma']
topic_encoded_df

sns.scatterplot("topic 1","topic 2",data= topic_encoded_df, hue = db.labels_, palette=sns.color_palette("tab20")[0:11])


# Permet de trouver le topic principal par document (ou produit)

def main_topic(row):
    topics = np.abs(row)
    main_topic = topics.sort_values(ascending=False).index[0]
    return main_topic

topic_encoded_df.loc[:, 'main_topic'] = 0

for i, row in topic_encoded_df.iloc[:,:-2].iterrows():
    topic_encoded_df.loc[i, 'main_topic'] = main_topic(row)

topic_encoded_df.head()
topic_encoded_df['main_topic'].value_counts()

# Création de WordCloud par topic généré.

wd = wordcloud.WordCloud(collocations=False, max_words=30)
col = [c for c in topic_encoded_df.columns if 'topic ' in c]
inc = 0
for topic in col:
    print('Topic n°' + str(inc))
    texts = " ".join(topic_encoded_df.loc[topic_encoded_df['main_topic']==topic,'documents'])
    cloud = wd.generate(texts)
    plt.imshow(cloud)
    plt.show()
    inc += 1
