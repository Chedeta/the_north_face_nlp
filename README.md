# Projet NLP - Clustering et LSA par l'étude d'un dataset de The North Face - Construction d'un système de recommandation d'articles connexes

<p align='center'><img src='https://www.experience-outdoor.com/wp-content/uploads/2019/08/The-North-Face-marque-de-v%C3%AAtement-et-%C3%A9quipement-outdoor.jpg' width='200px'></p>

Cliquez ici pour visionner la vidéo de présentation du sujet

## 1. Data Overview et objectifs
Les données fournies dans le dataset `sample-data.csv` sont des descriptifs produits. Chaque ligne contenant le nom du produit et une description de ce dernier. 
Pour utiliser les techniques de clustering et de LSA, on utilise principalement le lemma d'un document, ce qui appelle à un nettoyage, ou *pré-processing*, des textes afin qu'ils soient compris et utilisés efficacement par les algorithmes.

L'étude se déroule ainsi autour de trois axes :
<ul>
  <li>Le pré-processing des données, avec les librairies <code>spacy</code> et <code>re</code></li>
  <li>L'utilisation de <code>DBSCAN</code>, qui permet de faire du clustering sur les données, afin de rassembler les produits avec une description similaire</li>
  <li>Méthode LSA via <b>TSVD</b> de <code>scikit-learn</code>, ou <i>topic-modeling</i>, qui créé des topics pertinents en rapport avec la description des produits, chaque produit, une note par topic est associée, plus celle-ci se rapproche de 1, mieux le topic explique le produit.</li>
</ul>
## 2. Objectifs

## 3. 
