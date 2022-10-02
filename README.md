# Projet NLP - Clustering et LSA par l'étude d'un dataset de The North Face - Construction d'un système de recommandation d'articles connexes

<p align='center'><img src='https://www.experience-outdoor.com/wp-content/uploads/2019/08/The-North-Face-marque-de-v%C3%AAtement-et-%C3%A9quipement-outdoor.jpg' width='200px'></p>

<a href='https://share.vidyard.com/watch/AuW29CHLDU62brd7RXnAJG?'>Cliquez ici pour visionner la vidéo de présentation du sujet</a>

## 1. Data Overview et objectifs
Les données fournies dans le dataset `sample-data.csv` sont des descriptifs produits. Chaque ligne contenant le nom du produit et une description de ce dernier. 
Pour utiliser les techniques de clustering et de LSA, on utilise principalement le lemma d'un document, ce qui appelle à un nettoyage, ou *pré-processing*, des textes afin qu'ils soient compris et utilisés efficacement par les algorithmes.

L'étude s'articule ainsi autour de trois axes :
<ul>
  <li>Le pré-processing des données, avec les librairies <code>spacy</code> et <code>re</code></li>
  <li>L'utilisation de <code>DBSCAN</code>, qui permet de faire du clustering sur les données, afin de rassembler les produits avec une description similaire</li>
  <li>Méthode LSA via <code>TruncatedSVD</code> de <code>scikit-learn</code>, ou <i>topic-modeling</i>, qui créé des topics pertinents en rapport avec la description des produits.
    Pour chaque produit, une note par topic est associée, plus celle-ci se rapproche de 1, mieux le topic explique le produit.</li>
</ul>

## 2. Aperçu
Après un fine-tuning des hyper-paramètres du `DBSCAN` pour trouver les paramètres optimaux du clustering, faisant la balance entre le nombre de clusters et le nombre d'outliers, l'analyse des mots représentatifs du cluster est possible par un `WordCloud`, dont voici un exemple pour le cluster n°6 :
<p align='center'>
  <img src="https://i.ibb.co/jfvKJDt/1.png" alt="1" border="0">
  <br/>
  <img src="https://i.ibb.co/PrxCYmj/2.png" alt="2" border="0">
  <h5 align='center'>Sample de produits représentatifs du Cluster n°6</h5>
</p>

Le cluster 6, par exemple, regroupe majoritairement des produits à base de laine ou des articles de montagne, ainsi si un utilisateur cherche un produit chaud pour la montagne, l'algorithme pourra proposer différents articles connexes à sa recherche.

Le reste de l'étude est détaillé dans <a href='https://github.com/Chedeta/the_north_face_nlp/blob/main/NorthFace_final.ipynb'>le notebook</a> mis à disposition.

## 3. Crédits
Le projet a été effectué en collaboration avec <a href='https://github.com/Bebock'>Hélène</a>, <a href=''>Henri</a> et <a href='https://github.com/NBridelance'>Nicolas</a>.
