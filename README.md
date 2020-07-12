# Projet Transverse

Crée en juin 2020 par Rémy Voillet dans le cadre d'un cours sur le Data Mining.

## Présentation du projet

Ce petit outil en python permet d'analyser les performances/comportements de deux modèles différents de machine learning de machine learning qui sont :
* le SVC (Support Vector Classification - [Doc ici](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html))
* et le random Forest ([Doc ici](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html))

## Utilisation du projet

Pour lancer l'outil il faut avoir installer les librairies python :
* dash
* dash_core_components
* dash_html_components
* dash_bootstrap_components
* dash_table
* pandas
* numpy
* itertools
* sklearn
* io
* seaborn
* matplotlib
* plotly
* sys
* colorlover

Il suffit ensuite de récupérer le git et de lancer le fichier **app.py** via python

## Fonctionnalités

L'outil se divise en trois partie divisés en trois onglets : 

**Onglet 1 : Analyse descriptive du dataset**
Affiche certaines informations sur le dataset sélectionné (Student-performance [disponible ici](http://archive.ics.uci.edu/ml/datasets/Student+Performance))
Tel que le shape du dataset, des informations sur les colonnes, les moyennes, max, min et percentiles des données numériques, schémas, etc

**A noter :** Le dataset comportait 3 colonnes résultats, lamoyenne de maths de l'élève au 1er timester, aux deuxième et au troisième. Ces trois colonnes ont été régroupés pour avoir une variable de résultat booléene.
Cette valeur de résultat représente si l'élève à son année en maths, donc si la moyenne des 3 moyennes est supérieure à 10.

Cet onglet affiche aussi le dataset après feature engeneering (colonnes non numériques enlevés, colonnes ayant qu'une seule valeur enlevé et colonnes booléens transformé en numérique)

**Onglet 2 : Analyse du comportement et des performances du modèle**
Cet onglet est divisé en quatres colonnes. 

Les deux premières permettent de régler des variables du modèle SVC ainsi que de voir les performances de ce modèle via une courbe ROC et une matrice de confusion (représenté en camembert)

Les deux dernières permettent de régler des variables du modèle Random Forest ainsi que de voir les performances de ce modèle via une courbe ROC et une matrice de confusion (représenté en camembert)

**Onglet 3 : Dataset (attention, peut être très long)**
Ce dernier onglet affiche jute le dataset source entier sans aucune mdification.

## Le code

Le python est divisé en 4 parties :

**Générations des données d'affichage et modification du dataset**
Partie qui génère les informations affichés dans le premier onglet et qui modifie le dataset (ajoute la variable target et feature engeneering) 

**Création de l'architecture du DASH**
Partie qui génère tout les composants de l'outils (onglets, tableaux, containers pour les diagrammes,...)

**Mise en place de modification dynamique pour le modèle SVC**
Partie qui crée le modèle SVC en fonction des champs et qui affiche les diagrammes de performance

**Mise en place de modification dynamique pour le modèle RandomForest**
Partie qui crée le modèle RandomForest en fonction des champs et qui affiche les diagrammes de performance


#Améliorations prévues

- Améliorer grandement le design
- Permettre de choisir le modèle dans chaque colonne (pour en proposer plus)
