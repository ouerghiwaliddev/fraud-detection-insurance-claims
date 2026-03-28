# Insurance Fraud Clustering - Segmentation Non Supervisee des Reclamations

## Description du projet

Ce projet applique des methodes de **clustering (apprentissage non supervise)** sur un jeu de donnees de reclamations d'assurance automobile. L'objectif n'est pas de predire directement la fraude, mais de **segmenter le portefeuille de reclamations en profils types** et de verifier si certains segments concentrent un taux de fraude anormalement eleve.

### Pourquoi le clustering en assurance ?

En pratique, un assureur ne sait pas a l'avance quelles reclamations sont frauduleuses. Le clustering permet de :

- **Segmenter le portefeuille** en groupes de reclamations aux caracteristiques similaires
- **Identifier les segments a risque** : un cluster avec un taux de fraude de 40% vs 25% en moyenne = priorite d'audit
- **Cibler les investigations** : concentrer les ressources humaines sur les profils les plus suspects
- **Detecter les comportements atypiques** : les outliers (points de bruit dans DBSCAN) peuvent correspondre a des reclamations inhabituelles meritant une verification

Cette approche **complete** un modele supervise de detection de fraude (voir le projet `Insurance Fraud Classification`) en apportant une vision exploratoire sans a priori.

## Structure du projet

```
Insurance Fraud Clustering/
├── README.md
├── insurance_claims.csv            # Dataset source (1 000 reclamations)
└── clustering_analysis.ipynb       # Notebook d'analyse complet
```

## Dataset

**Fichier :** `insurance_claims.csv` (identique au projet de classification)

- **1 000 reclamations** d'assurance automobile
- **39 variables** decrivant l'assure, la police, l'incident et la reclamation
- La variable `fraud_reported` (Y/N) est **supprimee avant le clustering** mais conservee pour la comparaison finale

Pour le detail de chaque variable, consulter le README du projet `Insurance Fraud Classification`.

## Notebook d'analyse (`clustering_analysis.ipynb`)

### 1. Preprocessing

- Suppression de la variable cible `fraud_reported` (sauvegardee a part pour evaluation finale)
- Suppression des colonnes non exploitables (identifiants, dates, adresses)
- Remplacement des valeurs manquantes (`?`) par le mode de chaque colonne
- LabelEncoder sur `insured_sex`, One-Hot Encoding sur les autres variables categoriques
- Normalisation avec StandardScaler

### 2. Reduction de dimensionnalite (visualisation 2D)

Trois methodes appliquees pour projeter les donnees en 2D et visualiser la structure :

| Methode | Principe | Interet |
|---------|----------|---------|
| **PCA** | Projection lineaire sur les axes de variance maximale | Rapide, interpretable (% de variance expliquee), scree plot |
| **t-SNE** | Preservation des distances locales (voisinages) | Revele les groupes denses, bonne pour les structures non lineaires |
| **UMAP** | Preservation de la topologie locale et globale | Plus rapide que t-SNE, conserve mieux la structure globale |

Chaque projection est coloree par la fraude reelle (reference) puis par les clusters de chaque methode.

### 3. KMeans

**Selection du nombre optimal de clusters (K)** avec 4 metriques :

| Metrique | Principe | Critere |
|----------|----------|---------|
| **Elbow (Inertie)** | Somme des distances intra-cluster au carre | Chercher le "coude" (KneeLocator) |
| **Score de Silhouette** | Mesure la coherence des clusters : chaque point est-il plus proche de son cluster que des autres ? | Plus haut = meilleur (max = 1) |
| **Davies-Bouldin** | Ratio entre la dispersion intra-cluster et la separation inter-cluster | Plus bas = meilleur (min = 0) |
| **Calinski-Harabasz** | Ratio entre la variance inter-cluster et intra-cluster | Plus haut = meilleur |

Inclut un **Silhouette Plot** detaille montrant la contribution de chaque point.

### 4. Profiling des clusters

Analyse des **centroides** (point moyen de chaque cluster) pour comprendre le profil type de chaque segment :

- Tableau des valeurs moyennes en echelle originale
- **Heatmap** des centroides standardises (rouge = valeur elevee, bleu = faible)
- **Radar chart** (spider plot) comparant visuellement les profils

Exemple d'interpretation : un cluster avec des valeurs elevees sur `total_claim_amount`, `vehicle_claim`, faibles sur `witnesses` et `police_report_available` = profil potentiellement suspect.

### 5. Decision Tree sur les clusters

L'index des clusters est utilise comme **variable cible** d'un arbre de decision pour extraire les **regles interpretables** qui separent les segments.

- **Tree plot** : visualisation de l'arbre avec les conditions de decision
- **Feature importance** : quelles variables distinguent le plus les clusters

Interet metier : traduire les clusters en criteres concrets pour les analystes fraude (ex: "si montant > 50 000$ ET pas de rapport de police ET vehicule > 15 ans -> Cluster 2").

### 6. Clustering Hierarchique

- **Dendrogramme** : visualisation de la hierarchie de fusion des groupes (methode Ward)
- **AgglomerativeClustering** : application avec le meme K que KMeans pour comparaison
- Metriques et visualisation sur les 3 reductions

### 7. GMM (Gaussian Mixture Models)

- Selection du K via **BIC** (Bayesian Information Criterion) et **AIC** (Akaike Information Criterion)
- Contrairement a KMeans (clusters spheriques), le GMM modelise des clusters **elliptiques** avec des covariances differentes
- Chaque point a une **probabilite d'appartenance** a chaque cluster (soft clustering)

### 8. DBSCAN

- Estimation du parametre **epsilon** via la courbe K-Distance + KneeLocator
- DBSCAN ne necessite **pas de specifier K** a l'avance : il decouvre automatiquement le nombre de clusters
- Les points qui ne rentrent dans aucun cluster sont etiquetes comme **bruit** (outliers)
- Test de plusieurs valeurs d'epsilon pour trouver le meilleur compromis

Interet metier : les points de bruit sont des reclamations **atypiques** qui pourraient meriter une investigation manuelle.

### 9. Comparaison des methodes

#### Metriques internes (qualite des clusters)

Tableau comparatif des 4 methodes sur Silhouette, Davies-Bouldin et Calinski-Harabasz.

#### Correspondance avec la fraude (metriques externes)

| Metrique | Principe | Interpretation |
|----------|----------|----------------|
| **ARI** (Adjusted Rand Index) | Mesure l'accord entre les clusters et les etiquettes de fraude, ajuste pour le hasard | 0 = aleatoire, 1 = correspondance parfaite |
| **NMI** (Normalized Mutual Information) | Information partagee entre les clusters et la fraude | 0 = independants, 1 = information complete |
| **Taux de fraude par cluster** | Proportion de reclamations frauduleuses dans chaque cluster | Un cluster avec un taux tres superieur a la moyenne = segment a risque |

#### Visualisation finale

Vue d'ensemble sur UMAP : fraude reelle + les 4 methodes de clustering cote a cote, avec un recap des metriques.

### 10. Conclusion

Analyse des resultats et recommandations :

- Les fraudeurs **imitent les comportements legitimes** : la fraude ne forme generalement pas un cluster isole. Des ARI/NMI faibles sont donc attendus.
- Certains clusters concentrent neanmoins un **taux de fraude superieur a la moyenne**, ce qui permet de prioriser les investigations.
- Le clustering apporte une **vision complementaire** au modele supervise, utile pour la segmentation du portefeuille et l'audit cible.

## Installation et lancement

### Prerequis

- Python 3.9+

### Dependances

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy kneed umap-learn
```

### Lancement

```bash
jupyter notebook clustering_analysis.ipynb
```

## Technologies utilisees

- **Analyse :** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SciPy
- **Reduction :** PCA, t-SNE (sklearn), UMAP (umap-learn)
- **Clustering :** KMeans, AgglomerativeClustering, GaussianMixture, DBSCAN
- **Evaluation :** Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, NMI
- **Selection K :** KneeLocator (kneed), BIC/AIC
- **Interpretation :** DecisionTreeClassifier, dendrogramme (scipy)
