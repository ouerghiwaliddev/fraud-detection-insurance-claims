# Insurance Fraud Classification - Detection de Fraude Assurance

## Description du projet

Ce projet realise une analyse complete de la detection de fraude dans les reclamations d'assurance, depuis l'exploration des donnees jusqu'au deploiement d'une application web. Il combine un notebook d'analyse ML, une API de prediction (FastAPI) et une interface utilisateur interactive (Streamlit).

## Structure du projet

```
Insurance Fraud Classification/
├── README.md
├── requirements.txt              # Dependances backend + frontend
├── insurance_claims.csv          # Dataset source
├── insurance_fraud_analysis.ipynb # Notebook d'analyse complet
├── backend/
│   ├── prepare_model.py          # Script d'entrainement et export des artefacts
│   ├── app.py                    # API FastAPI
│   ├── model.pkl                 # Modele XGBoost entraine
│   ├── scaler.pkl                # StandardScaler
│   ├── label_encoder_sex.pkl     # LabelEncoder pour insured_sex
│   ├── feature_names.pkl         # Ordre des features
│   └── cat_mappings.pkl          # Mappings des colonnes one-hot
└── frontend/
    └── frontend.py               # Interface Streamlit
```

## Dataset

**Fichier :** `insurance_claims.csv`

- **1 000 observations** de reclamations d'assurance
- **39 colonnes** (apres suppression de la colonne vide `_c39`)

### Variables principales :

#### Informations sur l'assure

| Colonne                    | Type    | Description                                      |
|---------------------------|---------|--------------------------------------------------|
| months_as_customer        | int     | **Anciennete du client** : nombre de mois depuis la souscription du contrat d'assurance |
| age                       | int     | **Age de l'assure** en annees                    |
| insured_sex               | str     | **Sexe** de l'assure (MALE, FEMALE)              |
| insured_education_level   | str     | **Niveau d'etudes** : JD (Doctorat en Droit), MD (Doctorat en Medecine), PhD (Doctorat), Masters, College (equivalent Licence), Associate (Bac+2), High School (Lycee) |
| insured_occupation        | str     | **Profession** de l'assure (ex: exec-managerial, tech-support, sales...) |
| insured_hobbies           | str     | **Loisirs** de l'assure. Utilise par les assureurs pour evaluer le profil de risque : des hobbies dangereux (base-jumping, skydiving) peuvent indiquer un profil plus risque |
| insured_relationship      | str     | **Situation familiale** du titulaire : husband (mari), wife (epouse), own-child (enfant a charge), unmarried (celibataire), not-in-family, other-relative |
| capital-gains             | int     | **Plus-values financieres** declarees par l'assure (en $). Peut indiquer le niveau de richesse |
| capital-loss              | int     | **Moins-values financieres** declarees (en $, valeur negative). Pertes sur investissements |

#### Informations sur la police d'assurance

| Colonne                    | Type    | Description                                      |
|---------------------------|---------|--------------------------------------------------|
| policy_number             | int     | **Numero de police** : identifiant unique du contrat d'assurance |
| policy_bind_date          | str     | **Date de souscription** de la police d'assurance |
| policy_state              | str     | **Etat americain** ou la police a ete souscrite. OH = Ohio, IL = Illinois, IN = Indiana. Chaque etat a ses propres lois et reglementations en matiere d'assurance |
| policy_csl                | str     | **Limites de responsabilite civile** (Combined Single Limit). Exprime sous la forme X/Y (en milliers de $) : X = montant maximum paye **par personne** blessee, Y = montant maximum paye **par accident**. Ex: 250/500 signifie 250 000$ max par personne et 500 000$ max par accident |
| policy_deductable         | int     | **Franchise** : montant (en $) que l'assure doit payer de sa poche avant que l'assurance ne prenne en charge le reste. Ex: avec une franchise de 1 000$, si les reparations coutent 5 000$, l'assure paie 1 000$ et l'assureur paie 4 000$. Une franchise elevee = prime moins chere |
| policy_annual_premium     | float   | **Prime annuelle** : montant (en $) que l'assure paie chaque annee pour etre couvert. C'est le "prix" de l'assurance |
| umbrella_limit            | int     | **Limite parapluie** : couverture supplementaire (en $) qui s'ajoute au-dela des limites de la police standard. Agit comme un "filet de securite" en cas de sinistre tres couteux. 0 signifie que l'assure n'a pas de couverture supplementaire |

#### Informations sur l'incident

| Colonne                    | Type    | Description                                      |
|---------------------------|---------|--------------------------------------------------|
| incident_date             | str     | **Date de l'incident**                           |
| incident_type             | str     | **Type d'incident** : Single Vehicle Collision (accident avec un seul vehicule, ex: sortie de route), Multi-vehicle Collision (carambolage / accident impliquant plusieurs vehicules), Vehicle Theft (vol de vehicule), Parked Car (vehicule endommage a l'arret) |
| collision_type            | str     | **Type de collision** : Side Collision (choc lateral), Rear Collision (choc arriere), Front Collision (choc frontal). Vaut `?` en cas de vol ou vehicule gare |
| incident_severity         | str     | **Gravite de l'incident** : Trivial Damage (degats insignifiants), Minor Damage (degats legers), Major Damage (degats importants), Total Loss (perte totale, vehicule irreparable) |
| authorities_contacted     | str     | **Autorites contactees** apres l'incident : Police, Fire (pompiers), Ambulance, Other (autre), None (aucune). L'absence de contact avec les autorites peut etre un indicateur de fraude |
| incident_state            | str     | **Etat americain** ou l'incident s'est produit (peut differer de l'etat de la police) |
| incident_city             | str     | **Ville** ou l'incident s'est produit            |
| incident_location         | str     | **Adresse precise** de l'incident                |
| incident_hour_of_the_day  | int     | **Heure de l'incident** (0-23). Les incidents nocturnes peuvent etre plus suspects |
| number_of_vehicles_involved | int   | **Nombre de vehicules impliques** dans l'incident (1 a 4) |
| property_damage           | str     | **Dommages materiels** constates (YES/NO). Concerne les degats sur des biens autres que les vehicules (cloture, poteau, batiment...) |
| bodily_injuries           | int     | **Nombre de blessures corporelles** signalees (0 a 3) |
| witnesses                 | int     | **Nombre de temoins** de l'incident. Peu ou pas de temoins peut etre suspect |
| police_report_available   | str     | **Rapport de police disponible** (YES/NO). L'absence de rapport peut etre un signal de fraude |

#### Informations sur la reclamation

| Colonne                    | Type    | Description                                      |
|---------------------------|---------|--------------------------------------------------|
| total_claim_amount        | float   | **Montant total reclame** (en $). C'est la somme des 3 sous-reclamations ci-dessous |
| injury_claim              | float   | **Reclamation pour blessures** (en $) : frais medicaux, hospitalisation, douleur et souffrance |
| property_claim            | float   | **Reclamation pour dommages materiels** (en $) : reparation/remplacement de biens endommages (hors vehicule) |
| vehicle_claim             | float   | **Reclamation pour le vehicule** (en $) : reparation ou valeur de remplacement du vehicule endommage |

#### Informations sur le vehicule

| Colonne                    | Type    | Description                                      |
|---------------------------|---------|--------------------------------------------------|
| auto_make                 | str     | **Marque du vehicule** (Toyota, BMW, Ford, etc.) |
| auto_model                | str     | **Modele du vehicule** (Camry, X5, F150, etc.)   |
| auto_year                 | int     | **Annee de fabrication** du vehicule              |

#### Variable cible

| Colonne                    | Type    | Description                                      |
|---------------------------|---------|--------------------------------------------------|
| **fraud_reported**        | **str** | **Fraude detectee** : Y (Yes) = la reclamation a ete identifiee comme frauduleuse, N (No) = reclamation legitime. C'est la variable que le modele cherche a predire |

### Valeurs manquantes

Certaines colonnes contiennent des `?` au lieu de vraies valeurs. Cela concerne 3 colonnes :
- **collision_type** : pas de type de collision pour les vols de vehicules ou vehicules gares (logique)
- **property_damage** : information non renseignee dans certains cas
- **police_report_available** : information non renseignee dans certains cas

Ces `?` sont traites dans le preprocessing en les remplacant par la valeur la plus frequente (le mode) de chaque colonne.

## Notebook d'analyse (`insurance_fraud_analysis.ipynb`)

Le notebook est structure en 8 parties :

### 1. Analyse exploratoire (EDA) approfondie
- Statistiques descriptives et verification des donnees
- Detection des valeurs manquantes (NaN + `?`)
- Distribution de la variable cible (desequilibre des classes)
- Analyse croisee des variables categoriques vs fraude
- Histogrammes avec KDE et courbe normale theorique
- **Analyse Skewness & Kurtosis** avec interpretations et tests de normalite (Shapiro-Wilk)
- Box plots par groupe (fraude vs legitime) + quantification IQR des outliers
- Matrice de correlation et correlations avec la cible
- Violin plots, pairplots, scatter plots bivarries

### 2. Preprocessing
- Remplacement des `?` par le mode
- Suppression des colonnes non pertinentes
- LabelEncoder sur insured_sex
- One-Hot Encoding sur les variables categoriques multi-classes
- StandardScaler pour normalisation
- Split train/test 80/20 avec stratification

### 3. Modelisation baseline (5 modeles)
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree (avec visualisation de l'arbre)
- Random Forest (avec feature importance)
- XGBoost

### 4. Fine-tuning avec GridSearchCV
Optimisation par validation croisee stratifiee (5 folds, scoring F1) sur les 3 meilleurs modeles

### 5. Gestion du desequilibre avec SMOTE
Application de SMOTE sur le jeu d'entrainement pour equilibrer les classes

### 6. Selection de features (SelectKBest)
Tests avec K = 10, 15, 20 features utilisant f_classif

### 7. Cross-validation approfondie
Validation croisee stratifiee 10-Fold avec boxplots

### 8. Conclusion et rapport final

## Backend (FastAPI)

### `prepare_model.py`
Script de preparation qui :
1. Charge et pretraite le dataset insurance_claims.csv
2. Remplace les `?` par le mode de chaque colonne
3. Encode les variables categoriques (LabelEncoder + One-Hot)
4. Entraine le modele XGBoost avec GridSearchCV
5. Exporte 5 fichiers pickle : `model.pkl`, `scaler.pkl`, `label_encoder_sex.pkl`, `feature_names.pkl`, `cat_mappings.pkl`

### `app.py`
API REST avec deux endpoints :

| Methode | Endpoint   | Description                                          |
|---------|-----------|------------------------------------------------------|
| GET     | `/`       | Verification que l'API est active                    |
| POST    | `/predict`| Prediction de fraude pour une reclamation             |

Le body du POST accepte un JSON avec les features de la reclamation. Toutes les features ont des **valeurs par defaut** (medianes/modes du dataset), donc un body `{}` vide est valide.

Exemple de requete :
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 48, "incident_type": "Single Vehicle Collision", "incident_severity": "Total Loss", "total_claim_amount": 95000}'
```

Exemple de reponse :
```json
{
  "prediction": 1,
  "fraud_probability": 0.7823,
  "label": "Fraud",
  "risk_level": "High",
  "input_summary": {
    "age": 48,
    "incident_type": "Single Vehicle Collision",
    "incident_severity": "Total Loss",
    "total_claim_amount": 95000,
    "police_report_available": "YES"
  }
}
```

Documentation interactive : **http://localhost:8000/docs** (Swagger UI)

## Installation et lancement

### Prerequis
- Python 3.9+

### Installation des dependances
```bash
cd "Insurance Fraud Classification"
pip install -r requirements.txt
```

### Regenerer les fichiers pickle
```bash
cd backend
python prepare_model.py
```

### Lancer l'application

**Terminal 1 - Backend :**
```bash
cd backend
uvicorn app:app --host 127.0.0.1 --port 8000
```

**Terminal 2 - Frontend :**
```bash
cd frontend
streamlit run frontend.py
```

### Acces
| Service  | URL                          |
|----------|------------------------------|
| API      | http://localhost:8000         |
| Swagger  | http://localhost:8000/docs    |
| Frontend | http://localhost:8501         |

## Technologies utilisees

- **Analyse :** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, SciPy, imbalanced-learn
- **Backend :** FastAPI, Uvicorn, Pydantic
- **Frontend :** Streamlit, Plotly
- **Serialisation :** Pickle
