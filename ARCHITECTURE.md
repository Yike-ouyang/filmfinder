# Architecture du repository FilmFinder

## 1. Vue d'ensemble

FilmFinder transforme un profil Letterboxd en recommandations de films.

Le repository contient quatre ensembles :

1. une préparation des données brutes et un entraînement de Factorization
   Machines contextuelles ;
2. des expériences d'évaluation et d'optimisation de la recommandation ;
3. un explorateur marimo pour inspecter le pipeline utilisateur par utilisateur ;
4. un package de déploiement autonome capable d'importer un ZIP Letterboxd et
   de générer des recommandations sans réentraîner le modèle global.

Le modèle final est un **M5 contextuel** : un Factorization Machine qui utilise
l'utilisateur, le film, les genres, l'année et les studios. La recommandation
finale utilise une short-list KNN puis un classement M5.

```text
Letterboxd ZIP
      |
      v
letterboxd_import.py
      |
      v
profil utilisateur froid (notes + films vus)
      |
      v
FilmFinderEngine
  KNN centré -> candidats -> exclusion des films vus -> pool 100
      |
      v
M5 froid (ridge sur les facteurs utilisateur)
      |
      v
Top N recommandations
```

---

## 2. Arborescence commentée

```text
.
├── scripts/
│   ├── data_cleaning.py
│   ├── 1.Preprocessing.py
│   ├── 2.Processing.py
│   ├── 3.ExploringBias.py
│   ├── 4.DebugTraining.py
│   ├── 5.ContextFM.py
│   ├── 6.IncrementalVsRetraining.py
│   ├── 7.EvaluateM5.py
│   └── legacy/
├── recommandation/
│   ├── scripts/
│   │   ├── 8.RecommendationSystem.py
│   │   ├── 9.KNNM5Optimization.py
│   │   ├── 9.RecommendationExplorer.py
│   │   ├── recommendation_explorer_core.py
│   │   ├── test_recommendation_explorer_core.py
│   │   └── export_filmfinder_deployment.py
│   ├── recommendation_system/
│   ├── knn_m5_optimization/
│   └── prompts/
├── filmfinder_deployment/
│   ├── artifacts/
│   ├── filmfinder/
│   ├── letterboxd_import.py
│   ├── generate_recommendations.py
│   ├── verify_export.py
│   ├── test_letterboxd_pipeline.py
│   ├── requirements.txt
│   └── README.md
├── results/
├── docs/
├── prompts/
├── README.md
└── requirements.txt
```

### Rôles des principaux fichiers

| Fichier | Rôle |
|---|---|
| `scripts/data_cleaning.py` | Notebook de préparation final : CSV bruts → matrice CSR, IDs, contexte film |
| `scripts/1.Preprocessing.py` | Variante simplifiée de préparation |
| `scripts/2.Processing.py` | Entraînement des FM et diagnostics, incluant M5 |
| `scripts/3.ExploringBias.py` | Analyse des biais et des prédictions |
| `scripts/4.DebugTraining.py` | Diagnostics d'entraînement et régularisation |
| `scripts/5.ContextFM.py` | Comparaison FM user-film et FM contextuel |
| `scripts/6.IncrementalVsRetraining.py` | Adaptation locale vs réentraînement, `FMEngine`, features, splits |
| `scripts/7.EvaluateM5.py` | Évaluation warm/cold de M5 et calibration ridge |
| `recommandation/scripts/8.RecommendationSystem.py` | Évaluation Top-20 des stratégies de recommandation |
| `recommandation/scripts/9.KNNM5Optimization.py` | Optimisation KNN+M5 sur validation froide |
| `recommandation/scripts/9.RecommendationExplorer.py` | Notebook marimo d'explicabilité |
| `recommandation/scripts/recommendation_explorer_core.py` | Pipeline instrumenté réutilisé par l'explorateur |
| `recommandation/scripts/export_filmfinder_deployment.py` | Réentraînement final sur toutes les notes et export |
| `filmfinder_deployment/filmfinder/` | Moteur d'inférence autonome |
| `filmfinder_deployment/letterboxd_import.py` | Import et matching du ZIP Letterboxd |
| `filmfinder_deployment/generate_recommendations.py` | CLI d'inférence complète |

---

## 3. Contrats de données

### 3.1 Données nettoyées

Dossier défini par `FILMFINDER_CLEANING_DIR` :

```text
ratings_matrix.npz
user_ids.npy
movie_ids.npy
movie_context.json
```

| Fichier | Contenu |
|---|---|
| `ratings_matrix.npz` | Matrice creuse CSR utilisateurs × films, notes 1 à 10 |
| `user_ids.npy` | Identifiants alignés avec les lignes de la matrice |
| `movie_ids.npy` | Identifiants canoniques alignés avec les colonnes |
| `movie_context.json` | Métadonnées alignées avec les colonnes : titre, année, genres, studios, description |

Le dataset final de référence contient environ :

```text
9 728 981 notes
7 369 utilisateurs
40 981 films
```

### 3.2 Artefacts du système de recommandation

Dossier `recommandation/recommendation_system/` :

```text
config.json
models/M5.npz
splits/
  cold_cohorts.csv
  warm_cohorts.csv
  train_visible_indices.npy
  warm_hidden_indices.npy
selected_configs.json
warm/selected_configs.json
cold/selected_configs.json
```

Ces artefacts contiennent le M5 expérimental, les cohortes et les
configurations sélectionnées par `8.RecommendationSystem.py`.

### 3.3 Artefacts d'optimisation KNN+M5

Dossier `recommandation/knn_m5_optimization/` :

```text
config.json
selection/selected_configs.json
validation/grid_summary.csv
test/
figures/
summary.md
```

La configuration globale retenue expérimentalement est :

```text
algorithme       : knn_m5
similarité       : centered
n_neighbors      : 50
top_k par voisin : 20
pool             : 100
shrink           : 10.0
ranking          : M5
top_n par défaut : 20
```

---

## 4. Pipeline de préparation et d'expérimentation

### 4.1 Préparation

`scripts/data_cleaning.py` :

1. charge les CSV bruts Letterboxd ;
2. fusionne et déduplique les catalogues de films ;
3. filtre les films et utilisateurs trop peu actifs ;
4. construit la matrice creuse utilisateurs × films ;
5. sauvegarde les mappings et `movie_context.json`.

`scripts/1.Preprocessing.py` est une variante de préparation plus légère.

### 4.2 Modélisation

`scripts/2.Processing.py` entraîne plusieurs variantes de Factorization
Machines. M5 est le modèle contextuel final :

```text
features = user one-hot
         + film one-hot
         + genres
         + année normalisée
         + studios
```

Hyperparamètres de référence :

```text
n_factors      = 8
learning_rate  = 0.01
n_epochs       = 20
batch_size     = 65536
reg_bias       = 0
reg_factors    = 1e-6
seed           = 4042
optimiseur     = Adam
max_studios    = 4000
```

Les scripts 3 à 5 analysent les biais, la régularisation et l'apport du
contexte.

### 4.3 Évaluation M5 et cold start

`scripts/6.IncrementalVsRetraining.py` fournit :

- `FMEngine` ;
- le chargement du dataset ;
- le split train/validation/test ;
- `build_features` et le cache de features contextuelles ;
- l'adaptation locale des paramètres utilisateur.

`scripts/7.EvaluateM5.py` évalue :

- les performances warm-start par activité ;
- les performances cold-start ;
- la calibration ridge de l'utilisateur froid.

Régularisation froide retenue :

```text
cold_adapt_reg_bias    = 0.01
cold_adapt_reg_factors = 0.1
```

### 4.4 Recommandation

`recommandation/scripts/8.RecommendationSystem.py` compare :

- popularité bayésienne ;
- M5 direct ;
- KNN + M5 ;
- KNN + popularité.

Il exporte les cohortes, les splits, M5 et les configurations sélectionnées.

`recommandation/scripts/9.KNNM5Optimization.py` optimise uniquement le KNN sur
les utilisateurs froids, avec M5 figé :

1. similarité centrée ou latente ;
2. N voisins ;
3. top_k films par voisin ;
4. union et déduplication des candidats ;
5. suppression des films déjà vus ;
6. tri par support puis popularité ;
7. cap du pool ;
8. classement M5 ;
9. métriques Recall@K et Known NDCG@K.

---

## 5. Modèle et recommandation en production

### 5.1 M5 exporté

Le M5 de déploiement est réentraîné sur **toutes les notes disponibles**, avec
les hyperparamètres ci-dessus et **20 epochs fixes**, sans early stopping.

Il est stocké dans `filmfinder_deployment/artifacts/` avec :

```text
m5_weights.npz
m5_config.json
```

Le moteur ne réentraîne jamais ce modèle.

### 5.2 Features film précomputées

`movie_features.npz` contient une matrice CSR de features non-utilisateur pour
chaque film. Les colonnes utilisateur sont mises à zéro.

Le score M5 est décomposé en :

```text
item_base   = w0 + features_film @ w + terme latent propre au film
item_latent = features_film @ V
```

Pour un utilisateur connu :

```text
score = item_base + w[user] + item_latent · V[user]
```

Pour un nouvel utilisateur :

```text
score = item_base + theta[0] + item_latent · theta[1:]
```

`theta` est obtenu par ridge sur les films notés du nouvel utilisateur.

### 5.3 Short-list KNN

Le moteur de déploiement reconstruit à partir de `ratings_matrix.npz` :

- les moyennes utilisateur ;
- la matrice centrée ;
- les normes ;
- la similarité centrée avec shrinkage ;
- le cache des `top_k` films par voisin ;
- les scores de popularité bayésienne.

Pipeline :

```text
profil visible
  -> similarité centrée
  -> 50 voisins
  -> 20 films par voisin
  -> union + support
  -> suppression des films notés et vus
  -> tri support / popularité / ID
  -> pool 100
  -> scoring M5
  -> top N déterministe
```

Le fallback popularité est utilisé si la similarité ou l'union KNN est vide.

---

## 6. Package de déploiement

```text
filmfinder_deployment/
├── artifacts/
├── filmfinder/
│   ├── engine.py
│   ├── model.py
│   ├── features.py
│   ├── recommender.py
│   └── cold_start.py
├── letterboxd_import.py
├── generate_recommendations.py
├── verify_export.py
├── test_letterboxd_pipeline.py
├── requirements.txt
└── README.md
```

### Modules

| Module | Responsabilité |
|---|---|
| `engine.py` | Façade `FilmFinderEngine`, résolution user/movie, API existant/nouveau |
| `model.py` | Inférence M5 à partir des poids exportés |
| `features.py` | Chargement des artefacts, IDs et métadonnées |
| `recommender.py` | Similarité, voisins, candidats, pool, ranking, fallback |
| `cold_start.py` | Ridge utilisateur froid |
| `letterboxd_import.py` | Parsing ZIP, normalisation 0.5–5 → 1–10, matching |
| `generate_recommendations.py` | CLI complète et rapports JSON |

### Artefacts

| Fichier | Contenu |
|---|---|
| `m5_weights.npz` | `w0`, `w`, `V`, masque actif, dimensions |
| `m5_config.json` | Configuration, seed, features, version, diagnostics d'entraînement |
| `recommendation_config.json` | KNN, pool, shrink, fallback, ridge froide |
| `feature_mappings.json` | Ordre des features, familles, indices user/film |
| `movie_metadata.json` | ID interne, ID canonique, titre, année, genres, IDs source |
| `movie_features.npz` | Features film CSR sans la famille user |
| `user_ids.npy` | Ordre des utilisateurs |
| `movie_ids.npy` | Ordre des films |
| `ratings_matrix.npz` | Matrice CSR utilisée par le KNN |

Les posters, backdrops et champs TMDB explicites sont retournés à `null` s'ils
ne sont pas présents dans les métadonnées. Aucun poster n'est téléchargé.

---

## 7. Import Letterboxd et inférence

### 7.1 Parser

`letterboxd_import.py` :

- lit le ZIP avec `zipfile` et `csv` ;
- utilise `ratings.csv` pour les notes explicites ;
- utilise `watched.csv` pour exclure les films vus sans note ;
- normalise `0,5–5` en `1–10` par multiplication par 2 ;
- fait un matching :

```text
1. identifiant stable si réellement disponible
2. titre normalisé + année exacte
3. titre normalisé + année ± 1
4. similarité de titre >= 0.92 avec année ± 1
```

Les cas ambigus ne sont pas résolus arbitrairement. Les films non trouvés,
ambigus, invalides et les doublons sont signalés.

### 7.2 CLI

```bash
python filmfinder_deployment/generate_recommendations.py \
  --letterboxd-zip filmfinder_deployment/letterboxd-eslpej-2026-11-20-10-33-utc.zip \
  --artifacts filmfinder_deployment/artifacts \
  --top-n 100 \
  --output recommendations.json \
  --report import_report.json
```

Ou depuis la racine :

```bash
python -m filmfinder_deployment.generate_recommendations ...
```

Sorties :

```text
recommendations.json
import_report.json
```

`recommendations.json` contient le top classé avec :

```text
rank, movie_id, internal_movie_id, title, year, genres,
score, score_type, tmdb_id, poster_path, backdrop_path
```

`import_report.json` contient les compteurs de notes, films vus, matchs,
non-matchs, ambigus, doublons, correspondances fuzzy et statistiques de
génération.

---

## 8. Flux de fichiers

```text
CSV Kaggle / Letterboxd
        |
        v
scripts/data_cleaning.py
        |
        +--> ratings_matrix.npz
        +--> user_ids.npy
        +--> movie_ids.npy
        +--> movie_context.json
        |
        v
scripts/2.Processing.py et 7.EvaluateM5.py
        |
        v
recommandation/scripts/8.RecommendationSystem.py
        |
        +--> recommendation_system/models/M5.npz
        +--> recommendation_system/splits/
        +--> recommendation_system/warm|selected_configs.json
        |
        v
recommandation/scripts/9.KNNM5Optimization.py
        |
        +--> knn_m5_optimization/selection/selected_configs.json
        |
        v
recommandation/scripts/export_filmfinder_deployment.py
        |
        v
filmfinder_deployment/artifacts/
        |
        v
letterboxd_import.py + generate_recommendations.py
        |
        v
recommendations.json + import_report.json
```

---

## 9. Entraînement final et export

L'export est lancé hors du dossier de déploiement :

```bash
python recommandation/scripts/export_filmfinder_deployment.py \
  --cleaning-dir /Data/yike-ouyang/ml_training \
  --baseline-dir recommandation/recommendation_system \
  --knn-dir recommandation/knn_m5_optimization \
  --output-dir filmfinder_deployment
```

L'exportateur :

1. charge toutes les notes ;
2. reconstruit les features finales ;
3. réentraîne M5 sur 100 % des notes ;
4. construit les features film précomputées ;
5. exporte les poids, la matrice et les mappings ;
6. vérifie l'équivalence M5 et les Top-20 contre l'implémentation
   expérimentale ;
7. écrit `EXPORT_REPORT.md`.

---

## 10. Environnement et exécution

### Prérequis principaux

```text
Python 3.9+
NumPy
SciPy
marimo pour les notebooks
matplotlib pour les notebooks d'analyse
pandas pour l'exploration et les scripts d'expérimentation
```

Le package de déploiement ne nécessite que `numpy` et `scipy`.

### Variables d'environnement

```bash
export FILMFINDER_CLEANING_DIR=/Data/yike-ouyang/ml_training
export FILMFINDER_MODEL_OUTPUT_DIR=/Data/yike-ouyang/ml_training/modeling
export FILMFINDER_RECOMMENDATION_DIR=recommandation/recommendation_system
export FILMFINDER_KNN_M5_DIR=recommandation/knn_m5_optimization
```

### Explorateur marimo

```bash
marimo edit recommandation/scripts/9.RecommendationExplorer.py
```

---

## 11. Tests et vérifications

### Package de déploiement

```bash
python filmfinder_deployment/verify_export.py \
  --artifacts filmfinder_deployment/artifacts
```

Vérifie :

- chargement des artefacts ;
- recommandation utilisateur existant ;
- recommandation nouvel utilisateur ;
- exclusion des films notés ;
- tri des scores ;
- stabilité des poids globaux après adaptation froide.

### Pipeline Letterboxd

```bash
python -m unittest filmfinder_deployment.test_letterboxd_pipeline -v
```

Vérifie :

- parsing du ZIP fourni ;
- normalisation des notes ;
- matching ;
- absence de films notés/vus dans les recommandations ;
- 100 recommandations lorsque le pool le permet ;
- déterminisme ;
- intégrité des poids M5.

### Explorateur

```bash
FILMFINDER_EXPLORER_TESTS=1 python -m unittest \
  recommandation/scripts/test_recommendation_explorer_core.py -v
```

---

## 12. Fichiers générés et versionnement

Sont générés et ne doivent généralement pas être commités :

```text
results/
recommandation/recommendation_system/
recommandation/knn_m5_optimization/
filmfinder_deployment/artifacts/
recommendations.json
import_report.json
__pycache__/
```

Le code du package de déploiement, les tests et la documentation sont versionnés.
Le ZIP Letterboxd de démonstration est présent comme fixture de test.

---

## 13. Choix et limites

- M5 global reste figé en inférence ; seule l'adaptation utilisateur est
  recalculée.
- La configuration de recommandation est celle sélectionnée expérimentalement,
  sans réoptimisation au moment de l'inférence.
- Le KNN utilise la matrice complète de production, pas les splits d'évaluation.
- Les notes et films vus sont exclus avant le cap du pool.
- Les films Letterboxd non trouvés dans le catalogue ne peuvent pas être
  exclus ; ils sont signalés.
- Les métadonnées actuelles ne fournissent pas de poster/backdrop/TMDB
  explicite ; ces champs restent `null`.
- Le pool de candidats et le top N sont limités par la configuration globale
  (`pool=100`). Si le pool contient moins de films éligibles, la limite est
  reportée.
