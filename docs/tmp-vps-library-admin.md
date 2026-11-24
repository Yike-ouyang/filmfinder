# FilmFinder — librairie et dashboard admin sur VPS

> Document temporaire d’exploitation. Il peut être supprimé après la mise en
> production. La branche de référence est `feature/admin-dashboard`, basée sur
> `feature/film-library`.

## 1. Technologies

### Backend

- Python 3.9+
- FastAPI + Uvicorn
- Pydantic
- SQLAlchemy + Alembic
- SQLite (WAL)
- NumPy / SciPy pour le moteur existant
- Moteur M5/KNN inchangé : aucun réentraînement, aucune modification des poids
  ou de la configuration KNN

### Frontend

- Next.js 16 avec `output: "export"`
- React 19
- TypeScript
- Tailwind CSS 4
- Motion for React
- Lucide React
- Graphiques SVG natifs pour l’admin : aucune dépendance graphique ajoutée

### Infrastructure

- Un seul VPS, un vCPU, environ 1,6 GiB de RAM utilisable
- Caddy pour le HTTPS, les fichiers statiques et le reverse proxy `/api/*`
- systemd pour `filmfinder-api`
- Aucun Redis, Celery, service analytics externe ou worker supplémentaire

---

## 2. Ce que contient la livraison

### Librairie `/films`

- Catalogue paginé côté SQL : 50 films maximum par page.
- Recherche par titre normalisé (casse et accents ignorés).
- Filtre par genre.
- Tris : popularité, note pondérée, date de sortie.
- Limite temporelle visible : films jusqu’en 2019.
- Fiches détaillées avec affiche, résumé, genres, date et état personnel.
- Notes de 1 à 10 avec étoiles et demi-étoiles.
- Watchlist / Do Not Watch réutilisant les décisions Discover.
- Génération manuelle de recommandations à partir de 10 notes.

Endpoints principaux :

```http
GET  /api/v1/films
GET  /api/v1/films/facets
GET  /api/v1/films/{film_id}
GET  /api/v1/me/profile-summary
PUT  /api/v1/me/ratings/{film_id}
DELETE /api/v1/me/ratings/{film_id}
POST /api/v1/recommendations/from-saved-ratings
```

Tables ajoutées par la migration `0002_catalog_films` :

```text
catalog_films
catalog_film_genres
```

La source des résumés et dates est `movie_data.csv`, fichier brut d’environ
163 Mo, non versionné. Les identifiants FilmFinder/TMDB, les genres et les
affiches viennent des artefacts existants.

### Dashboard admin `/admin`

- Accessible aux seuls comptes `users.is_admin = 1`.
- 401 pour un visiteur anonyme, 403 pour un compte ordinaire.
- Vue read-only des comptes, inscriptions, connexions, utilisateurs actifs,
  profils vérifiés, notes, générations, imports Letterboxd et listes.
- Graphiques journaliers SVG : inscriptions, utilisateurs actifs,
  recommandations.
- Périodes : 7 jours, 30 jours, depuis le lancement.
- Table utilisateurs paginée avec recherche et fiche détaillée.
- Aucun mot de passe, hash, token, session, IP ou user-agent exposé.

Endpoints :

```http
GET /api/v1/admin/overview
GET /api/v1/admin/activity?period=7d|30d|all
GET /api/v1/admin/users?page=1&page_size=50&search=
GET /api/v1/admin/users/{user_id}
```

Migration `0003_admin_dashboard` :

```text
users.is_admin
analytics_events
index de reporting
```

L’admin ne dispose d’aucun endpoint de modification.

Définitions importantes :

- **Utilisateurs actifs** : utilisateurs distincts avec au moins une connexion
  réussie sur la période.
- **Générations de recommandations** : batches `ready` uniquement, datés par
  `completed_at`.
- **Imports Letterboxd** : événements enregistrés après un import réussi. Les
  imports antérieurs au déploiement du suivi ne sont pas inventés.
- **Imports anonymes** : inclus dans le total, mais pas dans le nombre
  d’utilisateurs.
- **Notes et listes** : état courant, pas un historique de tous les événements.
- Les dates sont en UTC.

---

## 3. Préparation du serveur

Variables d’environnement utiles dans `apps/backend/.env` :

```env
FILMFINDER_ARTIFACTS_DIR=/home/filmfinder/filmfinder/packages/recommendation-engine/artifacts
FILMFINDER_DB_PATH=/var/lib/filmfinder/filmfinder.db
CORS_ORIGINS=https://filmfinder.example
APP_BASE_URL=https://filmfinder.example
COOKIE_SECURE=true
ANALYTICS_RETENTION_DAYS=365
```

Le répertoire SQLite doit appartenir à l’utilisateur du service :

```bash
sudo mkdir -p /var/lib/filmfinder
sudo chown filmfinder:filmfinder /var/lib/filmfinder
sudo chmod 750 /var/lib/filmfinder
```

Le fichier `movie_data.csv` n’est pas dans Git. Pour la synchronisation du
catalogue, il faut soit le copier temporairement sur le VPS, soit exécuter la
synchronisation sur une machine disposant du fichier et importer ensuite les
deux tables du catalogue.

---

## 4. Mise à jour du code et des dépendances

Adapter `filmfinder` au nom d’utilisateur et au chemin réels du VPS.

```bash
cd /home/filmfinder/filmfinder

git fetch --all
git checkout feature/admin-dashboard
git pull

source .venv/bin/activate
pip install -e packages/recommendation-engine -e apps/backend
```

Si le déploiement utilise une branche déjà fusionnée dans `main`, remplacer la
branche ci-dessus par `main`.

---

## 5. Sauvegarde avant migration

```bash
cd /home/filmfinder/filmfinder
source .venv/bin/activate

.venv/bin/python apps/backend/scripts/backup_sqlite.py \
  --source /var/lib/filmfinder/filmfinder.db \
  --output-dir /var/lib/filmfinder/backups \
  --retention 14
```

Vérifier que la sauvegarde existe avant de continuer.

---

## 6. Migration SQLite

Arrêter le service le temps de la migration :

```bash
sudo systemctl stop filmfinder-api

cd /home/filmfinder/filmfinder
source .venv/bin/activate
cd apps/backend

alembic -c alembic.ini upgrade head
alembic -c alembic.ini current
```

Vérifier que la révision affichée est `0003_admin_dashboard`.

Migrations concernées :

```text
0002_catalog_films     catalogue /films
0003_admin_dashboard   rôle admin + analytics
```

Si la base est déjà en `0003_admin_dashboard`, aucune migration n’est rejouée.

---

## 7. Synchronisation du catalogue

Depuis la racine du repository, avec le venv activé :

```bash
cd /home/filmfinder/filmfinder

.venv/bin/python apps/backend/scripts/sync_film_catalog.py \
  --catalogue-csv /chemin/vers/movie_data.csv \
  --artifacts packages/recommendation-engine/artifacts
```

Contrôle rapide :

```bash
.venv/bin/python - <<'PY'
import sqlite3
con = sqlite3.connect("/var/lib/filmfinder/filmfinder.db")
print("films:", con.execute("select count(*) from catalog_films").fetchone()[0])
print("library:", con.execute(
    "select count(*) from catalog_films where in_library=1"
).fetchone()[0])
print("genres:", con.execute(
    "select count(distinct genre) from catalog_film_genres"
).fetchone()[0])
PY
```

Valeurs attendues avec les artefacts actuels :

```text
films   40981
library 38705
genres  19
```

Le script est idempotent et ne fait aucun appel TMDB.

---

## 8. Attribution du rôle administrateur

Créer d’abord un compte via `/signup`, puis attribuer le rôle depuis le shell
du serveur :

```bash
cd /home/filmfinder/filmfinder
source .venv/bin/activate

.venv/bin/python apps/backend/scripts/grant_admin.py \
  --email admin@example.com
```

Révoquer :

```bash
.venv/bin/python apps/backend/scripts/grant_admin.py \
  --email admin@example.com \
  --revoke
```

Le rôle ne peut pas être choisi depuis l’inscription ou une API publique.

---

## 9. Redémarrage et vérification du backend

```bash
sudo systemctl start filmfinder-api
sudo systemctl status filmfinder-api --no-pager

curl --fail-with-body https://filmfinder.example/health
curl --fail-with-body https://filmfinder.example/api/v1/ready
curl -s -o /dev/null -w '%{http_code}\n' \
  https://filmfinder.example/api/v1/films?page=1&page_size=5
```

Sans cookie de session, l’admin doit renvoyer 401 :

```bash
curl -s -o /dev/null -w '%{http_code}\n' \
  https://filmfinder.example/api/v1/admin/overview
```

---

## 10. Build et déploiement du frontend statique

```bash
cd /home/filmfinder/filmfinder/apps/frontend

npm ci
NEXT_PUBLIC_API_BASE_URL=https://filmfinder.example npm run build
rsync -avh out/ filmfinder@VPS:/var/www/filmfinder/
```

Le site statique ne nécessite aucun redémarrage. Le déploiement backend doit
être fait avant le frontend lorsque de nouvelles routes API sont introduites.

Routes attendues dans le build :

```text
/films
/admin
```

Caddy doit uniquement servir `/var/www/filmfinder` en statique et proxyfier
`/api/*` et `/health` vers FastAPI. Ne jamais exposer :

```text
/var/lib/filmfinder/filmfinder.db
apps/backend/var/
movie_data.csv
packages/recommendation-engine/artifacts/
```

---

## 11. Utilisation du dashboard

1. Se connecter avec le compte administrateur sur
   `https://filmfinder.example/login`.
2. Ouvrir `https://filmfinder.example/admin`.
3. Si le rôle vient d’être accordé et que la session était déjà ouverte,
   recharger la page ou se reconnecter : le frontend doit récupérer
   `is_admin: true` via `/api/v1/auth/me`.
4. Le lien **Admin** n’apparaît dans la navbar que pour les administrateurs.

Un compte ordinaire reçoit 403 ; un visiteur anonyme reçoit 401.

---

## 12. Purge des analytics

Les événements servent uniquement aux imports Letterboxd. La politique de
rétention est de 365 jours :

```bash
cd /home/filmfinder/filmfinder
source .venv/bin/activate

.venv/bin/python apps/backend/scripts/prune_analytics.py \
  --retention-days 365
```

Un timer systemd peut exécuter cette commande quotidiennement. Ce n’est pas un
service permanent supplémentaire.

---

## 13. Rollback rapide

En cas de problème après migration :

1. Arrêter `filmfinder-api`.
2. Restaurer la sauvegarde SQLite créée à l’étape 5.
3. Revenir au commit précédent.
4. Redémarrer le service.

```bash
sudo systemctl stop filmfinder-api
# restaurer le fichier .db depuis /var/lib/filmfinder/backups
sudo systemctl start filmfinder-api
```

Les migrations `0003` ajoutent une colonne et une table ; ne pas tenter de
rétrograder manuellement une base de production sans sauvegarde vérifiée.
