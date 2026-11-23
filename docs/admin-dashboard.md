# Admin dashboard

## Overview

`/admin` is a private, read-only dashboard for FilmFinder administrators. The
frontend is part of the static Next.js export, but every `/api/v1/admin/*`
endpoint independently verifies the session and the `users.is_admin` flag.

No external analytics service is used. The dashboard reuses SQLite timestamps
and a small `analytics_events` table for information that cannot be
reconstructed reliably.

## Administrator role

`users.is_admin` is a boolean added by migration `0003_admin_dashboard`. It is
never accepted from signup or any public API. Assign or revoke it from a shell
on the server:

```bash
python apps/backend/scripts/grant_admin.py --email admin@example.com
python apps/backend/scripts/grant_admin.py --email admin@example.com --revoke
```

The user must already exist. Authentication returns `is_admin` so the frontend
can hide the Admin link, but the backend always performs the real check:
anonymous requests receive 401, ordinary accounts receive 403, administrators
receive the dashboard.

## Metrics

All timestamps use UTC. “Last 7/30 days” means the 7/30 calendar days ending
today, including today.

| Metric | Definition | Historical coverage |
|---|---|---|
| Total accounts | `COUNT(users)` | Full |
| Registrations 7/30 | Users with `created_at` in the window | Full |
| Active users 7/30 | Distinct users with at least one successful login (`sessions.created_at`) in the window | Full |
| Successful logins | Number of `sessions` rows; one is created per successful login | Full |
| Unique login users | Distinct `sessions.user_id` | Full |
| Verified profiles | Users with `email_verified_at` not null | Full |
| Films rated | `user_films.rating IS NOT NULL` | Current state |
| Rating users | Distinct users with a current rating | Current state |
| Recommendation generations | `recommendation_batches.status = 'ready'` | Since batches were introduced |
| Recommendation users | Distinct batch users | Since batches were introduced |
| Letterboxd imports | Rows in `analytics_events` with `event_type='letterboxd_imported'` | Since import tracking was deployed |
| Import users | Distinct non-null event `user_id` | Since import tracking |
| Interested films | Current `user_swipes.action='interested'` rows | Current state |
| Interested users | Distinct users with at least one Interested film | Current state |
| Not Interested films | Current `user_swipes.action='not_interested'` rows | Current state |
| Not Interested users | Distinct users with at least one Not Interested film | Current state |

Important distinctions:

- **Active users** here means users who successfully logged in during the
  window. We deliberately do not create a separate daily activity event.
- **Logins** are total session-creation events; unique users are displayed
  separately.
- **Recommendations** count completed `ready` batches only. Failed or
  superseded batches are excluded.
- **Imports** include anonymous imports in the total (`user_id IS NULL`) but
  anonymous events cannot appear as a signed-in user count. Historical imports
  before tracking are not counted and are never invented.
- **Ratings and lists** are current states, not lifetime event counts. A rating
  change or deletion changes the current total.
- The per-user “last activity” field is the most recent of last login, last
  rating update, last ready recommendation batch and last import event. It is
  not a page-view tracker.
- No metric exposes passwords, password hashes, tokens, session identifiers,
  IP addresses, user agents, or raw archives.

## Charts

The activity endpoint returns one point per day for:

- registrations: `users.created_at`;
- active users: distinct users with a successful login that day;
- recommendations: ready batches completed that day.

The frontend renders native accessible SVG bar charts, with an `sr-only` data
table. No chart dependency is required and the static export stays small.
Periods are `7d`, `30d` and `all`.

## Analytics events

`analytics_events` intentionally stores only information that cannot be derived
from existing tables:

```text
user_registered           derived from users.created_at
email_verified            derived from users.email_verified_at
login_success             derived from sessions.created_at
recommendations_generated derived from recommendation_batches
film_rated                current state in user_films
film_swiped               current state + user_swipe_events
letterboxd_imported       recorded in analytics_events
```

An event is inserted only after the import request has completed successfully.
Anonymous imports store `user_id = NULL`. There is one event per successful
import request; re-importing the same archive is a new import and is counted.

Retention is `ANALYTICS_RETENTION_DAYS` (default 365). Prune with:

```bash
python apps/backend/scripts/prune_analytics.py
```

Registrations, logins, recommendations and current profile data are not
affected by analytics-event retention.

## API

```http
GET /api/v1/admin/overview
GET /api/v1/admin/activity?period=7d|30d|all
GET /api/v1/admin/users?page=1&page_size=50&search=...
GET /api/v1/admin/users/{user_id}
```

There are no admin mutation endpoints. All responses are read-only.
User list/detail responses include safe aggregate fields only; sessions and
credentials are never returned.

## Security and performance

- Session lookup is the only authorization source; no client-supplied user ID.
- `get_current_admin` returns 401 for anonymous and 403 for non-admin accounts.
- Admin queries are indexed aggregate/count queries. The user list limits the
  page before computing per-user aggregates, avoiding N+1 queries.
- The dashboard does not load the ML engine or any catalogue rows.
- SQLite remains the only data store; no worker, Redis, Celery or external
  analytics service is added.
