# Production Server Deployment

The public application consists of one scientific FastAPI service and its
bundled same-origin browser frontend. The production package adds Caddy for
automatic HTTPS and exposes no application-container port directly to the
internet.

## Architecture

```text
browser -> HTTPS :443 -> Caddy -> private Docker network -> model-api :8000
```

The application container:

- runs as non-root user `modelapi`;
- has a read-only filesystem and a small temporary filesystem;
- drops Linux capabilities and forbids privilege escalation;
- has CPU, memory, and process limits;
- exposes only its health-checked port to the private Docker network;
- stores no user inputs, inferred values, or exported workbooks.

Caddy terminates TLS, adds security headers, limits request bodies, performs
upstream health checks, and writes bounded JSON logs through Docker.

## Prerequisites

Use a current Linux server with:

- a public IPv4 or IPv6 address;
- a domain or subdomain whose DNS record points to the server;
- Docker Engine and the Docker Compose plugin;
- inbound TCP ports 80 and 443 and UDP port 443 open;
- SSH restricted according to the server provider's recommendations.

The deployment itself does not require Conda, Photochem, ERA5, or any source
papers.

The production image uses a pinned Python 3.12.14 base tag and the exact Linux
runtime in `code/requirements-api-lock.txt`. The resulting image digest should
be recorded with the software release and Zenodo metadata after the final
server build.

## Configure

Check out the exact release tag rather than a moving development branch:

```bash
git clone https://github.com/FabianZah/atmospheric-o2-triple-isotope-model.git
cd atmospheric-o2-triple-isotope-model
git checkout v0.1.0
```

Create the untracked production environment file:

```bash
cp deploy/.env.production.example deploy/.env.production
nano deploy/.env.production
```

Set at least:

```text
O2_MODEL_DOMAIN=model.example.org
ACME_EMAIL=researcher@example.org
```

Leave `O2_MODEL_CORS_ORIGINS` empty for the bundled browser application. Only
set it when a separately hosted frontend needs API access, using exact HTTPS
origins separated by commas.

The default application limits are 2 CPU cores, 2 GB memory, and 256 processes.
They can be changed in `deploy/.env.production` without editing the versioned
Compose file.

## Start

From the repository root:

```bash
docker compose \
  --env-file deploy/.env.production \
  -f deploy/compose.production.yaml \
  up --build -d
```

Inspect startup and health:

```bash
docker compose \
  --env-file deploy/.env.production \
  -f deploy/compose.production.yaml \
  ps

docker compose \
  --env-file deploy/.env.production \
  -f deploy/compose.production.yaml \
  logs --tail=100 model-api caddy
```

Caddy obtains and renews the HTTPS certificate automatically after DNS and
ports are correct.

## Verify

The verifier checks the health endpoint, publication-model identity,
operational domain, and the modern 294 ppm known answer:

```bash
python validation/verify_public_deployment.py \
  --base-url https://model.example.org
```

Also inspect:

- `https://model.example.org/` for the browser interface;
- `https://model.example.org/api/v1/health` for service health;
- `https://model.example.org/docs` for the generated API schema.

The same verifier runs against the built Linux container in continuous
integration.

## Logs And Privacy

```bash
docker compose \
  --env-file deploy/.env.production \
  -f deploy/compose.production.yaml \
  logs --follow --tail=100
```

Docker rotates both service logs at 10 MB and retains five files. Caddy access
logs contain request metadata such as timestamp, path, status, and client
address; they do not contain JSON request bodies. Select a retention policy
consistent with institutional privacy requirements.

## Update

Record the currently deployed tag first:

```bash
git describe --tags --always
```

Then check out a reviewed release and rebuild:

```bash
git fetch --tags
git checkout v0.1.1
docker compose \
  --env-file deploy/.env.production \
  -f deploy/compose.production.yaml \
  up --build -d
python validation/verify_public_deployment.py \
  --base-url https://model.example.org
```

Do not deploy directly from an unreviewed working branch.

## Roll Back

Check out the previous tag and rebuild the application image:

```bash
git checkout v0.1.0
docker compose \
  --env-file deploy/.env.production \
  -f deploy/compose.production.yaml \
  up --build -d
python validation/verify_public_deployment.py \
  --base-url https://model.example.org
```

The service has no database migration or persistent scientific state, so
rollback is limited to code, model data, and container dependencies.

## Backup

Keep private backups of:

- `deploy/.env.production`;
- the exact Git release tag and Zenodo archive DOI;
- optional Caddy volumes `o2-budget-model_caddy_data` and
  `o2-budget-model_caddy_config`.

The Caddy volumes contain certificate and proxy state, not scientific results.
They can be recreated from DNS and the environment configuration if necessary.

## Capacity And Scope

The default stack intentionally runs one application process. This bounds
memory use and is appropriate for low-volume scientific access. API grid sizes
are also bounded by the request schema. The reverse proxy allows up to five
minutes for a response so declared transient calculations can finish.

Before advertising the service for high-volume anonymous use, measure request
latency on the real server and add an upstream rate limiter or bounded job
queue. That is an operational scaling change; it must not alter model inputs,
equations, or returned provenance.

## Local Development Compose

The root `compose.yaml` remains a loopback-only development convenience:

```bash
docker compose up --build -d
```

It does not provide public HTTPS. Use `deploy/compose.production.yaml` for the
internet-facing server.
