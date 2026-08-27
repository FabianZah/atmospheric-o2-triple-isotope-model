# Server deployment

OXYTIB is a stateless FastAPI service with a bundled same-origin browser
interface. The canonical shared-server deployment uses the existing Traefik
instance and publishes OXYTIB below `/oxytib/`. A standalone Caddy stack is
included for servers dedicated to OXYTIB.

## Security boundary

The application container:

- runs as the non-root `modelapi` user;
- has a read-only filesystem and a small `noexec` temporary filesystem;
- drops Linux capabilities and forbids privilege escalation;
- has CPU, memory, and process limits;
- stores no submitted constraints, calculated results, or workbooks;
- rejects request bodies above 1 MiB;
- admits at most two simultaneous model calculations by default;
- validates API fields and bounds model grid sizes before calculation.

The Traefik route adds HTTPS, browser security headers, 30 requests per minute
per client with a burst of 10, and at most four in-flight requests. These are
conservative defaults for a small scientific service and can be changed in the
untracked deployment environment file.

Rate limiting protects model capacity; it does not absorb a network-saturating
distributed denial-of-service attack. Host firewall rules, provider-level
traffic protection, timely security updates, restricted SSH access, and
monitoring remain server-administration responsibilities. A CDN or provider
DDoS service can be placed in front of Traefik if traffic later warrants it.

## Traefik deployment

Prerequisites:

- Docker Engine and the Docker Compose plugin;
- the external Docker network `traefik-global-proxy`;
- an existing Traefik `websecure` entry point and `letsencrypt` resolver;
- a checked-out, reviewed OXYTIB release tag.

Create the untracked configuration:

```bash
cp deploy/.env.traefik.example deploy/.env.traefik
```

Set `OXYTIB_HOST` and retain an empty `OXYTIB_ROOT_PATH` because the declared
Traefik middleware strips `/oxytib` before forwarding requests. The browser
interface and API documentation derive the public prefix from their loaded
URLs. Keep `OXYTIB_TRAEFIK_ENABLE=false` during staging. Build and verify the
loopback service first:

```bash
docker compose \
  --env-file deploy/.env.traefik \
  -f deploy/compose.traefik.yaml \
  up --build -d

curl --fail http://127.0.0.1:18000/api/v1/health
python validation/verify_public_deployment.py \
  --base-url http://127.0.0.1:18000
```

After reviewing the container health and logs, set
`OXYTIB_TRAEFIK_ENABLE=true` and recreate only the OXYTIB service. The public
checks are then:

```bash
curl --fail https://example.org/oxytib/api/v1/health
python validation/verify_public_deployment.py \
  --base-url https://example.org/oxytib
```

## Standalone Caddy deployment

For a dedicated host without Traefik, copy
`deploy/.env.production.example` to the ignored
`deploy/.env.production`, configure the domain and ACME email, and run:

```bash
docker compose \
  --env-file deploy/.env.production \
  -f deploy/compose.production.yaml \
  up --build -d
```

Caddy terminates TLS, applies the same browser security policy, limits request
bodies, and proxies to the private application network. The application-level
compute limit remains active. Standard Caddy does not provide the Traefik rate
limiter, so a provider firewall or upstream rate limiter is recommended before
advertising a high-volume public service.

## Operations

Inspect status and bounded logs with the Compose file used for deployment:

```bash
docker compose --env-file deploy/.env.traefik \
  -f deploy/compose.traefik.yaml ps
docker compose --env-file deploy/.env.traefik \
  -f deploy/compose.traefik.yaml logs --tail=100 model-api
```

Deploy reviewed tags, not moving branches. Record the currently deployed tag
and image digest before every update. After rebuilding, run the remote verifier
and one browser calculation.

Rollback consists of checking out the preceding release tag, rebuilding the
OXYTIB service, and rerunning the verifier. OXYTIB has no database migrations
or persistent scientific state.

Keep private backups of the untracked deployment environment file and the
server's Traefik configuration. The source release itself is preserved by its
Git tag and archived DOI.

## Privacy and monitoring

Docker rotates application logs. Traefik or Caddy access logs may contain
timestamps, paths, response status, and client addresses; they must not record
JSON request bodies. Choose retention according to institutional policy.

Monitor at least container restarts, health-check failures, HTTP 429/503 rates,
CPU, memory, disk use, and proxy TLS renewal. Repeated 429 responses indicate
rate limiting; 503 responses from OXYTIB indicate that the bounded compute
capacity was occupied.
