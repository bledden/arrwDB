# Docker Deployment Guide for arrwDB

This guide covers deploying arrwDB using Docker and Docker Compose for local development, production, and with Temporal workflow integration.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- At least 4GB of available RAM
- Cohere API key (for embedding generation)

## Quick Start

### Development Mode (Simplest)

For local development with hot-reloading:

```bash
# Set your Cohere API key
export COHERE_API_KEY="your_cohere_api_key"

# Start in development mode
docker-compose -f docker-compose.dev.yml up

# Access the API at http://localhost:8000
# API docs at http://localhost:8000/docs
```

**Features:**
- Hot-reloading (code changes automatically refresh)
- Debug logging enabled
- Multi-tenancy disabled (no auth required)
- Rate limiting disabled
- Source code mounted for live editing

### Production Mode

For production deployment with security and monitoring:

```bash
# Set environment variables
export COHERE_API_KEY="your_cohere_api_key"
export GUNICORN_WORKERS=4
export CORS_ORIGINS="https://yourdomain.com"

# Start in production mode
docker-compose -f docker-compose.prod.yml up -d

# With monitoring (Prometheus + Grafana)
docker-compose -f docker-compose.prod.yml --profile monitoring up -d
```

**Features:**
- Gunicorn with multiple workers
- Multi-tenancy enabled (API key auth)
- Rate limiting enabled
- JSON logging for log aggregators
- Resource limits configured
- Health checks enabled
- Optional Prometheus metrics
- Optional Grafana dashboards

**Access:**
- API: http://localhost:8000
- Prometheus: http://localhost:9090 (if monitoring enabled)
- Grafana: http://localhost:3000 (if monitoring enabled, default password: admin)

### Full Stack Mode (with Temporal)

For workflow orchestration with Temporal:

```bash
# Set your Cohere API key
export COHERE_API_KEY="your_cohere_api_key"

# Start all services
docker-compose up -d

# Access services
# - API: http://localhost:8000
# - Temporal UI: http://localhost:8080
```

**Services:**
- arrwDB API (port 8000)
- Temporal Server (port 7233)
- Temporal Worker
- Temporal Web UI (port 8080)
- PostgreSQL (for Temporal persistence)

## Configuration

### Environment Variables

All configurations can be set via environment variables:

#### Server Settings
```bash
HOST=0.0.0.0                    # Bind address
PORT=8000                       # API port
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR
LOG_JSON_FORMAT=false           # true for production
```

#### Multi-Tenancy
```bash
MULTI_TENANCY_ENABLED=false     # Enable API key authentication
TENANTS_DB_PATH=/app/data/tenants.json
```

#### Rate Limiting
```bash
RATE_LIMIT_ENABLED=false        # Enable rate limiting
RATE_LIMIT_PER_MINUTE=60        # Requests per minute
RATE_LIMIT_BURST=10             # Burst allowance
```

#### Embedding Service
```bash
COHERE_API_KEY=your_key         # Required
EMBEDDING_MODEL=embed-english-v3.0
EMBEDDING_DIMENSION=1024
```

#### Performance
```bash
MAX_CHUNK_SIZE=1000
MAX_CHUNKS_PER_DOCUMENT=1000
MAX_CONCURRENT_REQUESTS=100
GUNICORN_WORKERS=4              # Production only
```

#### Event Bus & Job Queue
```bash
EVENT_BUS_ENABLED=true
JOB_QUEUE_ENABLED=true
JOB_QUEUE_MAX_WORKERS=4
JOB_QUEUE_MAX_RETRIES=3
```

### Using .env File

Create a `.env` file in the project root:

```bash
COHERE_API_KEY=your_cohere_api_key_here
GUNICORN_WORKERS=4
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
GRAFANA_PASSWORD=secure_password
```

Docker Compose will automatically load these variables.

## Data Persistence

### Development Mode
Data is stored in `./data` directory (bind mount):
```bash
./data/vectors      # Vector index data
./data/wal          # Write-ahead log
./data/snapshots    # Snapshots
./data/tenants.json # Tenant/API key data
```

### Production Mode
Data is stored in named Docker volumes:
```bash
arrwdb-prod-data    # Application data
arrwdb-prod-logs    # Application logs
```

To backup production data:
```bash
docker run --rm -v arrwdb-prod-data:/data -v $(pwd):/backup ubuntu tar czf /backup/arrwdb-backup.tar.gz /data
```

To restore:
```bash
docker run --rm -v arrwdb-prod-data:/data -v $(pwd):/backup ubuntu tar xzf /backup/arrwdb-backup.tar.gz -C /
```

## Health Checks

All configurations include health checks:

```bash
# Check API health
curl http://localhost:8000/health

# Check container health
docker ps

# View health check logs
docker inspect arrwdb-prod | jq '.[0].State.Health'
```

## Logs

### Development Mode
```bash
# Follow logs
docker-compose -f docker-compose.dev.yml logs -f

# View specific service
docker-compose -f docker-compose.dev.yml logs -f arrwdb-dev
```

### Production Mode
```bash
# Follow logs
docker-compose -f docker-compose.prod.yml logs -f

# Logs are also written to volume
docker run --rm -v arrwdb-prod-logs:/logs ubuntu cat /logs/access.log
docker run --rm -v arrwdb-prod-logs:/logs ubuntu cat /logs/error.log
```

## Scaling

### Horizontal Scaling (Production)

Scale API instances:
```bash
docker-compose -f docker-compose.prod.yml up -d --scale arrwdb-prod=3
```

### Vertical Scaling

Adjust resource limits in `docker-compose.prod.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '8.0'
      memory: 16G
```

## Monitoring (Production)

### Enable Prometheus + Grafana

```bash
docker-compose -f docker-compose.prod.yml --profile monitoring up -d
```

### Prometheus Configuration

Create `monitoring/prometheus.yml`:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'arrwdb'
    static_configs:
      - targets: ['arrwdb-prod:8000']
```

### Grafana Dashboards

1. Access Grafana at http://localhost:3000
2. Login with admin/admin (or your GRAFANA_PASSWORD)
3. Add Prometheus data source: http://prometheus:9090
4. Import dashboard or create custom metrics

## Troubleshooting

### Container won't start

Check logs:
```bash
docker logs arrwdb-prod
```

Common issues:
- Missing COHERE_API_KEY
- Port 8000 already in use
- Insufficient memory

### Performance Issues

Check resource usage:
```bash
docker stats arrwdb-prod
```

Adjust GUNICORN_WORKERS or resource limits as needed.

### Data Loss

Ensure volumes are properly configured:
```bash
docker volume ls
docker volume inspect arrwdb-prod-data
```

## Stopping and Cleanup

### Stop Services
```bash
# Development
docker-compose -f docker-compose.dev.yml down

# Production
docker-compose -f docker-compose.prod.yml down

# Full stack
docker-compose down
```

### Remove Volumes (CAUTION: Data Loss)
```bash
# Development (safe, data in ./data)
docker-compose -f docker-compose.dev.yml down

# Production (removes all data)
docker-compose -f docker-compose.prod.yml down -v

# Remove specific volume
docker volume rm arrwdb-prod-data
```

## Best Practices

### Development
1. Use `docker-compose.dev.yml` for local work
2. Mount source code for hot-reloading
3. Use DEBUG log level
4. Disable authentication for easier testing

### Production
1. Always use `docker-compose.prod.yml`
2. Set strong GRAFANA_PASSWORD
3. Configure CORS_ORIGINS to your domain
4. Enable monitoring profile
5. Set up regular backups
6. Use JSON logging with log aggregator
7. Configure resource limits
8. Enable multi-tenancy and rate limiting
9. Run behind reverse proxy (nginx/traefik)
10. Use HTTPS with valid certificates

## Security Considerations

1. **Never commit .env files to git**
2. **Rotate API keys regularly**
3. **Use Docker secrets in production** (Swarm/Kubernetes)
4. **Run containers as non-root user** (future enhancement)
5. **Keep Docker images updated**
6. **Scan images for vulnerabilities**
7. **Limit network exposure** (use reverse proxy)
8. **Enable audit logging** in production

## Next Steps

- See [KUBERNETES_DEPLOYMENT.md](./KUBERNETES_DEPLOYMENT.md) for Kubernetes deployment
- See [CI_CD.md](./CI_CD.md) for CI/CD pipeline setup
- See [MONITORING.md](./MONITORING.md) for advanced monitoring setup
