# arrwDB Helm Chart

Helm chart for deploying arrwDB - a production-grade vector database with 9 novel features.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2+
- PV provisioner support (if persistence is enabled)

## Installation

```bash
# Add the repository (when published)
helm repo add arrwdb https://bledden.github.io/arrwDB/helm

# Install
helm install my-arrwdb arrwdb/arrwdb

# Or install from local chart
helm install my-arrwdb ./deploy/helm/arrwdb
```

## Quick Start

### Basic Installation

```bash
helm install arrwdb ./deploy/helm/arrwdb
```

### With Custom Values

```bash
helm install arrwdb ./deploy/helm/arrwdb \
  --set replicaCount=3 \
  --set resources.requests.memory=2Gi \
  --set persistence.size=50Gi
```

### With Ingress

```bash
helm install arrwdb ./deploy/helm/arrwdb \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host=arrwdb.example.com \
  --set ingress.hosts[0].paths[0].path=/ \
  --set ingress.hosts[0].paths[0].pathType=Prefix
```

### With Authentication

```bash
helm install arrwdb ./deploy/helm/arrwdb \
  --set arrwdb.auth.enabled=true \
  --set arrwdb.auth.apiKey=your-secure-api-key
```

### With Autoscaling

```bash
helm install arrwdb ./deploy/helm/arrwdb \
  --set autoscaling.enabled=true \
  --set autoscaling.minReplicas=2 \
  --set autoscaling.maxReplicas=10
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Image repository | `ghcr.io/bledden/arrwdb` |
| `image.tag` | Image tag | Chart appVersion |
| `service.type` | Service type | `ClusterIP` |
| `service.port` | Service port | `8000` |
| `ingress.enabled` | Enable ingress | `false` |
| `resources.requests.cpu` | CPU request | `500m` |
| `resources.requests.memory` | Memory request | `1Gi` |
| `resources.limits.cpu` | CPU limit | `2000m` |
| `resources.limits.memory` | Memory limit | `4Gi` |
| `persistence.enabled` | Enable persistence | `true` |
| `persistence.size` | PVC size | `10Gi` |
| `autoscaling.enabled` | Enable HPA | `false` |

### arrwDB Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `arrwdb.host` | Bind host | `0.0.0.0` |
| `arrwdb.port` | Server port | `8000` |
| `arrwdb.workers` | Worker count | `4` |
| `arrwdb.logLevel` | Log level | `info` |
| `arrwdb.auth.enabled` | Enable auth | `false` |
| `arrwdb.auth.apiKey` | API key | `""` |
| `arrwdb.embedding.model` | Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `arrwdb.embedding.device` | Device (cpu/cuda/mps) | `cpu` |
| `arrwdb.index.defaultType` | Default index | `hnsw` |

### Novel Features

| Parameter | Description | Default |
|-----------|-------------|---------|
| `arrwdb.features.temperatureSearch.enabled` | Temperature search | `true` |
| `arrwdb.features.indexOracle.enabled` | Index oracle | `true` |
| `arrwdb.features.embeddingHealth.enabled` | Embedding health | `true` |
| `arrwdb.features.searchReplay.enabled` | Search replay (debug) | `false` |

## Upgrading

```bash
helm upgrade arrwdb ./deploy/helm/arrwdb
```

## Uninstalling

```bash
helm uninstall arrwdb
```

Note: This will not delete the PVC. To delete data:

```bash
kubectl delete pvc arrwdb
```

## Production Recommendations

1. **Enable persistence** with appropriate storage size
2. **Set resource limits** based on your workload
3. **Enable autoscaling** for variable load
4. **Configure ingress** with TLS for external access
5. **Enable authentication** for production deployments

Example production values:

```yaml
replicaCount: 3

resources:
  requests:
    cpu: 1000m
    memory: 4Gi
  limits:
    cpu: 4000m
    memory: 16Gi

persistence:
  enabled: true
  size: 100Gi
  storageClass: fast-ssd

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20

arrwdb:
  workers: 8
  auth:
    enabled: true
    existingSecret: arrwdb-credentials
  embedding:
    device: cuda  # If GPU available
```

## License

Apache 2.0
