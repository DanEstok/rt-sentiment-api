# Deployment Guide

## Overview

This guide covers deploying the RT Sentiment API in various environments, from local development to production Kubernetes clusters. The system supports multiple deployment strategies with focus on high availability and performance.

## Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/your-org/rt-sentiment-api.git
cd rt-sentiment-api

# Install dependencies
pip install -r requirements.txt

# Start API server
python src/app/main.py
```

### Docker Deployment

```bash
# Build images
docker build -f docker/Dockerfile.inference -t sentiment-api .

# Run with Docker Compose
docker-compose up -d
```

### Production Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f deploy/k8s/
```

## Environment Setup

### Development Environment

#### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Git
- 8GB+ RAM (for model loading)

#### Setup Steps

```bash
# 1. Clone repository
git clone https://github.com/your-org/rt-sentiment-api.git
cd rt-sentiment-api

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
export MODEL_PATH=./models
export PYTHONPATH=.

# 5. Download or train model
python src/training/train.py --dataset twitter --epochs 1 --export

# 6. Start development server
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Development Configuration

```bash
# .env file for development
MODEL_PATH=./models
LOG_LEVEL=DEBUG
BATCH_SIZE=4
BATCH_TIMEOUT=0.01
MAX_BATCH_SIZE=16
```

### Staging Environment

#### Docker Compose Setup

```yaml
# docker-compose.staging.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.inference
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models
      - LOG_LEVEL=INFO
      - BATCH_SIZE=8
      - MAX_BATCH_SIZE=32
    volumes:
      - ./models:/app/models:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources

volumes:
  grafana-storage:
```

#### Staging Deployment

```bash
# Deploy to staging
docker-compose -f docker-compose.staging.yml up -d

# Verify deployment
curl http://localhost:8000/healthz

# Run integration tests
python -m pytest tests/ -v
```

### Production Environment

#### Kubernetes Deployment

##### Namespace and ConfigMap

```yaml
# deploy/k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: sentiment-api
  labels:
    name: sentiment-api

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: sentiment-api-config
  namespace: sentiment-api
data:
  MODEL_PATH: "/app/models"
  LOG_LEVEL: "INFO"
  BATCH_SIZE: "16"
  MAX_BATCH_SIZE: "64"
  BATCH_TIMEOUT: "0.01"
```

##### Deployment

```yaml
# deploy/k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-api
  namespace: sentiment-api
  labels:
    app: sentiment-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentiment-api
  template:
    metadata:
      labels:
        app: sentiment-api
    spec:
      containers:
      - name: sentiment-api
        image: ghcr.io/your-org/rt-sentiment-api-inference:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: sentiment-api-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /healthz
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
      imagePullSecrets:
      - name: ghcr-secret
```

##### Service and Ingress

```yaml
# deploy/k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: sentiment-api-service
  namespace: sentiment-api
  labels:
    app: sentiment-api
spec:
  selector:
    app: sentiment-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sentiment-api-ingress
  namespace: sentiment-api
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.rt-sentiment.com
    secretName: sentiment-api-tls
  rules:
  - host: api.rt-sentiment.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: sentiment-api-service
            port:
              number: 80
```

##### Horizontal Pod Autoscaler

```yaml
# deploy/k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sentiment-api-hpa
  namespace: sentiment-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sentiment-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

## Helm Deployment

### Helm Chart Structure

```
deploy/helm/sentiment-api/
├── Chart.yaml
├── values.yaml
├── values-prod.yaml
├── values-staging.yaml
└── templates/
    ├── deployment.yaml
    ├── service.yaml
    ├── ingress.yaml
    ├── configmap.yaml
    ├── hpa.yaml
    ├── pdb.yaml
    └── serviceaccount.yaml
```

### Chart.yaml

```yaml
apiVersion: v2
name: sentiment-api
description: RT Sentiment Analysis API
type: application
version: 1.0.0
appVersion: "1.0.0"
keywords:
  - sentiment
  - api
  - ml
  - nlp
home: https://github.com/your-org/rt-sentiment-api
sources:
  - https://github.com/your-org/rt-sentiment-api
maintainers:
  - name: Your Team
    email: team@yourorg.com
```

### values.yaml

```yaml
# Default values for sentiment-api
replicaCount: 3

image:
  repository: ghcr.io/your-org/rt-sentiment-api-inference
  pullPolicy: IfNotPresent
  tag: "latest"

imagePullSecrets:
  - name: ghcr-secret

nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

podAnnotations: {}

podSecurityContext:
  fsGroup: 2000

securityContext:
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: true
  runAsNonRoot: true
  runAsUser: 1000

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: "nginx"
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api.rt-sentiment.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: sentiment-api-tls
      hosts:
        - api.rt-sentiment.com

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}

tolerations: []

affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app.kubernetes.io/name
            operator: In
            values:
            - sentiment-api
        topologyKey: kubernetes.io/hostname

config:
  modelPath: "/app/models"
  logLevel: "INFO"
  batchSize: 16
  maxBatchSize: 64
  batchTimeout: 0.01

persistence:
  enabled: true
  storageClass: "fast-ssd"
  accessMode: ReadOnlyMany
  size: 10Gi

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s
    path: /metrics

podDisruptionBudget:
  enabled: true
  minAvailable: 2
```

### Production Values

```yaml
# values-prod.yaml
replicaCount: 5

image:
  tag: "v1.0.0"  # Use specific version in production

resources:
  limits:
    cpu: 4000m
    memory: 8Gi
  requests:
    cpu: 2000m
    memory: 4Gi

autoscaling:
  minReplicas: 5
  maxReplicas: 50

config:
  logLevel: "WARNING"
  batchSize: 32
  maxBatchSize: 128

persistence:
  storageClass: "premium-ssd"
  size: 20Gi

monitoring:
  serviceMonitor:
    interval: 15s

podDisruptionBudget:
  minAvailable: 3
```

### Helm Deployment Commands

```bash
# Add Helm repository (if using external chart)
helm repo add sentiment-api https://charts.yourorg.com
helm repo update

# Install in staging
helm install sentiment-api-staging ./deploy/helm/sentiment-api \
  --namespace sentiment-api-staging \
  --create-namespace \
  --values ./deploy/helm/sentiment-api/values-staging.yaml

# Install in production
helm install sentiment-api-prod ./deploy/helm/sentiment-api \
  --namespace sentiment-api-prod \
  --create-namespace \
  --values ./deploy/helm/sentiment-api/values-prod.yaml

# Upgrade deployment
helm upgrade sentiment-api-prod ./deploy/helm/sentiment-api \
  --namespace sentiment-api-prod \
  --values ./deploy/helm/sentiment-api/values-prod.yaml

# Rollback if needed
helm rollback sentiment-api-prod 1 --namespace sentiment-api-prod
```

## CI/CD Pipeline

### GitHub Actions Deployment

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}-inference

jobs:
  deploy-production:
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
    
    - name: Deploy with Helm
      run: |
        helm upgrade --install sentiment-api-prod ./deploy/helm/sentiment-api \
          --namespace sentiment-api-prod \
          --create-namespace \
          --values ./deploy/helm/sentiment-api/values-prod.yaml \
          --set image.tag=${{ github.ref_name }} \
          --wait \
          --timeout 10m
    
    - name: Verify deployment
      run: |
        kubectl rollout status deployment/sentiment-api-prod -n sentiment-api-prod
        kubectl get pods -n sentiment-api-prod
    
    - name: Run smoke tests
      run: |
        # Wait for service to be ready
        kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=sentiment-api -n sentiment-api-prod --timeout=300s
        
        # Get service URL
        SERVICE_URL=$(kubectl get ingress sentiment-api-ingress -n sentiment-api-prod -o jsonpath='{.spec.rules[0].host}')
        
        # Run smoke tests
        curl -f https://$SERVICE_URL/healthz
        curl -f -X POST https://$SERVICE_URL/predict -H "Content-Type: application/json" -d '{"text": "Test message"}'
```

### Canary Deployment

```yaml
# .github/workflows/canary-deploy.yml
name: Canary Deployment

on:
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to deploy'
        required: true
      traffic_percentage:
        description: 'Percentage of traffic to route to canary'
        required: true
        default: '10'

jobs:
  canary-deploy:
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Deploy canary
      run: |
        # Deploy canary version
        helm upgrade --install sentiment-api-canary ./deploy/helm/sentiment-api \
          --namespace sentiment-api-prod \
          --values ./deploy/helm/sentiment-api/values-prod.yaml \
          --set image.tag=${{ github.event.inputs.version }} \
          --set nameOverride=sentiment-api-canary \
          --set replicaCount=2 \
          --wait
    
    - name: Configure traffic split
      run: |
        # Update ingress to split traffic
        kubectl patch ingress sentiment-api-ingress -n sentiment-api-prod --type='json' \
          -p='[{
            "op": "add",
            "path": "/metadata/annotations/nginx.ingress.kubernetes.io~1canary",
            "value": "true"
          }, {
            "op": "add", 
            "path": "/metadata/annotations/nginx.ingress.kubernetes.io~1canary-weight",
            "value": "${{ github.event.inputs.traffic_percentage }}"
          }]'
    
    - name: Monitor canary
      run: |
        # Monitor canary for 10 minutes
        sleep 600
        
        # Check error rates and latency
        ERROR_RATE=$(kubectl exec -n monitoring prometheus-0 -- \
          promtool query instant 'rate(http_requests_total{job="sentiment-api-canary",code!~"2.."}[5m]) / rate(http_requests_total{job="sentiment-api-canary"}[5m])')
        
        if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
          echo "High error rate detected, rolling back canary"
          exit 1
        fi
    
    - name: Promote or rollback
      run: |
        # If monitoring passed, promote canary
        helm upgrade sentiment-api-prod ./deploy/helm/sentiment-api \
          --namespace sentiment-api-prod \
          --values ./deploy/helm/sentiment-api/values-prod.yaml \
          --set image.tag=${{ github.event.inputs.version }} \
          --wait
        
        # Remove canary deployment
        helm uninstall sentiment-api-canary --namespace sentiment-api-prod
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'sentiment-api'
    static_configs:
      - targets: ['sentiment-api-service:80']
    metrics_path: '/metrics'
    scrape_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "RT Sentiment API",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(sentiment_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(sentiment_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(sentiment_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P95"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(sentiment_requests_total{code!~\"2..\"}[5m]) / rate(sentiment_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      }
    ]
  }
}
```

### Alert Rules

```yaml
# monitoring/alert_rules.yml
groups:
  - name: sentiment-api.rules
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(sentiment_request_duration_seconds_bucket[5m])) > 0.12
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency"
          description: "95th percentile latency is {{ $value }}s"
      
      - alert: HighErrorRate
        expr: rate(sentiment_requests_total{code!~"2.."}[5m]) / rate(sentiment_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate"
          description: "Error rate is {{ $value | humanizePercentage }}"
      
      - alert: PodCrashLooping
        expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Pod is crash looping"
          description: "Pod {{ $labels.pod }} is restarting frequently"
```

## Security Considerations

### Container Security

```dockerfile
# Security-hardened Dockerfile
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy application
COPY --chown=appuser:appuser . /app
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER appuser

# Run application
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Security

```yaml
# Security policies
apiVersion: v1
kind: Pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
  containers:
  - name: sentiment-api
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
    resources:
      limits:
        memory: "4Gi"
        cpu: "2"
      requests:
        memory: "2Gi"
        cpu: "1"
```

### Network Policies

```yaml
# Network policy to restrict traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: sentiment-api-netpol
  namespace: sentiment-api
spec:
  podSelector:
    matchLabels:
      app: sentiment-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
```

## Troubleshooting

### Common Issues

#### Pod Startup Issues

```bash
# Check pod status
kubectl get pods -n sentiment-api

# Check pod logs
kubectl logs -f deployment/sentiment-api -n sentiment-api

# Describe pod for events
kubectl describe pod <pod-name> -n sentiment-api

# Check resource usage
kubectl top pods -n sentiment-api
```

#### Performance Issues

```bash
# Check HPA status
kubectl get hpa -n sentiment-api

# Check resource limits
kubectl describe deployment sentiment-api -n sentiment-api

# Check node resources
kubectl top nodes

# Check service endpoints
kubectl get endpoints -n sentiment-api
```

#### Network Issues

```bash
# Test service connectivity
kubectl run test-pod --image=curlimages/curl -it --rm -- sh
curl http://sentiment-api-service.sentiment-api.svc.cluster.local/healthz

# Check ingress
kubectl get ingress -n sentiment-api
kubectl describe ingress sentiment-api-ingress -n sentiment-api

# Check DNS resolution
nslookup sentiment-api-service.sentiment-api.svc.cluster.local
```

### Debugging Commands

```bash
# Port forward for local testing
kubectl port-forward service/sentiment-api-service 8000:80 -n sentiment-api

# Execute commands in pod
kubectl exec -it deployment/sentiment-api -n sentiment-api -- bash

# Check configuration
kubectl get configmap sentiment-api-config -n sentiment-api -o yaml

# Check secrets
kubectl get secrets -n sentiment-api

# View recent events
kubectl get events -n sentiment-api --sort-by='.lastTimestamp'
```

## Best Practices

### Deployment Best Practices

1. **Use specific image tags** in production
2. **Implement health checks** for all services
3. **Set resource limits** to prevent resource exhaustion
4. **Use pod disruption budgets** for high availability
5. **Implement proper monitoring** and alerting

### Security Best Practices

1. **Run containers as non-root** users
2. **Use read-only root filesystems** when possible
3. **Implement network policies** to restrict traffic
4. **Scan images** for vulnerabilities regularly
5. **Use secrets management** for sensitive data

### Performance Best Practices

1. **Use horizontal pod autoscaling** for dynamic scaling
2. **Implement proper resource requests/limits**
3. **Use node affinity** for optimal placement
4. **Monitor and optimize** based on metrics
5. **Implement caching** where appropriate