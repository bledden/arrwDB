# Security & Code Quality Enhancements Roadmap

**Status**: In Progress
**Goal**: Achieve Auth0/AWS Cognito-level authentication and world-class security

## Completed ✅

### 1. Scopes/Permissions System & Audit Logging
**Files Created**:
- `app/auth/permissions.py` - OAuth 2.0-style scopes and RBAC
- `app/auth/audit.py` - Comprehensive audit logging

**Features**:
- 15+ granular permission scopes (libraries:read, documents:write, etc.)
- 5 predefined roles (Viewer, User, Developer, Admin, Superuser)
- Hierarchical admin scopes (admin:all grants everything)
- Structured JSON audit logging for SIEM integration
- 20+ audit event types (auth, authz, API keys, security events)
- Compliance-ready (GDPR, SOC 2, AWS CloudTrail-style)

**Commit**: `ca4fdf6`

---

## Remaining Enhancements

### 2. CORS Configuration ⏳
**Priority**: HIGH
**Effort**: 1-2 hours

**Implementation**:
```python
# app/middleware/cors.py
from fastapi.middleware.cors import CORSMiddleware

CORS_CONFIG = {
    "allow_origins": os.getenv("CORS_ORIGINS", "*").split(","),
    "allow_credentials": True,
    "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH"],
    "allow_headers": ["*"],
    "expose_headers": ["X-Total-Count", "X-Request-ID"],
    "max_age": 3600,  # Cache preflight for 1 hour
}
```

**Config additions** (`app/config.py`):
```python
CORS_ENABLED: bool = True
CORS_ORIGINS: str = "*"  # Comma-separated list
CORS_ALLOW_CREDENTIALS: bool = True
```

---

### 3. Request Size Limits & DoS Protection ⏳
**Priority**: HIGH
**Effort**: 2-3 hours

**Implementation**:
```python
# In main.py
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS.split(",")
)

# Add body size limit
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method in ["POST", "PUT", "PATCH"]:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.MAX_REQUEST_SIZE:
            return JSONResponse(
                status_code=413,
                content={"error": "Request too large"}
            )
    return await call_next(request)
```

**Config additions**:
```python
MAX_REQUEST_SIZE: int = 100 * 1024 * 1024  # 100MB
ALLOWED_HOSTS: str = "*"  # Comma-separated
```

---

### 4. Security Headers ⏳
**Priority**: HIGH
**Effort**: 1 hour

**Implementation**:
```python
# app/middleware/security_headers.py
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return response
```

---

### 5. HTTPS/TLS Configuration Guide ⏳
**Priority**: MEDIUM
**Effort**: 2 hours

**Create**: `docs/deployment/HTTPS_SETUP.md`

**Contents**:
- nginx reverse proxy configuration
- Caddy configuration (auto-SSL)
- Let's Encrypt setup
- Certificate renewal automation
- HSTS header configuration
- TLS 1.3 best practices

---

### 6. Secrets Manager Integration ⏳
**Priority**: MEDIUM
**Effort**: 3 hours

**Create**: `docs/deployment/SECRETS_MANAGEMENT.md`

**Implementation**:
```python
# app/secrets/manager.py
class SecretsManager:
    """Unified interface for secrets management."""

    def get_secret(self, key: str) -> str:
        # Try AWS Secrets Manager
        # Fall back to env vars
        # Support HashiCorp Vault
        pass
```

**Integrations**:
- AWS Secrets Manager
- HashiCorp Vault
- Azure Key Vault
- Google Secret Manager

---

### 7. Pre-commit Hooks ⏳
**Priority**: MEDIUM
**Effort**: 1 hour

**Create**: `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.11.0
    hooks:
      - id: black

  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

**Setup script**:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

---

### 8. Increase Test Coverage ⏳
**Priority**: MEDIUM
**Effort**: 5-8 hours

**Implementation**:
```bash
# Add pytest-cov configuration
# pytest.ini
[pytest]
addopts = --cov=app --cov=infrastructure --cov-report=html --cov-report=term-missing --cov-fail-under=80
```

**Focus areas**:
- Auth middleware tests
- Permission/scope validation tests
- Audit logging tests
- Security header tests
- Rate limiting tests
- Property-based tests with `hypothesis`

---

### 9. API Versioning Strategy ⏳
**Priority**: LOW
**Effort**: 2 hours

**Create**: `docs/api/VERSIONING_STRATEGY.md`

**Contents**:
- Semantic versioning (v1, v2, v3)
- Deprecation timeline (6 months notice)
- Breaking vs non-breaking changes
- Migration guides
- Version negotiation via Accept header
- Sunset header for deprecated endpoints

---

### 10. Monitoring & Observability ⏳
**Priority**: MEDIUM
**Effort**: 4-6 hours

**Implementation**:
```python
# app/observability/tracing.py
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup distributed tracing
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
```

**Integrations**:
- OpenTelemetry for distributed tracing
- Sentry for error tracking
- Prometheus metrics (already have instrumentator)
- Grafana dashboards

---

### 11. Performance Profiling ⏳
**Priority**: LOW
**Effort**: 3 hours

**Create**: `docs/performance/PROFILING_GUIDE.md`

**Tools**:
- py-spy for CPU profiling
- memory_profiler for memory analysis
- locust for load testing
- K6 for API performance testing

**Implementation**:
```python
# app/middleware/profiling.py
from pyinstrument import Profiler

@app.middleware("http")
async def profile_request(request: Request, call_next):
    if settings.ENABLE_PROFILING:
        profiler = Profiler()
        profiler.start()
        response = await call_next(request)
        profiler.stop()
        # Log slow requests
        if profiler.duration > 1.0:  # > 1 second
            logger.warning(f"Slow request: {profiler.output_text()}")
        return response
    return await call_next(request)
```

---

## Additional Auth Enhancements (For Future)

### JWT Support
- Add `PyJWT` dependency
- Implement token generation/validation
- Add refresh token rotation
- Short-lived access tokens (15 min)
- Long-lived refresh tokens (30 days)

### IP Whitelisting
- Add `allowed_ips` field to Tenant model
- CIDR range support
- Geo-blocking capabilities

### Multi-Key Support
- Allow multiple active API keys per tenant
- Graceful key rotation without downtime
- Key tagging (production, staging, dev)

### Enhanced Hashing
- Replace SHA-256 with Argon2 for API key hashing
- Add `argon2-cffi` dependency
- Configurable memory/time cost

---

## Timeline Estimate

**Phase 1 (Week 1)**: High Priority Security
- CORS Configuration (2h)
- Request Size Limits (3h)
- Security Headers (1h)
- **Total**: 6 hours

**Phase 2 (Week 2)**: Infrastructure & Testing
- Pre-commit Hooks (1h)
- Test Coverage to 80% (8h)
- **Total**: 9 hours

**Phase 3 (Week 3)**: Deployment & Monitoring
- HTTPS/TLS Guide (2h)
- Secrets Manager Integration (3h)
- Monitoring Setup (6h)
- **Total**: 11 hours

**Phase 4 (Week 4)**: Documentation & Polish
- API Versioning Strategy (2h)
- Performance Profiling Guide (3h)
- **Total**: 5 hours

**Grand Total**: ~31 hours of focused work

---

## Success Criteria

- [ ] OWASP Top 10 compliance
- [ ] SOC 2 audit-ready
- [ ] 80%+ test coverage
- [ ] All security headers present
- [ ] CORS properly configured
- [ ] Request size limits enforced
- [ ] Audit logs for all security events
- [ ] Pre-commit hooks preventing bad code
- [ ] Distributed tracing operational
- [ ] Performance profiling enabled

---

## References

- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [Auth0 Best Practices](https://auth0.com/docs/best-practices)
- [AWS Well-Architected Framework - Security Pillar](https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/welcome.html)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CWE Top 25 Most Dangerous Software Weaknesses](https://cwe.mitre.org/top25/)
