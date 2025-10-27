# API Versioning Strategy for arrwDB

This document defines arrwDB's approach to API versioning, deprecation, and backward compatibility.

## Table of Contents

- [Versioning Philosophy](#versioning-philosophy)
- [Version Numbering](#version-numbering)
- [URL-Based Versioning](#url-based-versioning)
- [Breaking vs Non-Breaking Changes](#breaking-vs-non-breaking-changes)
- [Deprecation Policy](#deprecation-policy)
- [Migration Guides](#migration-guides)
- [Version Support Timeline](#version-support-timeline)
- [Implementation Guidelines](#implementation-guidelines)

## Versioning Philosophy

arrwDB follows these principles:

1. **Stability First**: API stability is paramount for production users
2. **Semantic Versioning**: Version numbers convey meaning about changes
3. **Backward Compatibility**: Maintain compatibility within major versions
4. **Clear Communication**: Announce changes well in advance
5. **Migration Support**: Provide tools and guides for upgrading

## Version Numbering

arrwDB uses **Semantic Versioning 2.0.0** (semver):

```
MAJOR.MINOR.PATCH
```

### Version Components

- **MAJOR** (e.g., v1 → v2): Breaking changes that require code modifications
- **MINOR** (e.g., v1.0 → v1.1): New features, backward compatible
- **PATCH** (e.g., v1.0.0 → v1.0.1): Bug fixes, backward compatible

### Examples

```
v1.0.0 → v1.0.1   # Bug fix (safe to upgrade)
v1.0.1 → v1.1.0   # New feature (safe to upgrade)
v1.1.0 → v2.0.0   # Breaking change (requires migration)
```

## URL-Based Versioning

arrwDB uses URL path versioning for clarity and simplicity.

### Current API Structure

```
https://api.yourdomain.com/v1/libraries
https://api.yourdomain.com/v1/documents
https://api.yourdomain.com/v1/search
```

### Version in URL Path

**Recommended Approach:**

```
# Version 1
POST /v1/libraries
GET /v1/libraries/{library_id}
POST /v1/documents

# Version 2 (with breaking changes)
POST /v2/libraries
GET /v2/libraries/{library_id}
POST /v2/documents
```

### Why URL Versioning?

✅ **Pros:**
- Explicit and visible
- Easy to test different versions
- Simple to cache per version
- Works with all HTTP clients
- No custom headers needed

❌ **Alternatives NOT Used:**

```
# Header versioning (NOT used)
X-API-Version: 2

# Accept header versioning (NOT used)
Accept: application/vnd.arrwdb.v2+json

# Query parameter versioning (NOT used)
/libraries?version=2
```

## Breaking vs Non-Breaking Changes

### Breaking Changes (Require Major Version Bump)

These changes require a new major version (e.g., v1 → v2):

#### 1. Removing Endpoints

```diff
# v1
- DELETE /v1/libraries/{id}

# v2 (BREAKING: endpoint removed)
# Endpoint no longer exists
```

#### 2. Removing Request Fields

```diff
# v1
POST /v1/documents
{
  "title": "Document",
  "text": "Content",
  "metadata": {}  # <- Field removed in v2
}

# v2
POST /v2/documents
{
  "title": "Document",
  "text": "Content"
  # metadata field removed (BREAKING)
}
```

#### 3. Removing Response Fields

```diff
# v1
GET /v1/documents/{id}
{
  "id": "doc123",
  "title": "Document",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-02T00:00:00Z"  # <- Removed in v2
}

# v2
GET /v2/documents/{id}
{
  "id": "doc123",
  "title": "Document",
  "created_at": "2024-01-01T00:00:00Z"
  # updated_at removed (BREAKING)
}
```

#### 4. Changing Field Types

```diff
# v1
{
  "chunk_count": 42  # integer
}

# v2
{
  "chunk_count": "42"  # string (BREAKING: type changed)
}
```

#### 5. Changing Field Semantics

```diff
# v1
{
  "distance": 0.5  # Euclidean distance
}

# v2
{
  "distance": 0.5  # Cosine distance (BREAKING: meaning changed)
}
```

#### 6. Changing Error Response Format

```diff
# v1
{
  "error": "Library not found"
}

# v2
{
  "errors": [  # BREAKING: structure changed
    {"code": "NOT_FOUND", "message": "Library not found"}
  ]
}
```

#### 7. Making Optional Fields Required

```diff
# v1
POST /v1/documents
{
  "title": "Document",
  "metadata": {}  # optional
}

# v2
POST /v2/documents
{
  "title": "Document",
  "metadata": {}  # now required (BREAKING)
}
```

#### 8. Changing Authentication/Authorization

```diff
# v1: API key in header
X-API-Key: sk_...

# v2: Bearer token required (BREAKING)
Authorization: Bearer eyJ...
```

### Non-Breaking Changes (Safe for Minor/Patch Versions)

These changes can be made without bumping major version:

#### 1. Adding Optional Fields

```diff
# v1.0.0
POST /v1/documents
{
  "title": "Document",
  "text": "Content"
}

# v1.1.0 (NON-BREAKING: new optional field)
POST /v1/documents
{
  "title": "Document",
  "text": "Content",
  "tags": ["optional", "new"]  # New optional field
}
```

#### 2. Adding Response Fields

```diff
# v1.0.0
{
  "id": "doc123",
  "title": "Document"
}

# v1.1.0 (NON-BREAKING: clients ignore unknown fields)
{
  "id": "doc123",
  "title": "Document",
  "vector_count": 42  # New field added
}
```

#### 3. Adding New Endpoints

```diff
# v1.0.0
POST /v1/libraries
GET /v1/libraries

# v1.1.0 (NON-BREAKING: new endpoint)
POST /v1/libraries
GET /v1/libraries
+ GET /v1/libraries/{id}/statistics  # New endpoint
```

#### 4. Making Required Fields Optional

```diff
# v1.0.0
{
  "title": "Document",  # required
  "text": "Content"     # required
}

# v1.1.0 (NON-BREAKING: more permissive)
{
  "title": "Document",  # still works
  "text": null          # now optional
}
```

#### 5. Adding Enum Values

```diff
# v1.0.0
{
  "index_type": "hnsw" | "brute_force"
}

# v1.1.0 (NON-BREAKING if clients handle unknown values)
{
  "index_type": "hnsw" | "brute_force" | "ivf"  # New type added
}
```

**Note**: Only non-breaking if clients don't validate against a closed enum.

#### 6. Bug Fixes

```diff
# v1.0.0 (bug)
{
  "distance": -0.5  # Should never be negative
}

# v1.0.1 (PATCH: bug fix)
{
  "distance": 0.5  # Fixed
}
```

## Deprecation Policy

### Deprecation Timeline

arrwDB follows a **6-month minimum deprecation period**:

```
Month 0: Feature deprecated, warnings added
Month 3: Deprecation notice in release notes
Month 6: Feature removed in next major version
```

### Deprecation Process

#### 1. Announce Deprecation

Add deprecation notice to API responses:

```http
HTTP/1.1 200 OK
Warning: 299 - "Endpoint /v1/documents/batch deprecated. Use /v1/documents/bulk instead. Will be removed in v2.0.0 (2024-06-01)"
X-API-Deprecation: true
X-API-Sunset: 2024-06-01
```

#### 2. Update Documentation

```markdown
## POST /v1/documents/batch

> **⚠️ DEPRECATED**: This endpoint is deprecated as of v1.5.0 and will be removed in v2.0.0 (June 2024).
>
> **Migration**: Use [POST /v1/documents/bulk](#bulk-endpoint) instead.
```

#### 3. Add Logging

```python
@app.post("/v1/documents/batch")
async def batch_documents(request: Request):
    logger.warning(
        "Deprecated endpoint /v1/documents/batch called",
        extra={
            "tenant_id": request.state.tenant_id,
            "endpoint": "/v1/documents/batch",
            "deprecation_date": "2024-01-01",
            "removal_date": "2024-06-01",
            "alternative": "/v1/documents/bulk",
        }
    )
    # ... existing logic
```

#### 4. Notify Users

- **Release notes**: List all deprecations
- **Blog post**: Explain rationale and migration path
- **Email**: Contact users currently using deprecated features
- **In-app notification**: Show warnings in dashboard

#### 5. Monitor Usage

```python
from prometheus_client import Counter

deprecated_endpoint_calls = Counter(
    'arrwdb_deprecated_endpoint_calls_total',
    'Calls to deprecated endpoints',
    ['endpoint', 'version']
)

@app.post("/v1/documents/batch")
async def batch_documents():
    deprecated_endpoint_calls.labels(
        endpoint="/v1/documents/batch",
        version="v1"
    ).inc()
```

### Deprecation Headers

Standard HTTP headers for deprecation:

```http
Warning: 299 - "Deprecated endpoint. See https://docs.arrwdb.com/migrations/v2"
X-API-Deprecation: true
X-API-Sunset: 2024-06-01
Link: <https://docs.arrwdb.com/migrations/v2>; rel="deprecation"
```

## Migration Guides

### Migration Guide Template

Create a migration guide for each major version:

```markdown
# Migration Guide: v1 → v2

## Overview
- **Release Date**: 2024-06-01
- **Deprecation Period**: 6 months (2024-01-01 to 2024-06-01)
- **Breaking Changes**: 5
- **New Features**: 12

## Breaking Changes

### 1. Authentication Changes

**What Changed**: API key authentication replaced with OAuth 2.0

**v1 (Deprecated)**:
\```http
POST /v1/documents
X-API-Key: sk_your_key_here
\```

**v2 (New)**:
\```http
POST /v2/documents
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
\```

**Migration Steps**:
1. Obtain OAuth credentials from dashboard
2. Implement OAuth flow in your client
3. Update all API calls to use Bearer token
4. Test in staging environment
5. Deploy to production

**Code Example**:
\```python
# v1 (old)
import requests

response = requests.post(
    "https://api.arrwdb.com/v1/documents",
    headers={"X-API-Key": "sk_your_key"},
    json={"title": "Doc"}
)

# v2 (new)
response = requests.post(
    "https://api.arrwdb.com/v2/documents",
    headers={"Authorization": f"Bearer {access_token}"},
    json={"title": "Doc"}
)
\```

### 2. Response Format Changes

**What Changed**: Error responses now use structured format

**v1 (Deprecated)**:
\```json
{
  "error": "Library not found"
}
\```

**v2 (New)**:
\```json
{
  "errors": [
    {
      "code": "LIBRARY_NOT_FOUND",
      "message": "Library with ID 'lib123' not found",
      "field": "library_id"
    }
  ]
}
\```

**Migration Steps**:
1. Update error handling code to parse new format
2. Use error codes for programmatic handling
3. Display user-friendly messages from `message` field

## New Features

### Metadata Filtering

v2 introduces advanced metadata filtering...

## Testing Your Migration

\```bash
# Run migration tests
pytest tests/migration/test_v1_to_v2.py

# Validate v2 compatibility
arrwdb-cli validate-v2-migration
\```

## Rollback Plan

If you encounter issues after migration:

1. Revert to v1 endpoints (supported until 2024-12-01)
2. Report issues: support@arrwdb.com
3. Check status page: status.arrwdb.com

## Support

- **Documentation**: https://docs.arrwdb.com/v2
- **Migration Support**: migrations@arrwdb.com
- **Office Hours**: Tuesdays 2-4pm PT
\```

## Version Support Timeline

### Support Lifecycle

```
┌─────────────┬──────────────┬──────────────┬──────────────┐
│   Release   │    Active    │ Maintenance  │  End of Life │
│             │  Support     │     Only     │              │
├─────────────┼──────────────┼──────────────┼──────────────┤
│ v1.0.0      │ 12 months    │  6 months    │   After v2   │
│ v2.0.0      │ 12 months    │  6 months    │   TBD        │
└─────────────┴──────────────┴──────────────┴──────────────┘
```

### Support Phases

#### 1. Active Support (12 months)

- ✅ New features added
- ✅ Bug fixes
- ✅ Security patches
- ✅ Performance improvements
- ✅ Full documentation
- ✅ Priority support

#### 2. Maintenance Only (6 months)

- ❌ No new features
- ✅ Critical bug fixes only
- ✅ Security patches
- ❌ No performance improvements
- ✅ Documentation maintained
- ⚠️ Limited support

#### 3. End of Life

- ❌ No updates
- ⚠️ Security patches only if critical
- ❌ No support
- ⚠️ Documentation archived

### Version Support Table

| Version | Release Date | Active Until | Maintenance Until | EOL Date |
|---------|--------------|--------------|-------------------|----------|
| v1.0.0  | 2024-01-01   | 2025-01-01   | 2025-07-01        | 2025-07-01 |
| v2.0.0  | 2025-01-01   | 2026-01-01   | 2026-07-01        | TBD      |

## Implementation Guidelines

### FastAPI Version Routing

```python
from fastapi import APIRouter

# Version 1 router
router_v1 = APIRouter(prefix="/v1")

@router_v1.post("/documents")
async def create_document_v1(document: DocumentV1):
    """Create document - v1 API"""
    return create_document_logic_v1(document)

# Version 2 router
router_v2 = APIRouter(prefix="/v2")

@router_v2.post("/documents")
async def create_document_v2(document: DocumentV2):
    """Create document - v2 API"""
    return create_document_logic_v2(document)

# Register both versions
app.include_router(router_v1)
app.include_router(router_v2)
```

### Version-Specific Models

```python
from pydantic import BaseModel

# v1 models
class DocumentV1(BaseModel):
    title: str
    text: str
    metadata: dict = {}

# v2 models (with breaking changes)
class DocumentV2(BaseModel):
    title: str
    text: str
    tags: list[str] = []  # replaces metadata
    vector_model: str = "default"  # new required field
```

### Shared Logic with Adapters

```python
# Core logic (version-agnostic)
def create_document_core(title: str, text: str, tags: list[str]):
    """Core document creation logic"""
    # ... implementation

# v1 adapter
def create_document_logic_v1(document: DocumentV1):
    # Convert v1 format to core format
    tags = list(document.metadata.keys())
    return create_document_core(document.title, document.text, tags)

# v2 adapter
def create_document_logic_v2(document: DocumentV2):
    # v2 already matches core format
    return create_document_core(document.title, document.text, document.tags)
```

### Version Detection Middleware

```python
from fastapi import Request

@app.middleware("http")
async def add_version_header(request: Request, call_next):
    """Add API version to response headers"""
    response = await call_next(request)

    # Extract version from path
    if request.url.path.startswith("/v1"):
        response.headers["X-API-Version"] = "1"
    elif request.url.path.startswith("/v2"):
        response.headers["X-API-Version"] = "2"

    return response
```

### Deprecation Warning Middleware

```python
DEPRECATED_ENDPOINTS = {
    "/v1/documents/batch": {
        "sunset": "2024-06-01",
        "alternative": "/v1/documents/bulk",
        "removal_version": "v2.0.0"
    }
}

@app.middleware("http")
async def deprecation_warnings(request: Request, call_next):
    """Add deprecation warnings to responses"""
    response = await call_next(request)

    if request.url.path in DEPRECATED_ENDPOINTS:
        info = DEPRECATED_ENDPOINTS[request.url.path]
        response.headers["Warning"] = (
            f'299 - "Endpoint deprecated. '
            f'Will be removed in {info["removal_version"]} ({info["sunset"]}). '
            f'Use {info["alternative"]} instead."'
        )
        response.headers["X-API-Deprecation"] = "true"
        response.headers["X-API-Sunset"] = info["sunset"]

    return response
```

## Communication Checklist

When releasing a new major version:

- [ ] Create migration guide documentation
- [ ] Announce in release notes
- [ ] Blog post explaining changes
- [ ] Email existing API users
- [ ] Update code examples and SDKs
- [ ] Host migration webinar/office hours
- [ ] Update API documentation
- [ ] Add deprecation warnings to old version
- [ ] Monitor migration progress via metrics
- [ ] Provide migration support channel

## Best Practices Summary

1. **Never remove features without deprecation period** (minimum 6 months)
2. **Always provide migration guides** with code examples
3. **Use semantic versioning** consistently
4. **Communicate early and often** about breaking changes
5. **Support old versions** for reasonable time periods
6. **Monitor deprecated endpoint usage** to understand impact
7. **Provide automated migration tools** where possible
8. **Test migrations thoroughly** before major version release
9. **Document every breaking change** with alternatives
10. **Be conservative** with breaking changes

## References

- [Semantic Versioning 2.0.0](https://semver.org/)
- [RFC 7231 - HTTP Warning Header](https://tools.ietf.org/html/rfc7231#section-5.5)
- [RFC 8594 - Sunset HTTP Header](https://tools.ietf.org/html/rfc8594)
- [Stripe API Versioning](https://stripe.com/docs/api/versioning)
- [GitHub API Versioning](https://docs.github.com/en/rest/overview/api-versions)
