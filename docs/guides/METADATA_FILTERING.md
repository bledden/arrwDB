# Metadata Filtering Guide

This guide explains how to use metadata filtering to refine search results based on chunk metadata fields.

## Overview

Metadata filtering allows you to combine vector similarity search with structured data filtering. The system first performs semantic search using vector embeddings, then applies metadata filters to narrow down results.

**Key Features:**
- **8 comparison operators**: `eq`, `ne`, `gt`, `lt`, `gte`, `lte`, `in`, `contains`
- **AND logic**: Multiple filters must all be satisfied
- **Post-search filtering**: Filters applied after vector search for efficiency
- **Automatic oversampling**: System fetches 10x results then filters to requested k

## Available Metadata Fields

Each chunk has the following metadata fields available for filtering:

| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `created_at` | datetime | Chunk creation timestamp | `"2024-01-15T10:30:00"` |
| `page_number` | int (optional) | Page number from source document | `1`, `10`, `null` |
| `chunk_index` | int | Position within parent document | `0`, `1`, `2` (0-indexed) |
| `source_document_id` | UUID | Parent document identifier | `"123e4567-e89b-12d3-a456-426614174000"` |

**Note**: Document-level metadata (title, author, tags) is not directly filterable at the chunk level but is included in search results.

## Supported Operators

### Equality Operators

**`eq` (equals)**
```json
{"field": "chunk_index", "operator": "eq", "value": 0}
```
Matches chunks where the field exactly equals the value.

**`ne` (not equals)**
```json
{"field": "chunk_index", "operator": "ne", "value": 0}
```
Matches chunks where the field does not equal the value.

### Comparison Operators

**`gt` (greater than)**
```json
{"field": "chunk_index", "operator": "gt", "value": 5}
```
Matches chunks where the field is greater than the value.

**`lt` (less than)**
```json
{"field": "chunk_index", "operator": "lt", "value": 10}
```
Matches chunks where the field is less than the value.

**`gte` (greater than or equal)**
```json
{"field": "chunk_index", "operator": "gte", "value": 5}
```
Matches chunks where the field is greater than or equal to the value.

**`lte` (less than or equal)**
```json
{"field": "chunk_index", "operator": "lte", "value": 10}
```
Matches chunks where the field is less than or equal to the value.

### Collection Operators

**`in` (value in list)**
```json
{"field": "chunk_index", "operator": "in", "value": [0, 1, 2]}
```
Matches chunks where the field value is in the provided list.

**`contains` (string/list containment)**
```json
{"field": "custom_field", "operator": "contains", "value": "substring"}
```
For strings: checks if value is a substring.
For lists: checks if value is an element.

## Usage Examples

### Python SDK

#### Basic Filtering

```python
from sdk import VectorDBClient

client = VectorDBClient("http://localhost:8000")

# Filter for first chunks only (chunk_index == 0)
filters = [
    {"field": "chunk_index", "operator": "eq", "value": 0}
]

results = client.search_with_filters(
    library_id="your-library-id",
    query="machine learning",
    metadata_filters=filters,
    k=10
)

for result in results["results"]:
    print(f"Text: {result['chunk']['text'][:100]}...")
    print(f"Chunk Index: {result['chunk']['metadata']['chunk_index']}")
```

#### Range Filtering

```python
# Filter for chunks from the middle of documents (indices 2-9)
filters = [
    {"field": "chunk_index", "operator": "gte", "value": 2},
    {"field": "chunk_index", "operator": "lt", "value": 10}
]

results = client.search_with_filters(
    library_id="your-library-id",
    query="supervised learning",
    metadata_filters=filters,
    k=5
)
```

#### Multiple Filters (AND Logic)

```python
# Find recent chunks from specific document
filters = [
    {"field": "source_document_id", "operator": "eq", "value": "doc-uuid-here"},
    {"field": "chunk_index", "operator": "lt", "value": 5}
]

results = client.search_with_filters(
    library_id="your-library-id",
    query="neural networks",
    metadata_filters=filters,
    k=10
)
```

#### List-Based Filtering

```python
# Filter for chunks from specific positions
filters = [
    {"field": "chunk_index", "operator": "in", "value": [0, 5, 10, 15]}
]

results = client.search_with_filters(
    library_id="your-library-id",
    query="deep learning",
    metadata_filters=filters,
    k=20
)
```

### REST API (cURL)

#### Basic Request

```bash
curl -X POST http://localhost:8000/v1/libraries/{library_id}/search/filtered \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "k": 10,
    "metadata_filters": [
      {"field": "chunk_index", "operator": "eq", "value": 0}
    ]
  }'
```

#### Range Query

```bash
curl -X POST http://localhost:8000/v1/libraries/{library_id}/search/filtered \
  -H "Content-Type: application/json" \
  -d '{
    "query": "supervised learning",
    "k": 5,
    "metadata_filters": [
      {"field": "chunk_index", "operator": "gte", "value": 2},
      {"field": "chunk_index", "operator": "lt", "value": 10}
    ],
    "distance_threshold": 0.8
  }'
```

#### With Embeddings

```bash
curl -X POST http://localhost:8000/v1/libraries/{library_id}/search/filtered?include_embeddings=true \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural networks",
    "k": 10,
    "metadata_filters": [
      {"field": "chunk_index", "operator": "ne", "value": 0}
    ]
  }'
```

## Response Format

The filtered search endpoint returns the same response format as regular search:

```json
{
  "results": [
    {
      "chunk": {
        "id": "chunk-uuid",
        "text": "Chunk text content...",
        "metadata": {
          "created_at": "2024-01-15T10:30:00",
          "chunk_index": 2,
          "page_number": null,
          "source_document_id": "doc-uuid"
        }
      },
      "distance": 0.234,
      "document_id": "doc-uuid",
      "document_title": "Introduction to ML"
    }
  ],
  "query_time_ms": 15.34,
  "total_results": 5
}
```

## How It Works

### Execution Flow

1. **Vector Search**: System performs semantic search using query embedding
2. **Oversampling**: Fetches 10x the requested `k` results (e.g., k=10 → fetch 100)
3. **Filter Application**: Each metadata filter is applied with AND logic
4. **Result Limiting**: Returns top `k` results after filtering

### Why Oversample?

Fetching 10x results ensures sufficient matches after filtering. For example:

```python
# Request k=10 with restrictive filter
filters = [{"field": "chunk_index", "operator": "eq", "value": 0}]

# System flow:
# 1. Fetch 100 most similar chunks (10 × k)
# 2. Filter: keep only chunks with chunk_index == 0
# 3. Return top 10 of filtered results
```

If we only fetched 10 initial results, we might end up with 0-2 matches after filtering.

### Filter Logic

**AND Operation**: All filters must be satisfied:

```python
filters = [
    {"field": "chunk_index", "operator": "gte", "value": 2},
    {"field": "chunk_index", "operator": "lt", "value": 10}
]

# Chunk must satisfy: chunk_index >= 2 AND chunk_index < 10
# Result: Only chunks with indices 2-9 are returned
```

**No OR Support**: Currently only AND logic is supported. To simulate OR, make multiple requests.

## Performance Considerations

### Efficiency Tips

1. **Use specific filters**: More restrictive filters = better performance
   ```python
   # Good: Specific range
   {"field": "chunk_index", "operator": "in", "value": [0, 1, 2]}

   # Less efficient: Very broad range
   {"field": "chunk_index", "operator": "lt", "value": 1000}
   ```

2. **Filter selectivity**: Consider how many results will pass each filter
   - High selectivity (few matches): Increase k if needed
   - Low selectivity (many matches): Standard k works well

3. **Empty results**: If filters are too restrictive, you may get 0 results
   ```python
   # May return 0 results if no chunks match
   {"field": "chunk_index", "operator": "eq", "value": 999}
   ```

### Query Performance

**Typical Latency** (100K vectors, 256D):
- Regular search: 1.40ms average
- Filtered search: 2-5ms average (depends on filter complexity)

**Factors Affecting Performance**:
- Number of filters: More filters = slightly slower (each adds a check)
- Filter type: `eq` is fastest, `contains` is slowest
- Result set size: Larger k = more results to filter

## Common Patterns

### 1. Filter by Document Section

Get only chunks from the beginning of documents:

```python
filters = [{"field": "chunk_index", "operator": "lt", "value": 5}]
```

### 2. Exclude First Chunks

Skip introductory content:

```python
filters = [{"field": "chunk_index", "operator": "ne", "value": 0}]
```

### 3. Filter by Specific Document

Search within a single document:

```python
filters = [{"field": "source_document_id", "operator": "eq", "value": "doc-uuid"}]
```

### 4. Sample Evenly

Get every 5th chunk:

```python
filters = [{"field": "chunk_index", "operator": "in", "value": [0, 5, 10, 15, 20]}]
```

### 5. Recent Content Only

Filter by creation time (if timestamps matter):

```python
from datetime import datetime, timedelta

cutoff = (datetime.now() - timedelta(days=7)).isoformat()
filters = [{"field": "created_at", "operator": "gte", "value": cutoff}]
```

## Limitations

1. **No Document-Level Filtering**: Cannot directly filter by document title, author, or tags at the chunk level. These fields are returned in results but not filterable.

2. **AND Logic Only**: Multiple filters use AND logic. No built-in OR support.

3. **Post-Search Filtering**: Filters applied after vector search, not during. This is efficient but means:
   - Some computational cost already spent on non-matching results
   - Very restrictive filters may need higher k values

4. **No Aggregations**: Cannot compute statistics (count, avg, etc.) on filtered results.

5. **No Full-Text Search**: Metadata filtering is for structured fields only. Use the `query` parameter for semantic text search.

## Best Practices

1. **Combine with semantic search**: Metadata filters work best alongside meaningful queries
   ```python
   # Good: Semantic query + metadata filter
   query="machine learning algorithms"
   filters=[{"field": "chunk_index", "operator": "gte", "value": 2}]

   # Less useful: Empty query, filters only
   query=""  # Don't do this
   ```

2. **Test filter selectivity**: Check how many results match your filters
   ```python
   # Start broad, then narrow
   results_no_filter = client.search(library_id, query, k=100)
   results_with_filter = client.search_with_filters(library_id, query, filters, k=100)

   print(f"Without filter: {len(results_no_filter['results'])}")
   print(f"With filter: {len(results_with_filter['results'])}")
   ```

3. **Use distance threshold**: Combine with metadata filters for quality control
   ```python
   results = client.search_with_filters(
       library_id=lib_id,
       query="deep learning",
       metadata_filters=filters,
       k=10,
       distance_threshold=0.7  # Only accept good matches
   )
   ```

4. **Handle empty results**: Always check if filtering returned any matches
   ```python
   results = client.search_with_filters(library_id, query, filters, k=10)

   if results["total_results"] == 0:
       print("No chunks matched the filters, trying without filters...")
       results = client.search(library_id, query, k=10)
   ```

## Troubleshooting

### Problem: Getting 0 Results

**Causes:**
- Filters too restrictive
- Field values don't match expectations
- Wrong field names or types

**Solutions:**
```python
# 1. Try without filters first
results = client.search(library_id, query, k=10)
print(f"Without filters: {results['total_results']} results")

# 2. Check actual metadata values
for result in results["results"][:3]:
    print(result["chunk"]["metadata"])

# 3. Adjust filters based on actual data
```

### Problem: Slow Performance

**Causes:**
- Too many results to filter
- Complex filter conditions
- Large k value

**Solutions:**
```python
# 1. Reduce k if possible
results = client.search_with_filters(library_id, query, filters, k=5)  # Not 100

# 2. Use more selective filters
# Instead of: {"operator": "lt", "value": 1000}
# Use: {"operator": "in", "value": [0, 1, 2, 3, 4]}

# 3. Consider caching for repeated queries
```

### Problem: Unexpected Filter Behavior

**Causes:**
- Misunderstanding AND logic
- Type mismatches (string vs int)

**Solutions:**
```python
# 1. Test filters individually
filter1 = [{"field": "chunk_index", "operator": "gte", "value": 2}]
filter2 = [{"field": "chunk_index", "operator": "lt", "value": 10}]

results1 = client.search_with_filters(library_id, query, filter1, k=20)
results2 = client.search_with_filters(library_id, query, filter2, k=20)
results_both = client.search_with_filters(library_id, query, filter1 + filter2, k=20)

print(f"Filter 1: {results1['total_results']}")
print(f"Filter 2: {results2['total_results']}")
print(f"Both (AND): {results_both['total_results']}")

# 2. Check value types
# chunk_index is an integer, not a string!
{"field": "chunk_index", "operator": "eq", "value": 0}  # Correct
{"field": "chunk_index", "operator": "eq", "value": "0"}  # Wrong!
```

## See Also

- [API Reference](INDEX.md) - Complete API documentation
- [Quick Start Guide](QUICKSTART.md) - Getting started tutorial
- [SDK Documentation](../../sdk/README.md) - Python client library reference
