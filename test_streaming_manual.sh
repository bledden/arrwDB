#!/bin/bash
# Manual test for streaming endpoints

set -e

BASE_URL="http://localhost:8000/v1"

echo "================================"
echo "Streaming Endpoints Manual Test"
echo "================================"
echo ""

# Create library
echo "1. Creating test library..."
LIBRARY_RESPONSE=$(curl -s -X POST "$BASE_URL/libraries" \
  -H "Content-Type: application/json" \
  -d '{"name": "Manual Stream Test", "index_type": "brute_force"}')

LIBRARY_ID=$(echo "$LIBRARY_RESPONSE" | jq -r '.id')
echo "✓ Created library: $LIBRARY_ID"
echo ""

# Prepare NDJSON data with small chunks (faster embeddings)
echo "2. Preparing NDJSON test data (3 documents)..."
cat > /tmp/test_stream.ndjson <<'EOF'
{"title": "Doc 1", "texts": ["First document about AI"], "tags": ["test"]}
{"title": "Doc 2", "texts": ["Second document about ML"], "tags": ["test"]}
{"title": "Doc 3", "texts": ["Third document about NLP"], "tags": ["test"]}
EOF

echo "✓ Created NDJSON file"
cat /tmp/test_stream.ndjson
echo ""

# Stream documents
echo "3. Streaming documents to server..."
echo "Response:"
curl -X POST "$BASE_URL/libraries/$LIBRARY_ID/documents/stream" \
  -H "Content-Type: application/x-ndjson" \
  --data-binary @/tmp/test_stream.ndjson

echo ""
echo ""

# Verify documents were added
echo "4. Verifying documents in library..."
LIBRARY_CHECK=$(curl -s "$BASE_URL/libraries/$LIBRARY_ID")
DOC_COUNT=$(echo "$LIBRARY_CHECK" | jq '.documents | length')
echo "✓ Library has $DOC_COUNT documents"
echo ""

# Test streaming search
echo "5. Testing streaming search..."
echo "Response:"
curl -X POST "$BASE_URL/libraries/$LIBRARY_ID/search/stream?query=machine+learning&k=3"

echo ""
echo ""

# Test document export stream
echo "6. Testing document export stream..."
echo "Response:"
curl -s "$BASE_URL/libraries/$LIBRARY_ID/documents/stream"

echo ""
echo ""
echo "✅ Manual streaming test complete!"
