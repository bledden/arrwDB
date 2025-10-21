#!/bin/bash

echo "=========================================="
echo "Pre-Submission Validation Script"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Track results
PASS=0
FAIL=0

# Test 1: Run automated tests
echo "1. Running automated test suite..."
python3 -m pytest tests/unit tests/test_edge_cases.py -q --tb=no > /tmp/test_output.txt 2>&1
if grep -q "466 passed" /tmp/test_output.txt; then
    echo -e "${GREEN}✅ Tests: 466 passing${NC}"
    ((PASS++))
else
    echo -e "${RED}❌ Tests: Some failures${NC}"
    ((FAIL++))
fi

# Test 2: API Health Check
echo ""
echo "2. Testing API startup..."
python3 run_api.py > /tmp/api.log 2>&1 &
API_PID=$!
sleep 5

HEALTH=$(curl -s http://localhost:8000/health)
if echo "$HEALTH" | grep -q "healthy"; then
    echo -e "${GREEN}✅ API: Running and healthy${NC}"
    ((PASS++))
else
    echo -e "${RED}❌ API: Not responding${NC}"
    ((FAIL++))
fi

# Test 3: SDK Functionality
echo ""
echo "3. Testing SDK client..."
python3 << 'PYTHON_EOF'
try:
    from sdk import VectorDBClient
    client = VectorDBClient("http://localhost:8000")
    library = client.create_library(name="Validation Test", index_type="hnsw")
    doc = client.add_document(
        library_id=library["id"],
        title="Test",
        texts=["Test text"]
    )
    results = client.search(library_id=library["id"], query="test", k=1)
    print("✅ SDK: All operations working")
    exit(0)
except Exception as e:
    print(f"❌ SDK: Failed - {e}")
    exit(1)
PYTHON_EOF

if [ $? -eq 0 ]; then
    ((PASS++))
else
    ((FAIL++))
fi

# Test 4: Persistence
echo ""
echo "4. Testing persistence..."
if [ -f "data/wal/wal_00000001.log" ]; then
    echo -e "${GREEN}✅ Persistence: WAL files created${NC}"
    ((PASS++))
else
    echo -e "${RED}❌ Persistence: No WAL files${NC}"
    ((FAIL++))
fi

# Test 5: Docker
echo ""
echo "5. Testing Docker build..."
docker images | grep -q vectordb-api
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Docker: Image exists${NC}"
    ((PASS++))
else
    echo -e "${RED}⚠️  Docker: Image not built (run: docker build -t vectordb-api .)${NC}"
    ((FAIL++))
fi

# Cleanup
kill $API_PID 2>/dev/null
wait $API_PID 2>/dev/null

# Summary
echo ""
echo "=========================================="
echo "VALIDATION SUMMARY"
echo "=========================================="
echo -e "Passed: ${GREEN}$PASS${NC}"
echo -e "Failed: ${RED}$FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✅ ALL VALIDATIONS PASSED - READY TO SUBMIT!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some checks failed - review above${NC}"
    exit 1
fi
