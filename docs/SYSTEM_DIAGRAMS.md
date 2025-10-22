# System Design Visual Diagrams
## Architectural Visualizations for Demo Video

These diagrams can be shown during the demo video to illustrate system design concepts.

---

## 📐 Diagram 1: High-Level Architecture (Simple)

**Use this in:** Part 2 (Architecture Deep Dive)

```
┌─────────────────────────────────────────────────────────────┐
│                    REST API (FastAPI)                        │
│         POST /libraries, /documents, /search                 │
│              Automatic OpenAPI Docs (/docs)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               Service Layer (Business Logic)                 │
│         LibraryService    │    EmbeddingService              │
│     (Orchestration)       │    (Cohere Integration)          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          Repository Layer (Thread-Safe Data Access)          │
│              LibraryRepository + R-W Lock                    │
│     Multiple readers OR single writer (exclusive)            │
└──────┬──────────┬──────────┬──────────┬─────────────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
   ┌────────┐ ┌──────┐  ┌───────┐  ┌──────────┐
   │ Vector │ │Index │  │Embed  │  │Persistence│
   │ Store  │ │ (4x) │  │Contract│ │WAL+Snapshot│
   └────────┘ └──────┘  └───────┘  └──────────┘
```

**Talking Points:**
- "Four distinct layers, each with single responsibility"
- "Request flows top to bottom, responses flow back up"
- "Repository layer is the only place with locks - centralized concurrency control"

---

## 📐 Diagram 2: Detailed Component View (Medium)

**Use this in:** Part 2 (Architecture) or Part 7 (Wrap-up)

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   /health    │  │  /libraries  │  │   /search    │          │
│  │  (endpoint)  │  │  (endpoint)  │  │  (endpoint)  │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                   │
│         └─────────────────┼──────────────────┘                   │
└───────────────────────────┼──────────────────────────────────────┘
                            │
                            ▼ Depends on
┌─────────────────────────────────────────────────────────────────┐
│                     LibraryService                               │
│  ┌────────────────────────────────────────────────────┐         │
│  │ • create_library()                                 │         │
│  │ • add_document_with_embeddings()                   │         │
│  │ • search()                                         │         │
│  └────────────────────────────────────────────────────┘         │
│         │                                        │               │
│         │ Uses                                   │ Uses          │
│         ▼                                        ▼               │
│  ┌─────────────────┐                  ┌──────────────────┐     │
│  │EmbeddingService │                  │LibraryRepository │     │
│  │ (Cohere API)    │                  │  (Data Access)   │     │
│  └─────────────────┘                  └──────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                                                   │
                                                   │ Uses
                                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                           │
│                                                                  │
│  ┌────────────────────────────────────────────────────┐         │
│  │              LibraryRepository                     │         │
│  │   ┌──────────────────────────────────────────┐    │         │
│  │   │    ReaderWriterLock (Thread Safety)      │    │         │
│  │   │  • read(): Multiple concurrent allowed   │    │         │
│  │   │  • write(): Exclusive access             │    │         │
│  │   └──────────────────────────────────────────┘    │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                  │
│  Coordinates:                                                    │
│  ┌──────────────┐ ┌───────────┐ ┌───────────┐ ┌─────────────┐ │
│  │ VectorStore  │ │  Index    │ │ Embedding │ │ Persistence │ │
│  │              │ │ (4 types) │ │ Contract  │ │ (WAL+Snap)  │ │
│  │ • add_vector │ │ • HNSW    │ │ • validate│ │ • append_op │ │
│  │ • get_vector │ │ • LSH     │ │ • normalize│ │ • snapshot  │ │
│  │ • remove     │ │ • KD-Tree │ │           │ │ • recover   │ │
│  │ • dedup      │ │ • BruteF. │ │           │ │             │ │
│  └──────────────┘ └───────────┘ └───────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Talking Points:**
- "Dependency injection - FastAPI provides LibraryService to endpoints"
- "Service layer coordinates between embedding generation and data storage"
- "Repository is the single point of thread safety"
- "Four pluggable index implementations behind common interface"

---

## 📐 Diagram 3: HNSW Graph Structure (Visual)

**Use this in:** Part 3 (HNSW Deep Dive)

```
HNSW Graph Structure - Multi-Layer Navigation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Layer 2 (Top - Sparse "Highways")
   Node A ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━► Node D
     ↓                                              ↓
     ↓                                              ↓

Layer 1 (Medium Density)
   Node A ━━━━━━━━━► Node B ━━━━━━━━━━━━━━━━━━━► Node D
     ↓                 ↓                            ↓
     ↓                 ↓                            ↓

Layer 0 (Bottom - Dense "City Streets", ALL vectors)
   Node A ━━► Node B ━━► Node C ━━► Node D ━━► Node E
     ↕        ↕         ↕         ↕         ↕
   (local connections between nearest neighbors)


SEARCH EXAMPLE: Finding nearest neighbor to Query Q
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Start at Layer 2 (entry point = A)
   Q is closer to D than A
   ➜ Jump A → D (long-range connection)

Step 2: Descend to Layer 1 at D
   Q is closer to B than D
   ➜ Navigate D → B

Step 3: Descend to Layer 0 at B
   Q is very close to C
   ➜ Navigate B → C
   C is nearest neighbor! ✓

Total hops: 3 (vs checking all 5 nodes with brute force)
Complexity: O(log n) where n = number of vectors
```

**Simplified Version (Even Simpler):**

```
┌─────────────────────────────────────────────────┐
│  HNSW = Multi-Layer Graph (Like a Highway Map)  │
└─────────────────────────────────────────────────┘

Layer 2: [A]─────────────────────►[D]
         (few nodes, long jumps = highways)

Layer 1: [A]────►[B]──────────►[D]
         (medium density = main roads)

Layer 0: [A]─►[B]─►[C]─►[D]─►[E]
         (all vectors, local = city streets)

Search Strategy: Start at top (A) → Long jump (D)
                 → Descend → Local navigation (B→C)
                 → Find exact neighbor!

Why it's fast: Navigate mostly at top layers (few nodes)
               Only explore densely at the end (precise)
```

**Talking Points:**
- "Like Google Maps - zoom out for highways, zoom in for streets"
- "Top layer has maybe 10 nodes out of 10,000 - that's log(n)"
- "Each hop gets you closer until you find the nearest neighbor"
- "This is why we get O(log n) - limited hops through hierarchy"

---

## 📐 Diagram 4: Reader-Writer Lock Behavior

**Use this in:** Part 2 (Architecture) when explaining concurrency

```
READER-WRITER LOCK: How Concurrent Access Works
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Scenario 1: Multiple Readers (ALLOWED - No Blocking)
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│Thread 1 │  │Thread 2 │  │Thread 3 │  │Thread 4 │
│SEARCH   │  │SEARCH   │  │SEARCH   │  │SEARCH   │
└────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
     │            │            │            │
     └────────────┴────────────┴────────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │   Read Lock (R)     │  ✓ All proceed simultaneously
        │   Readers: 4        │  ✓ No blocking
        │   Writers: 0        │
        └─────────────────────┘


Scenario 2: Writer Arrives (Blocks New Readers)
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│Thread 1 │  │Thread 2 │  │Thread 3 │  │Thread 4 │
│SEARCH   │  │SEARCH   │  │ INSERT  │  │SEARCH   │
│(active) │  │(active) │  │(waiting)│  │(blocked)│
└────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
     │            │            │            │
     └────────────┴────────────┼────────────┘
                               │
                               ▼
        ┌─────────────────────────────────────────┐
        │   Read Lock (R)                         │
        │   Active Readers: 2 (Thread 1, 2)       │
        │   Waiting Writer: 1 (Thread 3)          │
        │   Blocked Readers: 1 (Thread 4)         │
        │                                          │
        │   Thread 4 must wait for writer!        │
        │   (Writer priority prevents starvation) │
        └─────────────────────────────────────────┘


Scenario 3: Writer Executes (Exclusive Access)
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│Thread 1 │  │Thread 2 │  │Thread 3 │  │Thread 4 │
│(done)   │  │(done)   │  │ INSERT  │  │SEARCH   │
│         │  │         │  │(active) │  │(blocked)│
└─────────┘  └─────────┘  └────┬────┘  └────┬────┘
                               │            │
                               └────────────┘
                               │
                               ▼
        ┌─────────────────────────────────────────┐
        │   Write Lock (W) - EXCLUSIVE            │
        │   Active Writers: 1 (Thread 3)          │
        │   Blocked Readers: 1 (Thread 4)         │
        │                                          │
        │   Only ONE writer, everyone else waits  │
        └─────────────────────────────────────────┘


KEY PROPERTIES:
✓ Multiple readers = OK (concurrent)
✓ Multiple writers = NO (exclusive)
✓ Reader + Writer = NO (exclusive)
✓ Writer waiting → New readers block (writer priority)
```

**Talking Points:**
- "Read-heavy workload benefits from concurrent reads"
- "Writer priority prevents updates from starving"
- "Typical vector DB: 90% searches (reads), 10% inserts (writes)"

---

## 📐 Diagram 5: Request Flow Example

**Use this in:** Part 5 (Live Demo) to show what happens behind the scenes

```
USER ADDS A DOCUMENT - Full Request Flow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Client Request
   POST /v1/libraries/{lib_id}/documents
   {
     "title": "ML Tutorial",
     "texts": ["ML is...", "Supervised learning..."]
   }
              │
              ▼
2. FastAPI Endpoint
   ┌─────────────────────────────────┐
   │ POST /documents endpoint        │
   │ • Validates request (Pydantic)  │
   │ • Extracts library_id, texts    │
   └────────────┬────────────────────┘
                │
                ▼
3. Service Layer
   ┌─────────────────────────────────┐
   │ LibraryService                  │
   │ • Calls EmbeddingService        │
   └────────┬───────────┬────────────┘
            │           │
            │           └──────► ┌────────────────────┐
            │                    │ EmbeddingService   │
            │                    │ • Calls Cohere API │
            │                    │ • Returns vectors  │
            │                    │   [0.1, 0.2, ...]  │
            │           ┌────────┴────────────────────┘
            │           │
            ▼           ▼
4. Repository Layer
   ┌──────────────────────────────────────┐
   │ LibraryRepository.add_document()     │
   │ • Acquires WRITE lock                │
   │ • Blocks all readers & writers       │
   │ └───────────────────────────────────┐│
   │         while write is active       ││
   └─────────────────────────────────────┘│
            │                              │
            ▼                              │
5. Persistence (WAL)                       │
   ┌──────────────────────────────────┐   │
   │ WriteAheadLog.append_operation() │   │
   │ • Logs: "ADD_DOCUMENT lib=X"     │   │
   │ • fsync() to disk (durable)      │   │
   └────────────┬─────────────────────┘   │
                │                          │
                ▼                          │
6. Vector Storage                          │
   ┌──────────────────────────────────┐   │
   │ VectorStore.add_vector()         │   │
   │ • Hash vector (dedup check)      │   │
   │ • Store if new, ref++ if exists  │   │
   │ • Returns vector index           │   │
   └────────────┬─────────────────────┘   │
                │                          │
                ▼                          │
7. Index Update                            │
   ┌──────────────────────────────────┐   │
   │ HNSWIndex.add_vector()           │   │
   │ • Assign layer (exponential)     │   │
   │ • Find M nearest neighbors       │   │
   │ • Create bidirectional edges     │   │
   │ • Update graph structure         │   │
   └────────────┬─────────────────────┘   │
                │                          │
                └──────────────────────────┘
                │
                ▼ Release WRITE lock
8. Response
   ┌──────────────────────────────────┐
   │ 201 Created                      │
   │ {                                │
   │   "id": "doc-uuid",              │
   │   "title": "ML Tutorial",        │
   │   "chunks": [...]                │
   │ }                                │
   └──────────────────────────────────┘

TOTAL TIME: ~100-200ms
• Cohere API: ~80ms (external)
• HNSW insert: ~3-5ms per vector
• WAL write: ~1ms (fsync)
• Serialization: ~10ms
```

**Simplified Version:**

```
Request → API → Service → Repository
                         ↓
            ┌────────────┴─────────────┐
            │                          │
            ▼                          ▼
    WAL (durability)          VectorStore + HNSW
    "log operation"           "store & index"
            │                          │
            └────────────┬─────────────┘
                         ↓
                    Response
```

**Talking Points:**
- "Six layers of abstraction, each with clear responsibility"
- "WAL ensures durability before applying changes"
- "Write lock guarantees no concurrent modifications"
- "HNSW graph updated incrementally, no full rebuild"

---

## 📐 Diagram 6: Search Operation Flow

**Use this for:** Part 5 (Live Demo) explaining search

```
USER SEARCHES - Request Flow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Query: "How does supervised learning work?"
              │
              ▼
2. Embed Query (Cohere API)
   ┌────────────────────────────────────┐
   │ EmbeddingService.embed_text()      │
   │ Input: "How does supervised..."    │
   │ Output: [0.15, 0.32, ..., 0.08]    │
   │         (1024-dim vector)          │
   └────────────────┬───────────────────┘
                    │
                    ▼
3. Search Index (HNSW)
   ┌────────────────────────────────────┐
   │ Repository.search() [READ lock]    │
   │ • Multiple searches can run now    │
   │ • No writers active                │
   │                                    │
   │ HNSWIndex.search(query, k=10)      │
   │ Layer 2: Start at entry, navigate │
   │ Layer 1: Descend, narrow search   │
   │ Layer 0: Find exact k=10 neighbors│
   │                                    │
   │ Returns: [(chunk_id, distance)]   │
   └────────────────┬───────────────────┘
                    │
                    ▼
4. Fetch Chunk Text
   ┌────────────────────────────────────┐
   │ Map vector IDs → Chunks            │
   │ chunk_id_1 → "Supervised learning..."│
   │ chunk_id_2 → "ML is a subset..."  │
   │ chunk_id_3 → "Neural networks..."  │
   └────────────────┬───────────────────┘
                    │
                    ▼
5. Return Results (Ranked)
   [
     {
       "text": "Supervised learning uses...",
       "distance": 0.12,  ← Most similar
       "document": "ML Tutorial"
     },
     {
       "text": "ML is a subset...",
       "distance": 0.35,
       "document": "ML Tutorial"
     }
   ]

TOTAL TIME: ~85-90ms
• Embed query: ~80ms (Cohere)
• HNSW search: 1-3ms ← THIS IS THE FLEX!
• Fetch text: <1ms
• Serialize: ~5ms
```

**Talking Points:**
- "Search is mostly embedding time (80ms) + network"
- "HNSW search itself is sub-3ms - that's the algorithmic win"
- "Multiple searches run concurrently (READ lock)"
- "Results ranked by distance (lower = more similar)"

---

## 📐 Diagram 7: Memory Efficiency (Deduplication)

**Use this for:** Part 4 (Production Features)

```
VECTOR DEDUPLICATION - Memory Savings
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WITHOUT Deduplication (Naive):
┌─────────────────────────────────────────────────┐
│ Document 1: "Copyright 2024..."                 │
│   Chunk 1 → [0.5, 0.2, ...] (4KB)              │
│   Chunk 2 → [0.1, 0.8, ...] (4KB)              │
│                                                 │
│ Document 2: "Copyright 2024..."  ← Same text!  │
│   Chunk 3 → [0.5, 0.2, ...] (4KB) ← Duplicate! │
│   Chunk 4 → [0.3, 0.7, ...] (4KB)              │
│                                                 │
│ Document 3: "Copyright 2024..."  ← Same again! │
│   Chunk 5 → [0.5, 0.2, ...] (4KB) ← Duplicate! │
│   Chunk 6 → [0.6, 0.1, ...] (4KB)              │
└─────────────────────────────────────────────────┘
Total: 6 chunks × 4KB = 24KB


WITH Deduplication (My Implementation):
┌─────────────────────────────────────────────────┐
│ VectorStore (Reference Counted):                │
│                                                 │
│ Vector 0: [0.5, 0.2, ...] → ref_count = 3      │
│           ↑       ↑        ↑                    │
│   Chunk 1 ┘       │        │                    │
│          Chunk 3 ─┘        │                    │
│                 Chunk 5 ───┘                    │
│                                                 │
│ Vector 1: [0.1, 0.8, ...] → ref_count = 1      │
│           ↑                                     │
│   Chunk 2 ┘                                     │
│                                                 │
│ Vector 2: [0.3, 0.7, ...] → ref_count = 1      │
│           ↑                                     │
│   Chunk 4 ┘                                     │
│                                                 │
│ Vector 3: [0.6, 0.1, ...] → ref_count = 1      │
│           ↑                                     │
│   Chunk 6 ┘                                     │
└─────────────────────────────────────────────────┘
Total: 4 unique vectors × 4KB = 16KB
Savings: (24KB - 16KB) / 24KB = 33%

In my tests: 48% savings!
```

**Talking Points:**
- "Repeated text is common: headers, footers, disclaimers"
- "Store vector once, multiple chunks reference it"
- "Reference counting tracks usage, frees when count → 0"
- "Real-world benefit: 48% less memory in validation tests"

---

## 🎨 How to Use These Diagrams

### Option 1: Screen Share While Recording
- Open this file in a markdown viewer or IDE
- Switch to diagram when explaining that concept
- Point at parts with cursor

### Option 2: Export as Images
Use a tool to convert to images:
```bash
# Using a tool like Monodraw (Mac) or asciiflow.com
# Or just screenshot the terminal
```

### Option 3: Slides (Post-Production)
- Create slides with these diagrams
- Insert between code sections in final video
- Add during editing phase

### Option 4: Live Draw (Advanced)
- Use a tablet/drawing app
- Draw simplified versions while explaining
- More personal but takes practice

---

## 📏 Complexity Levels

**Pick based on audience:**

### For Technical Interviewers:
- Use Diagrams 2, 3, 4, 5, 6 (detailed)
- They want to see you understand systems

### For Non-Technical:
- Use Diagrams 1, 3 (simplified versions)
- Focus on concepts, not implementation

### For Demo Video:
- Start with Diagram 1 (overview)
- Show Diagram 3 when explaining HNSW
- Maybe show Diagram 4 for R-W lock
- Keep others as backup if time allows

---

## 🎯 Quick Reference: Which Diagram When

| Video Section | Best Diagram | Why |
|---------------|--------------|-----|
| Part 1: Intro | Diagram 1 (Simple) | Quick overview |
| Part 2: Architecture | Diagram 2 or 4 | Show layers + concurrency |
| Part 3: HNSW | Diagram 3 (Visual) | Core algorithm |
| Part 4: Prod Features | Diagram 7 (Memory) | Show dedup benefit |
| Part 5: Live Demo | Diagram 5 or 6 | Request flow |
| Part 6: Testing | None needed | Just show results |
| Part 7: Wrap-up | Diagram 1 again | Bookend the demo |

---

These diagrams range from **simple** (Diagram 1 - 10 lines) to **detailed** (Diagram 5 - full flow).

**Recommendation for video:** Use Diagram 1 + Diagram 3 (HNSW). Those two alone tell the story. The others are backup if you want more depth.
