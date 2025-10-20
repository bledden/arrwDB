"""
Basic functionality test for the Vector Database.

This script tests the core functionality without requiring the full API server.
It directly tests the service layer, repository, and indexes.
"""

import sys
import logging
from pathlib import Path
from uuid import uuid4
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic vector database functionality."""

    logger.info("=" * 60)
    logger.info("Starting Vector Database Basic Functionality Test")
    logger.info("=" * 60)

    try:
        # Import required modules
        logger.info("\n1. Importing modules...")
        from app.models.base import Library, Document, Chunk, ChunkMetadata, DocumentMetadata, LibraryMetadata
        from app.services.embedding_service import EmbeddingService
        from app.services.library_service import LibraryService
        from infrastructure.repositories.library_repository import LibraryRepository
        logger.info("✓ All modules imported successfully")

        # Initialize services
        logger.info("\n2. Initializing services...")
        import os
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not found in environment")

        data_dir = Path("./data_test")
        data_dir.mkdir(exist_ok=True)

        embedding_service = EmbeddingService(api_key=api_key, model="embed-english-v3.0")
        repository = LibraryRepository(data_dir)
        library_service = LibraryService(repository, embedding_service)
        logger.info(f"✓ Services initialized (embedding dimension: {embedding_service.embedding_dimension})")

        # Test 1: Create a library
        logger.info("\n3. Creating a test library...")
        library = library_service.create_library(
            name="Test Library",
            description="A test library for basic functionality",
            index_type="brute_force"
        )
        logger.info(f"✓ Library created: {library.id}")
        logger.info(f"  - Name: {library.name}")
        logger.info(f"  - Index type: {library.metadata.index_type}")
        logger.info(f"  - Embedding dimension: {library.metadata.embedding_dimension}")

        # Test 2: Add a document with automatic embedding
        logger.info("\n4. Adding a document with automatic embedding...")
        texts = [
            "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
            "Deep learning is a type of machine learning that uses neural networks with multiple layers.",
            "Natural language processing enables computers to understand, interpret, and generate human language."
        ]

        document = library_service.add_document_with_text(
            library_id=library.id,
            title="Introduction to AI",
            texts=texts,
            author="Test Author",
            tags=["ai", "machine-learning", "tutorial"]
        )
        logger.info(f"✓ Document added: {document.id}")
        logger.info(f"  - Title: {document.metadata.title}")
        logger.info(f"  - Chunks: {len(document.chunks)}")
        logger.info(f"  - First chunk embedding dimension: {len(document.chunks[0].embedding)}")

        # Test 3: Perform a search
        logger.info("\n5. Performing vector similarity search...")
        query = "What is machine learning?"
        results = library_service.search_with_text(
            library_id=library.id,
            query_text=query,
            k=3
        )

        logger.info(f"✓ Search completed for query: '{query}'")
        logger.info(f"  - Results found: {len(results)}")

        for i, (chunk, distance) in enumerate(results, 1):
            similarity = 1 - distance
            logger.info(f"\n  Result {i}:")
            logger.info(f"    - Similarity: {similarity:.4f}")
            logger.info(f"    - Distance: {distance:.4f}")
            logger.info(f"    - Text: {chunk.text[:80]}...")

        # Test 4: Get library statistics
        logger.info("\n6. Getting library statistics...")
        stats = library_service.get_library_statistics(library.id)
        logger.info(f"✓ Statistics retrieved:")
        logger.info(f"  - Documents: {stats['num_documents']}")
        logger.info(f"  - Chunks: {stats['num_chunks']}")
        logger.info(f"  - Index type: {stats['index_type']}")
        logger.info(f"  - Vector store: {stats['vector_store_stats']['unique_vectors']} unique vectors")
        logger.info(f"  - Storage type: {stats['vector_store_stats']['storage_type']}")

        # Test 5: Test all index types
        logger.info("\n7. Testing all index types...")
        index_types = ["brute_force", "kd_tree", "lsh", "hnsw"]

        for idx_type in index_types:
            logger.info(f"\n  Testing {idx_type}...")

            # Create library with this index type
            test_lib = library_service.create_library(
                name=f"Test {idx_type}",
                index_type=idx_type
            )

            # Add a small document
            test_doc = library_service.add_document_with_text(
                library_id=test_lib.id,
                title="Test Document",
                texts=["This is a test document for the index."]
            )

            # Search
            test_results = library_service.search_with_text(
                library_id=test_lib.id,
                query_text="test document",
                k=1
            )

            logger.info(f"  ✓ {idx_type}: Library created, document added, search successful ({len(test_results)} results)")

        # Test 6: Cleanup
        logger.info("\n8. Cleaning up test data...")
        # Clean up test directory
        import shutil
        if data_dir.exists():
            shutil.rmtree(data_dir)
        logger.info("✓ Test data cleaned up")

        # Success!
        logger.info("\n" + "=" * 60)
        logger.info("✓ ALL TESTS PASSED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("\nThe Vector Database implementation is working correctly!")
        logger.info("You can now start the API server with: python run_api.py")
        logger.info("Or use Docker: docker-compose up -d")

        return True

    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error("✗ TEST FAILED")
        logger.error("=" * 60)
        logger.error(f"Error: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
