# arrwDB MCP Server

Model Context Protocol (MCP) server for arrwDB vector database. Enables AI assistants like Claude to interact with arrwDB for semantic search, document management, and novel vector operations.

## Features

### Tools Available

| Tool | Description |
|------|-------------|
| `arrwdb_search` | Semantic similarity search |
| `arrwdb_temperature_search` | Exploration/exploitation controlled search |
| `arrwdb_list_libraries` | List all libraries |
| `arrwdb_get_library` | Get library details |
| `arrwdb_create_library` | Create new library |
| `arrwdb_library_stats` | Get library statistics |
| `arrwdb_add_document` | Add document with auto-embedding |
| `arrwdb_index_oracle` | Get index type recommendation |
| `arrwdb_embedding_health` | Analyze embedding quality |
| `arrwdb_health_check` | Check server health |

## Installation

### From Source

```bash
cd mcp-server
cargo build --release
```

Binary will be at `target/release/arrwdb-mcp`.

### Pre-built Binary

Download from [Releases](https://github.com/bledden/arrwDB/releases).

## Usage

### With Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "arrwdb": {
      "command": "/path/to/arrwdb-mcp",
      "args": ["--url", "http://localhost:8000"],
      "env": {
        "ARRWDB_API_KEY": "your-api-key"
      }
    }
  }
}
```

### With Claude Code

Add to your project's MCP configuration:

```json
{
  "mcp": {
    "servers": {
      "arrwdb": {
        "command": "arrwdb-mcp",
        "args": ["--url", "http://localhost:8000"]
      }
    }
  }
}
```

### Standalone

```bash
# Basic usage
arrwdb-mcp --url http://localhost:8000

# With API key
arrwdb-mcp --url http://localhost:8000 --api-key your-key

# With debug logging
arrwdb-mcp --url http://localhost:8000 --debug
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARRWDB_URL` | arrwDB server URL | `http://localhost:8000` |
| `ARRWDB_API_KEY` | API key for authentication | None |

## Example Usage in Claude

Once configured, you can ask Claude things like:

> "Search my research papers for information about neural network architectures"

> "Create a new library called 'project-docs' with HNSW indexing"

> "Add this document to my knowledge base: [paste text]"

> "Use temperature search to find diverse results about machine learning"

> "Check the health of embeddings in my library"

## Development

```bash
# Run in development
cargo run -- --url http://localhost:8000 --debug

# Run tests
cargo test

# Build release
cargo build --release
```

## Protocol Details

This server implements the Model Context Protocol (MCP) specification version 2024-11-05, using:
- JSON-RPC 2.0 message format
- stdio transport
- Tools capability for function calling

## License

Apache 2.0
