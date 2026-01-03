//! MCP Tool definitions for arrwDB

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Tool input schema following JSON Schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInputSchema {
    #[serde(rename = "type")]
    pub schema_type: String,
    pub properties: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

/// Tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: ToolInputSchema,
}

/// All available arrwDB tools
pub fn get_tools() -> Vec<Tool> {
    vec![
        // ====================================================================
        // Search Tools
        // ====================================================================
        Tool {
            name: "arrwdb_search".to_string(),
            description: "Search for documents in an arrwDB library using semantic similarity. \
                Returns the most relevant text chunks based on vector similarity to the query.".to_string(),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: serde_json::json!({
                    "library_id": {
                        "type": "string",
                        "description": "The UUID of the library to search in"
                    },
                    "query": {
                        "type": "string",
                        "description": "The search query text"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    }
                }),
                required: Some(vec!["library_id".to_string(), "query".to_string()]),
            },
        },
        Tool {
            name: "arrwdb_temperature_search".to_string(),
            description: "Search with temperature control for exploration vs exploitation. \
                Temperature 0.0 returns deterministic top results, higher values (up to 2.0) \
                introduce diversity for serendipitous discovery.".to_string(),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: serde_json::json!({
                    "library_id": {
                        "type": "string",
                        "description": "The UUID of the library to search in"
                    },
                    "query": {
                        "type": "string",
                        "description": "The search query text"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Temperature for result sampling (0.0=greedy, 2.0=exploratory)",
                        "default": 1.0,
                        "minimum": 0.0,
                        "maximum": 2.0
                    }
                }),
                required: Some(vec!["library_id".to_string(), "query".to_string()]),
            },
        },

        // ====================================================================
        // Library Management
        // ====================================================================
        Tool {
            name: "arrwdb_list_libraries".to_string(),
            description: "List all available libraries in arrwDB. Returns library IDs, names, \
                index types, and creation dates.".to_string(),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: serde_json::json!({}),
                required: None,
            },
        },
        Tool {
            name: "arrwdb_get_library".to_string(),
            description: "Get detailed information about a specific library including statistics.".to_string(),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: serde_json::json!({
                    "library_id": {
                        "type": "string",
                        "description": "The UUID of the library"
                    }
                }),
                required: Some(vec!["library_id".to_string()]),
            },
        },
        Tool {
            name: "arrwdb_create_library".to_string(),
            description: "Create a new library for storing documents and vectors. \
                Choose an index type based on your workload: 'brute_force' for small datasets, \
                'hnsw' for large datasets with fast approximate search.".to_string(),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: serde_json::json!({
                    "name": {
                        "type": "string",
                        "description": "Name of the library"
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description of the library"
                    },
                    "index_type": {
                        "type": "string",
                        "description": "Index type: brute_force, kd_tree, lsh, hnsw, or ivf",
                        "enum": ["brute_force", "kd_tree", "lsh", "hnsw", "ivf"],
                        "default": "brute_force"
                    }
                }),
                required: Some(vec!["name".to_string()]),
            },
        },
        Tool {
            name: "arrwdb_library_stats".to_string(),
            description: "Get statistics for a library including document count, chunk count, \
                and index information.".to_string(),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: serde_json::json!({
                    "library_id": {
                        "type": "string",
                        "description": "The UUID of the library"
                    }
                }),
                required: Some(vec!["library_id".to_string()]),
            },
        },

        // ====================================================================
        // Document Management
        // ====================================================================
        Tool {
            name: "arrwdb_add_document".to_string(),
            description: "Add a document with text chunks to a library. Embeddings are \
                generated automatically. Use this to index new content for search.".to_string(),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: serde_json::json!({
                    "library_id": {
                        "type": "string",
                        "description": "The UUID of the library"
                    },
                    "title": {
                        "type": "string",
                        "description": "Document title"
                    },
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of text chunks to embed and index"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags for filtering"
                    }
                }),
                required: Some(vec![
                    "library_id".to_string(),
                    "title".to_string(),
                    "texts".to_string()
                ]),
            },
        },

        // ====================================================================
        // Novel Features
        // ====================================================================
        Tool {
            name: "arrwdb_index_oracle".to_string(),
            description: "Get an intelligent recommendation for the optimal index type \
                based on the library's workload patterns. Analyzes query patterns and \
                data characteristics to suggest the best index.".to_string(),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: serde_json::json!({
                    "library_id": {
                        "type": "string",
                        "description": "The UUID of the library to analyze"
                    }
                }),
                required: Some(vec!["library_id".to_string()]),
            },
        },
        Tool {
            name: "arrwdb_embedding_health".to_string(),
            description: "Analyze the health of embeddings in a library. Detects outliers, \
                degeneracy (collapsed embeddings), and distribution drift over time.".to_string(),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: serde_json::json!({
                    "library_id": {
                        "type": "string",
                        "description": "The UUID of the library to analyze"
                    }
                }),
                required: Some(vec!["library_id".to_string()]),
            },
        },
        Tool {
            name: "arrwdb_health_check".to_string(),
            description: "Check if the arrwDB server is healthy and responding.".to_string(),
            input_schema: ToolInputSchema {
                schema_type: "object".to_string(),
                properties: serde_json::json!({}),
                required: None,
            },
        },
    ]
}
