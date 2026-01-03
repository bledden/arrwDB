//! MCP Server implementation for arrwDB

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::client::ArrwDBClient;
use crate::tools::{get_tools, Tool};

/// JSON-RPC request
#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

/// JSON-RPC response
#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

/// MCP Server capabilities
#[derive(Debug, Serialize)]
struct ServerCapabilities {
    tools: ToolsCapability,
}

#[derive(Debug, Serialize)]
struct ToolsCapability {
    #[serde(rename = "listChanged")]
    list_changed: bool,
}

/// MCP Server info
#[derive(Debug, Serialize)]
struct ServerInfo {
    name: String,
    version: String,
}


/// arrwDB MCP Server
pub struct ArrwDBServer {
    client: Arc<ArrwDBClient>,
    tools: Vec<Tool>,
}

impl ArrwDBServer {
    pub fn new(base_url: &str, api_key: Option<String>) -> Result<Self> {
        let client = ArrwDBClient::new(base_url, api_key)?;
        let tools = get_tools();

        Ok(Self {
            client: Arc::new(client),
            tools,
        })
    }

    /// Run the server using stdio transport
    pub async fn run_stdio(self) -> Result<()> {
        let stdin = std::io::stdin();
        let stdout = Arc::new(Mutex::new(std::io::stdout()));
        let reader = BufReader::new(stdin.lock());

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            tracing::debug!("Received: {}", line);

            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!("Failed to parse request: {}", e);
                    continue;
                }
            };

            let response = self.handle_request(request).await;

            let response_str = serde_json::to_string(&response)?;
            tracing::debug!("Sending: {}", response_str);

            let mut stdout = stdout.lock().await;
            writeln!(stdout, "{}", response_str)?;
            stdout.flush()?;
        }

        Ok(())
    }

    async fn handle_request(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        let id = request.id.unwrap_or(Value::Null);

        let result = match request.method.as_str() {
            "initialize" => self.handle_initialize(request.params).await,
            "initialized" => Ok(Value::Null), // Notification, no response needed
            "tools/list" => self.handle_tools_list().await,
            "tools/call" => self.handle_tools_call(request.params).await,
            "ping" => Ok(json!({})),
            _ => Err(anyhow!("Method not found: {}", request.method)),
        };

        match result {
            Ok(value) => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: Some(value),
                error: None,
            },
            Err(e) => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32603,
                    message: e.to_string(),
                    data: None,
                }),
            },
        }
    }

    async fn handle_initialize(&self, _params: Option<Value>) -> Result<Value> {
        Ok(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": ServerCapabilities {
                tools: ToolsCapability {
                    list_changed: false,
                },
            },
            "serverInfo": ServerInfo {
                name: "arrwdb-mcp".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        }))
    }

    async fn handle_tools_list(&self) -> Result<Value> {
        Ok(json!({
            "tools": self.tools,
        }))
    }

    async fn handle_tools_call(&self, params: Option<Value>) -> Result<Value> {
        let params = params.ok_or_else(|| anyhow!("Missing params"))?;
        let name = params["name"].as_str().ok_or_else(|| anyhow!("Missing tool name"))?;
        let arguments = params.get("arguments").cloned().unwrap_or(json!({}));

        let result = self.execute_tool(name, arguments).await;

        match result {
            Ok(text) => Ok(json!({
                "content": [{
                    "type": "text",
                    "text": text,
                }],
            })),
            Err(e) => Ok(json!({
                "content": [{
                    "type": "text",
                    "text": format!("Error: {}", e),
                }],
                "isError": true,
            })),
        }
    }

    async fn execute_tool(&self, name: &str, args: Value) -> Result<String> {
        match name {
            // Search tools
            "arrwdb_search" => {
                let library_id = args["library_id"].as_str()
                    .ok_or_else(|| anyhow!("Missing library_id"))?;
                let query = args["query"].as_str()
                    .ok_or_else(|| anyhow!("Missing query"))?;
                let k = args["k"].as_u64().unwrap_or(5) as usize;

                let results = self.client.search(library_id, query, k).await?;
                Ok(serde_json::to_string_pretty(&results)?)
            }

            "arrwdb_temperature_search" => {
                let library_id = args["library_id"].as_str()
                    .ok_or_else(|| anyhow!("Missing library_id"))?;
                let query = args["query"].as_str()
                    .ok_or_else(|| anyhow!("Missing query"))?;
                let k = args["k"].as_u64().unwrap_or(5) as usize;
                let temperature = args["temperature"].as_f64().unwrap_or(1.0) as f32;

                let results = self.client.temperature_search(library_id, query, k, temperature).await?;
                Ok(serde_json::to_string_pretty(&results)?)
            }

            // Library management
            "arrwdb_list_libraries" => {
                let libraries = self.client.list_libraries().await?;
                Ok(serde_json::to_string_pretty(&libraries)?)
            }

            "arrwdb_get_library" => {
                let library_id = args["library_id"].as_str()
                    .ok_or_else(|| anyhow!("Missing library_id"))?;
                let library = self.client.get_library(library_id).await?;
                Ok(serde_json::to_string_pretty(&library)?)
            }

            "arrwdb_create_library" => {
                let name = args["name"].as_str()
                    .ok_or_else(|| anyhow!("Missing name"))?;
                let description = args["description"].as_str();
                let index_type = args["index_type"].as_str().unwrap_or("brute_force");

                let library = self.client.create_library(name, description, index_type).await?;
                Ok(serde_json::to_string_pretty(&library)?)
            }

            "arrwdb_library_stats" => {
                let library_id = args["library_id"].as_str()
                    .ok_or_else(|| anyhow!("Missing library_id"))?;
                let stats = self.client.get_library_stats(library_id).await?;
                Ok(serde_json::to_string_pretty(&stats)?)
            }

            // Document management
            "arrwdb_add_document" => {
                let library_id = args["library_id"].as_str()
                    .ok_or_else(|| anyhow!("Missing library_id"))?;
                let title = args["title"].as_str()
                    .ok_or_else(|| anyhow!("Missing title"))?;
                let texts: Vec<&str> = args["texts"].as_array()
                    .ok_or_else(|| anyhow!("Missing texts array"))?
                    .iter()
                    .filter_map(|v| v.as_str())
                    .collect();
                let tags: Option<Vec<&str>> = args["tags"].as_array()
                    .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect());

                let document = self.client.add_document(library_id, title, texts, tags).await?;
                Ok(serde_json::to_string_pretty(&document)?)
            }

            // Novel features
            "arrwdb_index_oracle" => {
                let library_id = args["library_id"].as_str()
                    .ok_or_else(|| anyhow!("Missing library_id"))?;
                let recommendation = self.client.get_index_recommendation(library_id).await?;
                Ok(serde_json::to_string_pretty(&recommendation)?)
            }

            "arrwdb_embedding_health" => {
                let library_id = args["library_id"].as_str()
                    .ok_or_else(|| anyhow!("Missing library_id"))?;
                let health = self.client.analyze_embedding_health(library_id).await?;
                Ok(serde_json::to_string_pretty(&health)?)
            }

            "arrwdb_health_check" => {
                let health = self.client.health_check().await?;
                Ok(serde_json::to_string_pretty(&health)?)
            }

            _ => Err(anyhow!("Unknown tool: {}", name)),
        }
    }
}
