//! HTTP client for arrwDB API

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// arrwDB HTTP client
#[derive(Clone)]
pub struct ArrwDBClient {
    client: Client,
    base_url: String,
    api_key: Option<String>,
}

// ============================================================================
// API Types
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct Library {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub index_type: String,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub title: String,
    pub author: Option<String>,
    pub tags: Vec<String>,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub text: String,
    pub distance: f32,
    pub document_id: String,
    pub chunk_index: usize,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub query_time_ms: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LibraryStats {
    pub total_documents: usize,
    pub total_chunks: usize,
    pub index_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub uptime_seconds: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IndexRecommendation {
    pub recommended_index: String,
    pub reasoning: String,
    pub workload_analysis: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingHealth {
    pub outlier_count: usize,
    pub degeneracy_score: f32,
    pub drift_detected: bool,
    pub details: Option<HashMap<String, serde_json::Value>>,
}

// ============================================================================
// Client Implementation
// ============================================================================

impl ArrwDBClient {
    pub fn new(base_url: &str, api_key: Option<String>) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
        })
    }

    fn build_request(&self, method: reqwest::Method, endpoint: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.base_url, endpoint);
        let mut request = self.client.request(method, &url);

        if let Some(ref key) = self.api_key {
            request = request.bearer_auth(key);
        }

        request
    }

    // ========================================================================
    // Health
    // ========================================================================

    pub async fn health_check(&self) -> Result<HealthStatus> {
        let response = self.build_request(reqwest::Method::GET, "/health")
            .send()
            .await
            .context("Failed to check health")?;

        response.json().await.context("Failed to parse health response")
    }

    // ========================================================================
    // Libraries
    // ========================================================================

    pub async fn list_libraries(&self) -> Result<Vec<Library>> {
        let response = self.build_request(reqwest::Method::GET, "/v1/libraries")
            .send()
            .await
            .context("Failed to list libraries")?;

        response.json().await.context("Failed to parse libraries response")
    }

    pub async fn get_library(&self, library_id: &str) -> Result<Library> {
        let response = self.build_request(
            reqwest::Method::GET,
            &format!("/v1/libraries/{}", library_id)
        )
            .send()
            .await
            .context("Failed to get library")?;

        response.json().await.context("Failed to parse library response")
    }

    pub async fn create_library(
        &self,
        name: &str,
        description: Option<&str>,
        index_type: &str,
    ) -> Result<Library> {
        let mut payload = serde_json::json!({
            "name": name,
            "index_type": index_type,
        });

        if let Some(desc) = description {
            payload["description"] = serde_json::Value::String(desc.to_string());
        }

        let response = self.build_request(reqwest::Method::POST, "/v1/libraries")
            .json(&payload)
            .send()
            .await
            .context("Failed to create library")?;

        response.json().await.context("Failed to parse create library response")
    }

    #[allow(dead_code)]
    pub async fn delete_library(&self, library_id: &str) -> Result<()> {
        self.build_request(
            reqwest::Method::DELETE,
            &format!("/v1/libraries/{}", library_id)
        )
            .send()
            .await
            .context("Failed to delete library")?;

        Ok(())
    }

    pub async fn get_library_stats(&self, library_id: &str) -> Result<LibraryStats> {
        let response = self.build_request(
            reqwest::Method::GET,
            &format!("/v1/libraries/{}/statistics", library_id)
        )
            .send()
            .await
            .context("Failed to get library statistics")?;

        response.json().await.context("Failed to parse statistics response")
    }

    // ========================================================================
    // Documents
    // ========================================================================

    pub async fn add_document(
        &self,
        library_id: &str,
        title: &str,
        texts: Vec<&str>,
        tags: Option<Vec<&str>>,
    ) -> Result<Document> {
        let payload = serde_json::json!({
            "title": title,
            "texts": texts,
            "tags": tags.unwrap_or_default(),
        });

        let response = self.build_request(
            reqwest::Method::POST,
            &format!("/v1/libraries/{}/documents", library_id)
        )
            .json(&payload)
            .send()
            .await
            .context("Failed to add document")?;

        response.json().await.context("Failed to parse document response")
    }

    // ========================================================================
    // Search
    // ========================================================================

    pub async fn search(
        &self,
        library_id: &str,
        query: &str,
        k: usize,
    ) -> Result<SearchResponse> {
        let payload = serde_json::json!({
            "query": query,
            "k": k,
        });

        let response = self.build_request(
            reqwest::Method::POST,
            &format!("/v1/libraries/{}/search", library_id)
        )
            .json(&payload)
            .send()
            .await
            .context("Failed to search")?;

        response.json().await.context("Failed to parse search response")
    }

    pub async fn temperature_search(
        &self,
        corpus_id: &str,
        query: &str,
        k: usize,
        temperature: f32,
    ) -> Result<SearchResponse> {
        let payload = serde_json::json!({
            "query_text": query,
            "k": k,
            "temperature": temperature,
        });

        let response = self.build_request(
            reqwest::Method::POST,
            &format!("/v1/temperature-search/corpora/{}/search", corpus_id)
        )
            .json(&payload)
            .send()
            .await
            .context("Failed to temperature search")?;

        response.json().await.context("Failed to parse temperature search response")
    }

    // ========================================================================
    // Novel Features
    // ========================================================================

    pub async fn get_index_recommendation(&self, corpus_id: &str) -> Result<IndexRecommendation> {
        let response = self.build_request(
            reqwest::Method::GET,
            &format!("/v1/index-oracle/corpora/{}/analyze", corpus_id)
        )
            .send()
            .await
            .context("Failed to get index recommendation")?;

        response.json().await.context("Failed to parse recommendation response")
    }

    pub async fn analyze_embedding_health(&self, corpus_id: &str) -> Result<EmbeddingHealth> {
        let response = self.build_request(
            reqwest::Method::GET,
            &format!("/v1/embedding-health/corpora/{}/analyze", corpus_id)
        )
            .send()
            .await
            .context("Failed to analyze embedding health")?;

        response.json().await.context("Failed to parse health response")
    }
}
