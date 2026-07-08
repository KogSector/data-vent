use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use crate::config::Config;
use crate::services::vector_search::FalkorDBClient;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk_id: String,
    pub content: String,
    pub score: f64,
    pub metadata: serde_json::Value,
    pub source: String,
    pub chunk_type: String,
    pub source_id: String,
    pub document_id: String,
    pub depth: i32,
    #[serde(default)]
    pub matched_by_chunks: Vec<String>,
}

pub struct IntelligentRetriever {
    falkordb_client: FalkorDBClient,
    http_client: reqwest::Client,
    gemini_api_key: String,
    gemini_base_url: String,
    embedding_model: String,
}

impl IntelligentRetriever {
    pub fn new(falkordb_client: FalkorDBClient, config: &Config) -> Self {
        Self {
            falkordb_client,
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(15))
                .build()
                .unwrap_or_default(),
            gemini_api_key: config.gemini_api_key.clone(),
            gemini_base_url: config.gemini_base_url.clone(),
            embedding_model: config.gemini_embedding_model.clone(),
        }
    }

    pub async fn _close(&self) {
        let _ = self.falkordb_client._close().await;
    }

    pub async fn vectorize_query(&self, query: &str) -> Vec<f64> {
        if self.gemini_api_key.is_empty() {
            warn!("GEMINI_API_KEY not set");
            return vec![];
        }

        let url = format!(
            "{}/v1/models/{}:embedContent?key={}",
            self.gemini_base_url, self.embedding_model, self.gemini_api_key
        );

        let body = serde_json::json!({
            "model": format!("models/{}", self.embedding_model),
            "content": {
                "parts": [{"text": query}]
            }
        });

        match self.http_client.post(&url).json(&body).send().await {
            Ok(resp) => {
                if let Ok(data) = resp.json::<serde_json::Value>().await {
                    if let Some(embedding) = data.get("embedding") {
                        if let Some(values) = embedding.get("values") {
                            if let Some(arr) = values.as_array() {
                                return arr.iter().filter_map(|v| v.as_f64()).collect();
                            }
                        }
                    }
                }
            }
            Err(e) => {
                error!("vectorize_query_failed: {}", e);
            }
        }
        vec![]
    }

    pub async fn vector_search(&self, graph_name: &str, query_vectors: &[f64], limit: usize) -> Vec<SearchResult> {
        if query_vectors.is_empty() {
            return vec![];
        }

        let embedding_str = serde_json::to_string(query_vectors).unwrap_or_default();
        let cypher = format!(
            "CALL db.idx.vector.queryNodes('Vector_Chunk', 'embeddings', {}, vecf32({})) YIELD node, score \
             RETURN node.id AS chunk_id, node.content AS content, node.chunk_type AS chunk_type, \
             node.source_id AS source_id, node.metadata AS metadata, score AS score \
             ORDER BY score DESC",
            limit, embedding_str
        );

        match self.falkordb_client.query(graph_name, &cypher).await {
            Ok(val) => self.parse_graph_results(&val, true),
            Err(e) => {
                error!("vector_search_failed: {}", e);
                vec![]
            }
        }
    }

    pub async fn text_search(&self, graph_name: &str, query: &str, limit: usize) -> Vec<SearchResult> {
        // Implement simple text search via regex for word boundaries and length > 2
        let words: Vec<&str> = query.split_whitespace().collect();
        let keywords: Vec<&str> = words.into_iter().filter(|w| w.len() > 2).collect();
        if keywords.is_empty() {
            return vec![];
        }
        let fts_query = keywords.join(" ");
        let cypher = format!(
            "CALL db.idx.fulltext.queryNodes('Vector_Chunk', '{}') YIELD node, score \
             RETURN node.id AS chunk_id, node.content AS content, node.chunk_type AS chunk_type, \
             node.source_id AS source_id, node.metadata AS metadata, score AS score \
             ORDER BY score DESC LIMIT {}",
            fts_query, limit
        );

        match self.falkordb_client.query(graph_name, &cypher).await {
            Ok(val) => self.parse_graph_results(&val, true),
            Err(e) => {
                error!("text_search_failed: {}", e);
                vec![]
            }
        }
    }

    pub async fn dfs_traversal(
        &self,
        graph_name: &str,
        start_chunk_ids: &[String],
        max_depth: usize,
        min_relevance: f64,
        max_results: usize,
    ) -> Vec<SearchResult> {
        if start_chunk_ids.is_empty() {
            return vec![];
        }

        let ids_str = start_chunk_ids
            .iter()
            .map(|id| format!("'{}'", id))
            .collect::<Vec<_>>()
            .join(", ");

        let cypher = format!(
            "MATCH path = (start:Vector_Chunk)-[*1..{}]-(n:Vector_Chunk) \
             WHERE start.id IN [{}] AND NOT n.id IN [{}] \
             RETURN DISTINCT n.id AS chunk_id, n.content AS content, n.chunk_type AS chunk_type, \
             n.source_id AS source_id, n.metadata AS metadata, length(path) AS depth \
             LIMIT {}",
            max_depth, ids_str, ids_str, max_results
        );

        match self.falkordb_client.query(graph_name, &cypher).await {
            Ok(val) => {
                let mut chunks = self.parse_graph_results(&val, false);
                for chunk in &mut chunks {
                    chunk.score = 0.1f64.max(1.0 - (chunk.depth as f64 * 0.2));
                }
                chunks.into_iter().filter(|c| c.score >= min_relevance).collect()
            }
            Err(e) => {
                error!("dfs_traversal_failed: {}", e);
                vec![]
            }
        }
    }

    fn parse_graph_results(&self, raw_result: &serde_json::Value, is_vector: bool) -> Vec<SearchResult> {
        let mut results = vec![];
        if let Some(arr) = raw_result.as_array() {
            if arr.len() >= 2 {
                let headers_raw = &arr[0];
                let rows_raw = &arr[1];

                let mut headers = vec![];
                if let Some(h_arr) = headers_raw.as_array() {
                    for h in h_arr {
                        if let Some(s) = h.as_str() {
                            headers.push(s.to_string());
                        } else {
                            headers.push(h.to_string());
                        }
                    }
                }

                if let Some(r_arr) = rows_raw.as_array() {
                    for row in r_arr {
                        if let Some(cols) = row.as_array() {
                            let mut chunk_id = String::new();
                            let mut content = String::new();
                            let mut chunk_type = String::new();
                            let mut source_id = String::new();
                            let mut metadata = serde_json::Value::Null;
                            let mut score = 0.0;
                            let mut depth = 0;

                            for (i, val) in cols.iter().enumerate() {
                                if i >= headers.len() {
                                    continue;
                                }
                                let col_name = &headers[i];
                                let v_str = val.as_str().unwrap_or("").to_string();

                                match col_name.as_str() {
                                    "chunk_id" => chunk_id = v_str,
                                    "content" => content = v_str,
                                    "chunk_type" => chunk_type = v_str,
                                    "source_id" => source_id = v_str,
                                    "metadata" => {
                                        if !v_str.is_empty() && v_str != "None" {
                                            if let Ok(m) = serde_json::from_str(&v_str) {
                                                metadata = m;
                                            }
                                        }
                                    }
                                    "score" if is_vector => {
                                        if let Some(n) = val.as_f64() {
                                            score = n;
                                        } else if let Ok(n) = v_str.parse::<f64>() {
                                            score = n;
                                        }
                                    }
                                    "depth" => {
                                        if let Some(n) = val.as_i64() {
                                            depth = n as i32;
                                        } else if let Ok(n) = v_str.parse::<i32>() {
                                            depth = n;
                                        }
                                    }
                                    _ => {}
                                }
                            }

                            results.push(SearchResult {
                                chunk_id,
                                content,
                                score,
                                metadata,
                                source: "falkordb".to_string(),
                                chunk_type,
                                document_id: source_id.clone(),
                                source_id,
                                depth,
                                matched_by_chunks: vec![],
                            });
                        }
                    }
                }
            }
        }
        results
    }

    pub async fn retrieve(
        &self,
        graph_name: &str,
        query: &str,
        _group_ids: Option<Vec<String>>,
        num_results: usize,
        _center_node_uuid: Option<String>,
    ) -> Vec<SearchResult> {
        let vector = self.vectorize_query(query).await;
        let mut results = vec![];

        if !vector.is_empty() {
            results = self.vector_search(graph_name, &vector, num_results).await;
        }

        if results.is_empty() {
            info!("Vector search returned 0 results, falling back to text search for: {}", query);
            results = self.text_search(graph_name, query, num_results).await;
        }
        results
    }
}
