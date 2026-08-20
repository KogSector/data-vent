use serde_json::Value;
use tracing::info;

use crate::config::Config;

pub struct FalkorDBClient {
    _client: Option<redis::Client>,
    connection: Option<redis::aio::ConnectionManager>,
    is_dummy: bool,
}

impl FalkorDBClient {
    pub async fn new(config: &Config) -> anyhow::Result<Self> {
        let auth = if let Some(ref pwd) = config.falkordb_password {
            format!("{}:{}@", config.falkordb_username, pwd)
        } else {
            String::new()
        };
        
        let scheme = if config.falkordb_use_tls { "rediss" } else { "redis" };
        let url = format!("{}://{}{}:{}/{}", scheme, auth, config.falkordb_host, config.falkordb_port, config.falkordb_database);
        let client = redis::Client::open(url)?;
        
        // Add timeout for connection to handle Render free tier spin-up delays
        let connection = tokio::time::timeout(
            std::time::Duration::from_secs(30), // 30 second timeout for free tier spin-up
            redis::aio::ConnectionManager::new(client.clone())
        ).await
        .map_err(|_| anyhow::anyhow!("FalkorDB connection timed out after 30s (free tier spin-up?)"))??;
        
        Ok(Self {
            _client: Some(client),
            connection: Some(connection),
            is_dummy: false,
        })
    }

    pub fn new_dummy() -> Self {
        // Create a dummy client that will fail gracefully on queries
        // This allows the service to start even without FalkorDB
        Self {
            _client: None,
            connection: None,
            is_dummy: true,
        }
    }

    pub async fn initialize_indexes(&self, graph_name: &str, vector_dim: u16) -> anyhow::Result<()> {
        info!("Initializing FalkorDB indexes for graph: {} (dim: {})", graph_name, vector_dim);
        // Equivalent to db.idx.vector.createNodeIndex
        let cypher = format!("CALL db.idx.vector.createNodeIndex('Vector_Chunk', 'embeddings', {}, 'COSINE')", vector_dim);
        let _ = self.query(graph_name, &cypher).await;
        
        let fts_cypher = "CALL db.idx.fulltext.createNodeIndex('Vector_Chunk', 'content')";
        let _ = self.query(graph_name, fts_cypher).await;
        Ok(())
    }

    pub async fn query(&self, graph_name: &str, cypher: &str) -> anyhow::Result<Value> {
        if self.is_dummy || self.connection.is_none() {
            // Return empty result for dummy client
            return Ok(serde_json::json!({
                "results": [],
                "error": "FalkorDB not connected"
            }));
        }

        let conn = self.connection.as_ref().unwrap();
        let mut conn_clone = conn.clone();
        let result: redis::Value = redis::cmd("GRAPH.QUERY")
            .arg(graph_name)
            .arg(cypher)
            .arg("--compact")
            .query_async(&mut conn_clone)
            .await?;
            
        // We will return a rough JSON representation, but since parsing GRAPH.QUERY output is complex, 
        // we might need to write a custom parser in IntelligentRetriever.
        // For now we map it to string/Value loosely.
        let json_result = parse_redis_value_to_json(&result);
        Ok(json_result)
    }

    pub async fn _close(&self) -> anyhow::Result<()> {
        // Drop the multiplexed connection
        Ok(())
    }
}

// A helper to recursively turn a redis::Value into serde_json::Value
fn parse_redis_value_to_json(val: &redis::Value) -> Value {
    match val {
        redis::Value::Nil => Value::Null,
        redis::Value::Int(i) => Value::Number(serde_json::Number::from(*i)),
        redis::Value::Data(d) => {
            if let Ok(s) = std::str::from_utf8(d) {
                Value::String(s.to_string())
            } else {
                Value::String(format!("{:?}", d))
            }
        },
        redis::Value::Bulk(b) => {
            let vec: Vec<Value> = b.iter().map(parse_redis_value_to_json).collect();
            Value::Array(vec)
        },
        redis::Value::Status(s) => Value::String(s.clone()),
        redis::Value::Okay => Value::String("OK".to_string()),
    }
}
