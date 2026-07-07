use envconfig::Envconfig;

#[allow(dead_code)]
#[derive(Envconfig, Debug, Clone)]
pub struct Config {
    #[envconfig(from = "DATA_VENT_PORT", default = "3002")]
    pub app_port: u16,
    
    #[envconfig(from = "HOST", default = "0.0.0.0")]
    pub host: String,
    
    #[envconfig(from = "GRPC_PORT", default = "50051")]
    pub grpc_port: u16,
    
    #[envconfig(from = "GRPC_HOST", default = "0.0.0.0")]
    pub grpc_host: String,
    
    #[envconfig(from = "ENVIRONMENT", default = "production")]
    pub environment: String,
    
    // FalkorDB
    #[envconfig(from = "FALKORDB_HOST", default = "localhost")]
    pub falkordb_host: String,
    
    #[envconfig(from = "FALKORDB_PORT", default = "6379")]
    pub falkordb_port: u16,
    
    #[envconfig(from = "FALKORDB_USERNAME", default = "default")]
    pub falkordb_username: String,
    
    #[envconfig(from = "FALKORDB_PASSWORD")]
    pub falkordb_password: Option<String>,
    
    #[envconfig(from = "FALKORDB_DATABASE", default = "0")]
    pub falkordb_database: u16,
    
    #[envconfig(from = "FALKORDB_GRAPH_NAME", default = "confuse_graph")]
    pub falkordb_graph_name: String,
    
    #[envconfig(from = "FALKORDB_VECTOR_DIMENSION", default = "768")]
    pub falkordb_vector_dimension: u16,
    
    #[envconfig(from = "FALKORDB_SIMILARITY_THRESHOLD", default = "0.7")]
    pub falkordb_similarity_threshold: f64,
    
    #[envconfig(from = "FALKORDB_MAX_RESULTS", default = "10")]
    pub falkordb_max_results: u32,
    
    // Downstream Services
    #[envconfig(from = "EMBEDDINGS_GRPC_ADDR", default = "http://localhost:50052")]
    pub embeddings_grpc_addr: String,
    
    #[envconfig(from = "EMBEDDINGS_SERVICE_URL", default = "http://localhost:8000")]
    pub embeddings_service_url: String,
    
    #[envconfig(from = "GEMINI_API_KEY")]
    pub gemini_api_key: String,
    
    #[envconfig(from = "GEMINI_BASE_URL", default = "https://generativelanguage.googleapis.com")]
    pub gemini_base_url: String,
    
    #[envconfig(from = "GEMINI_EMBEDDING_MODEL", default = "text-embedding-004")]
    pub gemini_embedding_model: String,
    
    #[envconfig(from = "CLIENT_CONNECTOR_URL", default = "http://localhost:8001")]
    pub client_connector_url: String,
    
    #[envconfig(from = "CLIENT_CONNECTOR_GRPC_ADDR", default = "http://localhost:50053")]
    pub client_connector_grpc_addr: String,
    
    // Retrieval Pipeline
    #[envconfig(from = "PIPELINE_MAX_QUERY_CHUNKS", default = "5")]
    pub pipeline_max_query_chunks: usize,
    
    #[envconfig(from = "PIPELINE_PER_CHUNK_TIMEOUT", default = "5.0")]
    pub pipeline_per_chunk_timeout: f64,
    
    #[envconfig(from = "PIPELINE_VECTOR_TOP_K", default = "10")]
    pub pipeline_vector_top_k: usize,
    
    #[envconfig(from = "PIPELINE_DFS_DEPTH", default = "2")]
    pub pipeline_dfs_depth: usize,
    
    #[envconfig(from = "PIPELINE_DFS_MIN_RELEVANCE", default = "0.5")]
    pub pipeline_dfs_min_relevance: f64,
    
    #[envconfig(from = "PIPELINE_DFS_MAX_RESULTS", default = "20")]
    pub pipeline_dfs_max_results: usize,
    
    #[envconfig(from = "PIPELINE_MAX_TOTAL_RESULTS", default = "50")]
    pub pipeline_max_total_results: usize,
    
    #[envconfig(from = "PIPELINE_VECTOR_WEIGHT", default = "0.7")]
    pub pipeline_vector_weight: f64,
    
    #[envconfig(from = "PIPELINE_GRAPH_WEIGHT", default = "0.3")]
    pub pipeline_graph_weight: f64,
    
    #[envconfig(from = "PIPELINE_CROSS_CHUNK_WEIGHT", default = "0.1")]
    pub pipeline_cross_chunk_weight: f64,
    
    #[envconfig(from = "LOG_LEVEL", default = "INFO")]
    pub log_level: String,
}
