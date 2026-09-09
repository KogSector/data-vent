use envconfig::Envconfig;

#[allow(dead_code)]
#[derive(Envconfig, Debug, Clone)]
pub struct Config {
    #[envconfig(from = "DATA_VENT_PORT", default = "3002")]
    pub app_port: u16,
    
    #[envconfig(from = "HOST", default = "0.0.0.0")]
    pub host: String,
    
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
    
    #[envconfig(from = "FALKORDB_USE_TLS", default = "false")]
    pub falkordb_use_tls: bool,
    
    // Downstream Services
    #[envconfig(from = "NVIDIA_NIM_API_KEY")]
    pub nvidia_nim_api_key: Option<String>,
    
    #[envconfig(from = "NVIDIA_NIM_BASE_URL", default = "https://integrate.api.nvidia.com")]
    pub nvidia_nim_base_url: String,
    
    #[envconfig(from = "DEFAULT_EMBEDDING_MODEL", default = "nv-embed-v1")]
    pub default_embedding_model: String,
    
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

    // FalkorDB Graph Algorithms
    #[envconfig(from = "ENABLE_BFS", default = "true")]
    pub enable_bfs: bool,

    #[envconfig(from = "ENABLE_PAGERANK", default = "true")]
    pub enable_pagerank: bool,

    #[envconfig(from = "ENABLE_BETWEENNESS", default = "false")]
    pub enable_betweenness: bool,

    #[envconfig(from = "ENABLE_WCC", default = "false")]
    pub enable_wcc: bool,

    #[envconfig(from = "ENABLE_SPPATHS", default = "false")]
    pub enable_sppaths: bool,

    #[envconfig(from = "PAGERANK_BOOST_WEIGHT", default = "0.15")]
    pub pagerank_boost_weight: f64,

    // HNSW Configuration
    #[envconfig(from = "HNSW_M", default = "24")]
    pub hnsw_m: u32,

    #[envconfig(from = "HNSW_EF_CONSTRUCTION", default = "250")]
    pub hnsw_ef_construction: u32,

    #[envconfig(from = "HNSW_EF_RUNTIME", default = "25")]
    pub hnsw_ef_runtime: u32,

    #[envconfig(from = "HNSW_SIMILARITY_FUNCTION", default = "COSINE")]
    pub hnsw_similarity_function: String,

    #[envconfig(from = "HNSW_MODE", default = "balanced")]
    pub hnsw_mode: String,

    // Multi-Level Caching
    #[envconfig(from = "CACHE_EMBEDDING_CAPACITY", default = "1000")]
    pub cache_embedding_capacity: usize,

    #[envconfig(from = "CACHE_EMBEDDING_TTL_SECS", default = "3600")]
    pub cache_embedding_ttl_secs: u64,

    #[envconfig(from = "CACHE_VECTOR_CAPACITY", default = "1000")]
    pub cache_vector_capacity: usize,

    #[envconfig(from = "CACHE_VECTOR_TTL_SECS", default = "300")]
    pub cache_vector_ttl_secs: u64,

    #[envconfig(from = "CACHE_ALGORITHM_CAPACITY", default = "500")]
    pub cache_algorithm_capacity: usize,

    #[envconfig(from = "CACHE_ALGORITHM_TTL_SECS", default = "1800")]
    pub cache_algorithm_ttl_secs: u64,

    #[envconfig(from = "CACHE_RESULT_CAPACITY", default = "500")]
    pub cache_result_capacity: usize,

    #[envconfig(from = "CACHE_RESULT_TTL_SECS", default = "120")]
    pub cache_result_ttl_secs: u64,

    // Advanced Result Fusion
    #[envconfig(from = "FUSION_ALGORITHM", default = "wrrf")]
    pub fusion_algorithm: String,

    #[envconfig(from = "WRRF_K", default = "60.0")]
    pub wrrf_k: f64,
    
    #[envconfig(from = "LOG_LEVEL", default = "INFO")]
    pub log_level: String,
}
