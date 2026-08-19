mod config;
mod services;
mod grpc_server;

use axum::{routing::{get, post}, Router, Json};
use envconfig::Envconfig;
use serde::Deserialize;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{info, error};

use config::Config;
use services::intelligent_retriever::IntelligentRetriever;
use services::parallel_search::ParallelSearchDispatcher;
use services::query_decomposer::QueryDecomposer;
use services::result_aggregator::ResultAggregator;
use services::vector_search::FalkorDBClient;
use services::query_decomposer::QueryChunk;

#[derive(Clone)]
struct AppState {
    retriever: Arc<IntelligentRetriever>,
    decomposer: Arc<QueryDecomposer>,
    dispatcher: Arc<ParallelSearchDispatcher>,
    aggregator: Arc<ResultAggregator>,
    default_graph_name: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load config
    dotenvy::from_filename_override(".env.map").ok();
    dotenvy::from_filename_override(".env.secret").ok();
    dotenvy::from_filename_override(".env.local").ok();
    
    tracing_subscriber::fmt::init();
    
    info!("Starting data-vent (Rust)");

    let config = Config::init_from_env().unwrap_or_else(|e| {
        tracing::warn!("Config error, using defaults: {}", e);
        Config {
            app_port: 3002,
            host: "0.0.0.0".to_string(),
            grpc_port: 50051,
            grpc_host: "0.0.0.0".to_string(),
            falkordb_host: "localhost".to_string(),
            falkordb_port: 6379,
            falkordb_username: "default".to_string(),
            falkordb_password: None,
            falkordb_database: 0,
            falkordb_graph_name: "confuse_graph".to_string(),
            falkordb_vector_dimension: 768,
            falkordb_similarity_threshold: 0.7,
            falkordb_max_results: 10,
            falkordb_use_tls: false,
            embeddings_grpc_addr: "embeddings-service:3011".to_string(),
            embeddings_service_url: "http://embeddings-service:3011".to_string(),
            nvidia_nim_api_key: None,
            nvidia_nim_base_url: "https://integrate.api.nvidia.com".to_string(),
            default_embedding_model: "nv-embed-v1".to_string(),
            client_connector_url: "http://client-connector:8095".to_string(),
            client_connector_grpc_addr: "client-connector:8095".to_string(),
            pipeline_max_query_chunks: 5,
            pipeline_per_chunk_timeout: 5.0,
            pipeline_vector_top_k: 10,
            pipeline_dfs_depth: 2,
            pipeline_dfs_min_relevance: 0.5,
            pipeline_dfs_max_results: 20,
            pipeline_max_total_results: 50,
            pipeline_vector_weight: 0.7,
            pipeline_graph_weight: 0.3,
            pipeline_cross_chunk_weight: 0.1,
            log_level: "INFO".to_string(),
        }
    });
    
    // Initialize FalkorDB (optional - allow service to start without it)
    let falkordb_client = match FalkorDBClient::new(&config).await {
        Ok(client) => {
            info!("FalkorDB connected successfully");
            let _ = client.initialize_indexes(&config.falkordb_graph_name, config.falkordb_vector_dimension).await;
            client
        }
        Err(e) => {
            error!("Failed to connect to FalkorDB: {}. Service will start in degraded mode.", e);
            // Create a dummy client that returns empty results
            FalkorDBClient::new_dummy()
        }
    };

    // Initialize Services
    let retriever = Arc::new(IntelligentRetriever::new(falkordb_client, &config));
    let decomposer = Arc::new(QueryDecomposer::new(config.pipeline_max_query_chunks));
    let dispatcher = Arc::new(ParallelSearchDispatcher::new(
        config.pipeline_per_chunk_timeout,
        config.pipeline_vector_top_k,
        config.pipeline_dfs_depth,
        config.pipeline_dfs_min_relevance,
        config.pipeline_dfs_max_results,
    ));
    let aggregator = Arc::new(ResultAggregator::new(
        config.pipeline_max_total_results,
        config.pipeline_dfs_min_relevance, // use this for min avg score approx
        3, // min chunks for completion
        config.pipeline_vector_weight,
        config.pipeline_graph_weight,
        config.pipeline_cross_chunk_weight,
    ));

    let state = AppState {
        retriever: retriever.clone(),
        decomposer: decomposer.clone(),
        dispatcher: dispatcher.clone(),
        aggregator: aggregator.clone(),
        default_graph_name: config.falkordb_graph_name.clone(),
    };

    // Start gRPC server in background
    let grpc_addr: SocketAddr = format!("{}:{}", config.grpc_host, config.grpc_port).parse()?;
    let grpc_retriever = retriever.clone();
    let grpc_decomposer = decomposer.clone();
    let grpc_dispatcher = dispatcher.clone();
    let grpc_aggregator = aggregator.clone();
    let grpc_default_graph_name = config.falkordb_graph_name.clone();
    tokio::spawn(async move {
        if let Err(e) = grpc_server::start_grpc_server(
            grpc_addr,
            grpc_retriever,
            grpc_decomposer,
            grpc_dispatcher,
            grpc_aggregator,
            grpc_default_graph_name,
        ).await {
            error!("gRPC server failed: {}", e);
        }
    });

    // Define REST routes
    let app = Router::new()
        .route("/", get(|| async { axum::Json(serde_json::json!({"status": "ok"})) }))
        .route("/health", get(health_check))
        .route("/api/v1/retrieve", post(retrieve_handler))
        .with_state(state);

    // Use PORT from environment (Render) or fall back to config
    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(config.app_port);
    
    let addr: SocketAddr = format!("{}:{}", config.host, port).parse()?;
    
    info!("HTTP server listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "data-vent",
        "version": "0.2.0",
        "pipeline": "active"
    }))
}

#[derive(Deserialize)]
struct RetrieveRequest {
    intent: String,
    keywords: Vec<String>,
    #[serde(default = "default_limit")]
    limit: usize,
    _source_ids: Option<Vec<String>>,
    pub falkordb_graph_name: Option<String>,
}
fn default_limit() -> usize { 20 }

async fn retrieve_handler(
    axum::extract::State(state): axum::extract::State<AppState>,
    headers: axum::http::HeaderMap,
    Json(req): Json<RetrieveRequest>,
) -> Json<serde_json::Value> {
    let start = std::time::Instant::now();
    let decomp_res = state.decomposer.decompose(&req.intent).await;
    
    let mut all_chunks = decomp_res.chunks;
    for kw in req.keywords {
        all_chunks.push(QueryChunk {
            text: kw.clone(),
            intent: "entity_lookup".to_string(),
            weight: 1.0,
            original_span: (0, 0),
            tokens: kw.split_whitespace().map(|s| s.to_string()).collect(),
        });
    }

    let graph_name = if let Some(user_id) = headers.get("x-user-id").and_then(|h| h.to_str().ok()) {
        format!("graph-{}", user_id)
    } else {
        req.falkordb_graph_name.unwrap_or(state.default_graph_name.clone())
    };

    let search_res = state.dispatcher.dispatch(&graph_name, all_chunks, &state.retriever).await;
    let agg_res = state.aggregator.aggregate(search_res, &req.intent, req.limit);

    let mut results = vec![];
    for c in agg_res.chunks {
        results.push(serde_json::json!({
            "chunk_id": c.chunk_id,
            "content": c.content,
            "final_score": c.final_score,
            "vector_score": c.vector_score,
            "graph_score": c.graph_score,
            "cross_chunk_boost": c.cross_chunk_boost,
            "chunk_type": c.chunk_type,
            "source_id": c.source_id,
            "document_id": c.document_id,
            "metadata": c.metadata,
            "matched_by_chunks": c.matched_by_chunks,
        }));
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    Json(serde_json::json!({
        "results": results,
        "total_results": agg_res.total_results,
        "unique_sources": agg_res.unique_sources,
        "vector_matches": agg_res.vector_matches,
        "graph_matches": agg_res.graph_matches,
        "completion_reached": agg_res.completion_reached,
        "total_time_ms": elapsed,
    }))
}
