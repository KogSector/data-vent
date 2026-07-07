mod config;
mod services;
mod grpc_server;

use axum::{routing::{get, post}, Router, Json};
use envconfig::Envconfig;
use serde::{Deserialize, Serialize};
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
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load config
    dotenvy::from_filename_override(".env.map").ok();
    dotenvy::from_filename_override(".env.secret").ok();
    dotenvy::from_filename_override(".env.local").ok();
    
    tracing_subscriber::fmt::init();
    
    info!("Starting data-vent (Rust)");

    let config = Config::init_from_env().unwrap();
    
    // Initialize FalkorDB
    let falkordb_client = FalkorDBClient::new(&config).await?;
    let _ = falkordb_client.initialize_indexes().await;

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
    };

    // Start gRPC server in background
    let grpc_addr: SocketAddr = format!("{}:{}", config.grpc_host, config.grpc_port).parse()?;
    let grpc_retriever = retriever.clone();
    let grpc_decomposer = decomposer.clone();
    let grpc_dispatcher = dispatcher.clone();
    let grpc_aggregator = aggregator.clone();
    tokio::spawn(async move {
        if let Err(e) = grpc_server::start_grpc_server(
            grpc_addr,
            grpc_retriever,
            grpc_decomposer,
            grpc_dispatcher,
            grpc_aggregator,
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

    let addr: SocketAddr = format!("{}:{}", config.host, config.app_port).parse()?;
    
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
    source_ids: Option<Vec<String>>,
}
fn default_limit() -> usize { 20 }

async fn retrieve_handler(
    axum::extract::State(state): axum::extract::State<AppState>,
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

    let search_res = state.dispatcher.dispatch(all_chunks, &state.retriever).await;
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
