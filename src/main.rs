mod infra;
mod services;

use axum::{
    response::sse::{Event, KeepAlive, Sse},
    routing::{get, post},
    Json, Router,
};
use envconfig::Envconfig;
use serde::Deserialize;
use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{error, info};

use infra::Config;
use services::cache::MultiLevelCache;
use services::context_aware_retrieval::{ContextAwareRetriever, ContextOptions};
use services::graph_algorithms::PathResult;
use services::intelligent_retriever::IntelligentRetriever;
use services::parallel_search::ParallelSearchDispatcher;
use services::query_decomposer::{QueryChunk, QueryDecomposer};
use services::result_aggregator::ResultAggregator;
use services::vector_search::FalkorDBClient;

#[derive(Clone)]
struct AppState {
    retriever: Arc<IntelligentRetriever>,
    decomposer: Arc<QueryDecomposer>,
    dispatcher: Arc<ParallelSearchDispatcher>,
    aggregator: Arc<ResultAggregator>,
    context_retriever: Arc<ContextAwareRetriever>,
    cache: Arc<MultiLevelCache>,
    default_graph_name: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Setup panic handler to catch and log panics
    std::panic::set_hook(Box::new(|panic_info| {
        tracing::error!("Panic occurred: {}", panic_info);
    }));

    // Load config
    tracing::info!("Loading environment variables...");
    dotenvy::from_filename_override(".env.map").ok();
    dotenvy::from_filename_override(".env.secret").ok();
    dotenvy::from_filename_override(".env.local").ok();
    tracing::info!("Environment variables loaded");

    tracing_subscriber::fmt::init();

    info!("Starting data-vent (Rust)");

    let config = Config::init_from_env().unwrap_or_else(|e| {
        tracing::warn!("Config error, using defaults: {}", e);
        Config {
            app_port: 3002,
            host: "0.0.0.0".to_string(),
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
            nvidia_nim_api_key: None,
            nvidia_nim_base_url: "https://integrate.api.nvidia.com".to_string(),
            default_embedding_model: "nv-embed-v1".to_string(),
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
            enable_bfs: true,
            enable_pagerank: true,
            enable_betweenness: false,
            enable_wcc: false,
            enable_sppaths: false,
            pagerank_boost_weight: 0.15,
            hnsw_m: 24,
            hnsw_ef_construction: 250,
            hnsw_ef_runtime: 25,
            hnsw_similarity_function: "COSINE".to_string(),
            hnsw_mode: "balanced".to_string(),
            cache_embedding_capacity: 1000,
            cache_embedding_ttl_secs: 3600,
            cache_vector_capacity: 1000,
            cache_vector_ttl_secs: 300,
            cache_algorithm_capacity: 500,
            cache_algorithm_ttl_secs: 1800,
            cache_result_capacity: 500,
            cache_result_ttl_secs: 120,
            fusion_algorithm: "wrrf".to_string(),
            wrrf_k: 60.0,
            log_level: "INFO".to_string(),
        }
    });

    // Initialize FalkorDB (optional - allow service to start without it)
    let falkordb_client = match FalkorDBClient::new(&config).await {
        Ok(client) => {
            info!("FalkorDB connected successfully");
            let _ = client
                .initialize_indexes(&config.falkordb_graph_name, config.falkordb_vector_dimension)
                .await;
            client
        }
        Err(e) => {
            error!(
                "Failed to connect to FalkorDB: {}. Service will start in degraded mode.",
                e
            );
            FalkorDBClient::new_dummy()
        }
    };

    // Initialize Multi-Level Cache
    let cache = Arc::new(MultiLevelCache::new(&config));

    // Initialize Services
    let retriever = Arc::new(IntelligentRetriever::new(falkordb_client, cache.clone(), &config));
    let decomposer = Arc::new(QueryDecomposer::new(config.pipeline_max_query_chunks));
    let dispatcher = Arc::new(ParallelSearchDispatcher::new(
        config.pipeline_per_chunk_timeout,
        config.pipeline_vector_top_k,
        config.pipeline_dfs_depth,
        config.pipeline_dfs_min_relevance,
        config.pipeline_dfs_max_results,
        config.enable_bfs,
    ));
    let aggregator = Arc::new(ResultAggregator::new(
        config.pipeline_max_total_results,
        config.pipeline_dfs_min_relevance,
        3,
        config.pipeline_vector_weight,
        config.pipeline_graph_weight,
        config.pipeline_cross_chunk_weight,
        config.fusion_algorithm.clone(),
        config.wrrf_k,
        config.pagerank_boost_weight,
    ));
    let context_retriever = Arc::new(ContextAwareRetriever::new(
        retriever.clone(),
        decomposer.clone(),
        dispatcher.clone(),
        aggregator.clone(),
        retriever.graph_algo.clone(),
    ));

    let state = AppState {
        retriever: retriever.clone(),
        decomposer: decomposer.clone(),
        dispatcher: dispatcher.clone(),
        aggregator: aggregator.clone(),
        context_retriever: context_retriever.clone(),
        cache: cache.clone(),
        default_graph_name: config.falkordb_graph_name.clone(),
    };

    // Define REST routes
    let app = Router::new()
        .route("/", get(|| async { axum::Json(serde_json::json!({"status": "ok"})) }))
        .route("/health", get(health_check))
        .route("/api/v1/retrieve", post(retrieve_handler))
        .route("/api/v1/retrieve/stream", post(retrieve_stream_handler))
        .route("/api/v1/retrieve/context", post(context_retrieve_handler))
        .route("/api/v1/retrieve/multi-hop", post(multi_hop_handler))
        .route("/api/v1/cache/invalidate", post(cache_invalidate_handler))
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
        "version": "0.3.0",
        "pipeline": "active",
        "algorithms": ["bfs", "sppaths", "pagerank", "betweenness", "wcc"],
        "fusion": "wrrf",
        "cache": "4-tier-lru"
    }))
}

#[derive(Deserialize, Debug, Clone)]
pub struct AlgorithmOptions {
    #[serde(default = "default_true")]
    pub enable_bfs: bool,
    #[serde(default = "default_true")]
    pub enable_pagerank: bool,
    #[serde(default)]
    pub enable_betweenness: bool,
    #[serde(default)]
    pub enable_wcc: bool,
    #[serde(default)]
    pub enable_sppaths: bool,
}
fn default_true() -> bool {
    true
}

#[derive(Deserialize, Debug, Clone)]
pub struct HnswRequestConfig {
    #[serde(default = "default_hnsw_mode")]
    pub mode: String,
}
fn default_hnsw_mode() -> String {
    "balanced".to_string()
}

#[derive(Deserialize)]
struct RetrieveRequest {
    intent: String,
    keywords: Vec<String>,
    #[serde(default = "default_limit")]
    limit: usize,
    _source_ids: Option<Vec<String>>,
    pub falkordb_graph_name: Option<String>,
    pub algorithms: Option<AlgorithmOptions>,
    pub hnsw_config: Option<HnswRequestConfig>,
}
fn default_limit() -> usize {
    20
}

#[derive(Deserialize)]
struct ContextRetrieveRequest {
    intent: String,
    #[serde(default)]
    keywords: Vec<String>,
    #[serde(default = "default_limit")]
    limit: usize,
    pub falkordb_graph_name: Option<String>,
    pub context: ContextOptions,
}

#[derive(Deserialize)]
struct MultiHopRequest {
    source_chunk_id: String,
    target_chunk_id: String,
    #[serde(default = "default_max_hops")]
    max_hops: usize,
    #[serde(default)]
    relationship_types: Vec<String>,
    pub falkordb_graph_name: Option<String>,
}
fn default_max_hops() -> usize {
    3
}

#[derive(Deserialize)]
struct CacheInvalidateRequest {
    pub graph_name: Option<String>,
}

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
        req.falkordb_graph_name.unwrap_or_else(|| state.default_graph_name.clone())
    };

    let search_res = state.dispatcher.dispatch(&graph_name, all_chunks, &state.retriever).await;

    // Optional PageRank & Betweenness Centrality boosting
    let candidate_ids: Vec<String> = search_res
        .chunk_results
        .iter()
        .flat_map(|cr| {
            cr.vector_results
                .iter()
                .chain(cr.graph_results.iter())
                .map(|r| r.chunk_id.clone())
        })
        .collect();

    let pagerank_scores = if req.algorithms.as_ref().map_or(true, |a| a.enable_pagerank) {
        state.retriever.get_pagerank_scores(&graph_name, &candidate_ids).await
    } else {
        HashMap::new()
    };

    let betweenness_scores = if req.algorithms.as_ref().map_or(false, |a| a.enable_betweenness) {
        state.retriever.get_betweenness_scores(&graph_name, &candidate_ids, 64).await
    } else {
        HashMap::new()
    };

    let agg_res = state.aggregator.aggregate_with_centrality(
        search_res,
        &req.intent,
        req.limit,
        &pagerank_scores,
        &betweenness_scores,
    );

    let mut results = vec![];
    for c in agg_res.chunks {
        results.push(serde_json::json!({
            "chunk_id": c.chunk_id,
            "content": c.content,
            "final_score": c.final_score,
            "vector_score": c.vector_score,
            "graph_score": c.graph_score,
            "cross_chunk_boost": c.cross_chunk_boost,
            "pagerank_score": c.pagerank_score,
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

async fn retrieve_stream_handler(
    axum::extract::State(state): axum::extract::State<AppState>,
    headers: axum::http::HeaderMap,
    Json(req): Json<RetrieveRequest>,
) -> Sse<impl futures::Stream<Item = Result<Event, Infallible>>> {
    let (mut tx, rx) = futures::channel::mpsc::channel::<Result<Event, Infallible>>(100);

    tokio::spawn(async move {
        let start = std::time::Instant::now();
        let decomp_res = state.decomposer.decompose(&req.intent).await;

        let mut all_chunks = decomp_res.chunks;
        for kw in &req.keywords {
            all_chunks.push(QueryChunk {
                text: kw.clone(),
                intent: "entity_lookup".to_string(),
                weight: 1.0,
                original_span: (0, 0),
                tokens: kw.split_whitespace().map(|s| s.to_string()).collect(),
            });
        }

        let chunk_infos: Vec<serde_json::Value> = all_chunks
            .iter()
            .map(|c| {
                serde_json::json!({
                    "text": c.text,
                    "intent": c.intent,
                    "weight": c.weight,
                })
            })
            .collect();

        let _ = tx.try_send(Ok(Event::default().event("query_decomposition").data(
            serde_json::json!({
                "query_chunks": chunk_infos,
                "decomposition_time_ms": decomp_res.decomposition_time_ms,
            })
            .to_string(),
        )));

        let graph_name = if let Some(user_id) = headers.get("x-user-id").and_then(|h| h.to_str().ok()) {
            format!("graph-{}", user_id)
        } else {
            req.falkordb_graph_name.unwrap_or_else(|| state.default_graph_name.clone())
        };

        let search_res = state.dispatcher.dispatch(&graph_name, all_chunks, &state.retriever).await;

        let candidate_ids: Vec<String> = search_res
            .chunk_results
            .iter()
            .flat_map(|cr| {
                cr.vector_results
                    .iter()
                    .chain(cr.graph_results.iter())
                    .map(|r| r.chunk_id.clone())
            })
            .collect();

        let pagerank_scores = if req.algorithms.as_ref().map_or(true, |a| a.enable_pagerank) {
            state.retriever.get_pagerank_scores(&graph_name, &candidate_ids).await
        } else {
            HashMap::new()
        };

        let agg_res = state.aggregator.aggregate_with_centrality(
            search_res,
            &req.intent,
            req.limit,
            &pagerank_scores,
            &HashMap::new(),
        );

        for (idx, c) in agg_res.chunks.iter().enumerate() {
            let chunk_data = serde_json::json!({
                "index": idx,
                "chunk_id": c.chunk_id,
                "content": c.content,
                "final_score": c.final_score,
                "vector_score": c.vector_score,
                "graph_score": c.graph_score,
                "cross_chunk_boost": c.cross_chunk_boost,
                "pagerank_score": c.pagerank_score,
                "chunk_type": c.chunk_type,
                "source_id": c.source_id,
                "document_id": c.document_id,
                "metadata": c.metadata,
                "matched_by_chunks": c.matched_by_chunks,
            });

            let _ = tx.try_send(Ok(Event::default().event("chunk_result").data(chunk_data.to_string())));
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let _ = tx.try_send(Ok(Event::default().event("done").data(
            serde_json::json!({
                "total_results": agg_res.total_results,
                "unique_sources": agg_res.unique_sources,
                "vector_matches": agg_res.vector_matches,
                "graph_matches": agg_res.graph_matches,
                "completion_reached": agg_res.completion_reached,
                "total_time_ms": elapsed,
            })
            .to_string(),
        )));
    });

    Sse::new(rx).keep_alive(KeepAlive::default())
}

async fn context_retrieve_handler(
    axum::extract::State(state): axum::extract::State<AppState>,
    headers: axum::http::HeaderMap,
    Json(req): Json<ContextRetrieveRequest>,
) -> Json<serde_json::Value> {
    let graph_name = if let Some(user_id) = headers.get("x-user-id").and_then(|h| h.to_str().ok()) {
        format!("graph-{}", user_id)
    } else {
        req.falkordb_graph_name.unwrap_or_else(|| state.default_graph_name.clone())
    };

    let agg_res = state
        .context_retriever
        .retrieve_with_context(&graph_name, &req.intent, req.keywords, req.context, req.limit)
        .await;

    let mut results = vec![];
    for c in agg_res.chunks {
        results.push(serde_json::json!({
            "chunk_id": c.chunk_id,
            "content": c.content,
            "final_score": c.final_score,
            "vector_score": c.vector_score,
            "graph_score": c.graph_score,
            "cross_chunk_boost": c.cross_chunk_boost,
            "pagerank_score": c.pagerank_score,
            "chunk_type": c.chunk_type,
            "source_id": c.source_id,
            "document_id": c.document_id,
            "metadata": c.metadata,
            "matched_by_chunks": c.matched_by_chunks,
        }));
    }

    Json(serde_json::json!({
        "results": results,
        "total_results": agg_res.total_results,
        "unique_sources": agg_res.unique_sources,
        "vector_matches": agg_res.vector_matches,
        "graph_matches": agg_res.graph_matches,
        "completion_reached": agg_res.completion_reached,
        "total_time_ms": agg_res.aggregation_time_ms,
    }))
}

async fn multi_hop_handler(
    axum::extract::State(state): axum::extract::State<AppState>,
    headers: axum::http::HeaderMap,
    Json(req): Json<MultiHopRequest>,
) -> Json<serde_json::Value> {
    let start = std::time::Instant::now();
    let graph_name = if let Some(user_id) = headers.get("x-user-id").and_then(|h| h.to_str().ok()) {
        format!("graph-{}", user_id)
    } else {
        req.falkordb_graph_name.unwrap_or_else(|| state.default_graph_name.clone())
    };

    let paths: Vec<PathResult> = state
        .context_retriever
        .multi_hop_reasoning(
            &graph_name,
            &req.source_chunk_id,
            &req.target_chunk_id,
            req.max_hops,
            req.relationship_types,
        )
        .await;

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    Json(serde_json::json!({
        "paths": paths,
        "total_paths": paths.len(),
        "source_chunk_id": req.source_chunk_id,
        "target_chunk_id": req.target_chunk_id,
        "total_time_ms": elapsed,
    }))
}

async fn cache_invalidate_handler(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(req): Json<CacheInvalidateRequest>,
) -> Json<serde_json::Value> {
    let graph = req.graph_name.unwrap_or_else(|| state.default_graph_name.clone());
    state.cache.invalidate_graph(&graph).await;
    Json(serde_json::json!({
        "status": "success",
        "message": format!("Cache invalidated for graph: {}", graph),
    }))
}
