use std::time::Instant;
use tokio::time::timeout;
use tracing::{error, info, warn};

use crate::services::intelligent_retriever::{IntelligentRetriever, SearchResult};
use crate::services::query_decomposer::QueryChunk;

pub struct ChunkSearchResult {
    pub query_chunk: QueryChunk,
    pub vector_results: Vec<SearchResult>,
    pub graph_results: Vec<SearchResult>,
    pub search_time_ms: f64,
    pub error: Option<String>,
}

pub struct ParallelSearchResult {
    pub chunk_results: Vec<ChunkSearchResult>,
    pub total_vector_hits: usize,
    pub total_graph_hits: usize,
    pub total_time_ms: f64,
    pub chunks_searched: usize,
    pub chunks_failed: usize,
}

pub struct ParallelSearchDispatcher {
    per_chunk_timeout: f64,
    vector_top_k: usize,
    dfs_depth: usize,
    dfs_min_relevance: f64,
    dfs_max_results: usize,
}

impl ParallelSearchDispatcher {
    pub fn new(
        per_chunk_timeout: f64,
        vector_top_k: usize,
        dfs_depth: usize,
        dfs_min_relevance: f64,
        dfs_max_results: usize,
    ) -> Self {
        Self {
            per_chunk_timeout,
            vector_top_k,
            dfs_depth,
            dfs_min_relevance,
            dfs_max_results,
        }
    }

    pub async fn dispatch(
        &self,
        chunks: Vec<QueryChunk>,
        retriever: &IntelligentRetriever,
    ) -> ParallelSearchResult {
        let start = Instant::now();
        if chunks.is_empty() {
            return ParallelSearchResult {
                chunk_results: vec![],
                total_vector_hits: 0,
                total_graph_hits: 0,
                total_time_ms: 0.0,
                chunks_searched: 0,
                chunks_failed: 0,
            };
        }

        let mut vectors = vec![];
        for chunk in &chunks {
            let vec = retriever.vectorize_query(&chunk.text).await;
            vectors.push(vec);
        }

        let mut tasks = vec![];
        for (i, chunk) in chunks.into_iter().enumerate() {
            let vector = if i < vectors.len() { vectors[i].clone() } else { vec![] };
            let timeout_dur = std::time::Duration::from_secs_f64(self.per_chunk_timeout);
            
            // Cannot easily move self and retriever into a spawn without Arc.
            // Since we're in async, we can just execute sequentially for now, or use futures::future::join_all
            // Actually, we can use tokio::spawn if we Arc them, but for simplicity let's just do sequential for the rewrite MVP,
            // OR use futures::future::join_all without spawning.
            
            let fut = async move {
                let s = Instant::now();
                let mut chunk_res = ChunkSearchResult {
                    query_chunk: chunk,
                    vector_results: vec![],
                    graph_results: vec![],
                    search_time_ms: 0.0,
                    error: None,
                };
                
                let do_search = async {
                    let mut v_res = vec![];
                    if !vector.is_empty() {
                        v_res = retriever.vector_search(&vector, self.vector_top_k).await;
                    }
                    if v_res.is_empty() {
                        v_res = retriever.text_search(&chunk_res.query_chunk.text, self.vector_top_k).await;
                    }
                    
                    let mut g_res = vec![];
                    if !v_res.is_empty() {
                        let seed_ids: Vec<String> = v_res.iter().take(3).map(|r| r.chunk_id.clone()).collect();
                        g_res = retriever.dfs_traversal(&seed_ids, self.dfs_depth, self.dfs_min_relevance, self.dfs_max_results).await;
                    }
                    (v_res, g_res)
                };

                match timeout(timeout_dur, do_search).await {
                    Ok((v, g)) => {
                        chunk_res.vector_results = v;
                        chunk_res.graph_results = g;
                    }
                    Err(_) => {
                        chunk_res.error = Some("timeout".to_string());
                    }
                }
                chunk_res.search_time_ms = s.elapsed().as_secs_f64() * 1000.0;
                chunk_res
            };
            tasks.push(fut);
        }

        let raw_results = futures::future::join_all(tasks).await;

        let mut chunk_results = vec![];
        let mut total_vector = 0;
        let mut total_graph = 0;
        let mut failed = 0;
        let chunks_searched = raw_results.len();

        for res in raw_results {
            total_vector += res.vector_results.len();
            total_graph += res.graph_results.len();
            if res.error.is_some() {
                failed += 1;
            }
            chunk_results.push(res);
        }

        let total_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        info!("parallel_search_completed chunks={} vector_hits={} graph_hits={} failed={} time_ms={:.2}", 
              chunks_searched, total_vector, total_graph, failed, total_time_ms);

        ParallelSearchResult {
            chunk_results,
            total_vector_hits: total_vector,
            total_graph_hits: total_graph,
            total_time_ms,
            chunks_searched,
            chunks_failed: failed,
        }
    }
}
