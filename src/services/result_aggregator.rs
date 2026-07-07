use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tracing::info;

use crate::services::intelligent_retriever::SearchResult;
use crate::services::parallel_search::ParallelSearchResult;

pub struct ScoredChunk {
    pub chunk_id: String,
    pub content: String,
    pub final_score: f64,
    pub vector_score: f64,
    pub graph_score: f64,
    pub cross_chunk_boost: f64,
    pub chunk_type: String,
    pub source_id: String,
    pub document_id: String,
    pub metadata: serde_json::Value,
    pub matched_by_chunks: Vec<String>,
    pub _depth: i32,
}

pub struct AggregatedResult {
    pub chunks: Vec<ScoredChunk>,
    pub total_results: usize,
    pub unique_sources: usize,
    pub vector_matches: usize,
    pub graph_matches: usize,
    pub completion_reached: bool,
    pub aggregation_time_ms: f64,
}

pub struct ResultAggregator {
    max_results: usize,
    min_avg_score: f64,
    min_chunks_for_completion: usize,
    vector_weight: f64,
    graph_weight: f64,
    cross_chunk_weight: f64,
}

impl ResultAggregator {
    pub fn new(
        max_results: usize,
        min_avg_score: f64,
        min_chunks_for_completion: usize,
        vector_weight: f64,
        graph_weight: f64,
        cross_chunk_weight: f64,
    ) -> Self {
        Self {
            max_results,
            min_avg_score,
            min_chunks_for_completion,
            vector_weight,
            graph_weight,
            cross_chunk_weight,
        }
    }

    pub fn aggregate(
        &self,
        parallel_result: ParallelSearchResult,
        _original_query: &str,
        limit: usize,
    ) -> AggregatedResult {
        let start = Instant::now();
        let limit = if limit == 0 { self.max_results } else { limit };

        let mut chunk_map: HashMap<String, ChunkAccumulator> = HashMap::new();

        for chunk_result in &parallel_result.chunk_results {
            if chunk_result.error.is_some() {
                continue;
            }

            let chunk_text = &chunk_result.query_chunk.text;
            let chunk_weight = chunk_result.query_chunk.weight;

            for node in &chunk_result.vector_results {
                let acc = chunk_map.entry(node.chunk_id.clone()).or_insert_with(|| ChunkAccumulator::new(node.clone()));
                acc.vector_scores.push(node.score * chunk_weight);
                acc.matched_by.insert(chunk_text.clone());
            }

            for node in &chunk_result.graph_results {
                let acc = chunk_map.entry(node.chunk_id.clone()).or_insert_with(|| ChunkAccumulator::new(node.clone()));
                acc.graph_scores.push(node.score * chunk_weight);
                acc.matched_by.insert(chunk_text.clone());
            }
        }

        if chunk_map.is_empty() {
            return AggregatedResult {
                chunks: vec![],
                total_results: 0,
                unique_sources: 0,
                vector_matches: parallel_result.total_vector_hits,
                graph_matches: parallel_result.total_graph_hits,
                completion_reached: false,
                aggregation_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            };
        }

        let total_query_chunks = parallel_result.chunks_searched.max(1) as f64;
        let mut scored: Vec<ScoredChunk> = vec![];

        for (chunk_id, acc) in chunk_map {
            let best_vector = acc.vector_scores.iter().cloned().fold(0.0, f64::max);
            let best_graph = acc.graph_scores.iter().cloned().fold(0.0, f64::max);
            let cross_boost = acc.matched_by.len() as f64 / total_query_chunks;

            let final_score = best_vector * self.vector_weight
                + best_graph * self.graph_weight
                + cross_boost * self.cross_chunk_weight;

            scored.push(ScoredChunk {
                chunk_id,
                content: acc.node.content,
                final_score,
                vector_score: best_vector,
                graph_score: best_graph,
                cross_chunk_boost: cross_boost,
                chunk_type: acc.node.chunk_type,
                source_id: acc.node.source_id.clone(),
                document_id: acc.node.document_id,
                metadata: acc.node.metadata,
                matched_by_chunks: acc.matched_by.into_iter().collect(),
                _depth: acc.node.depth,
            });
        }

        scored.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        let completion = self.check_completion(&scored);
        let mut unique_sources_set = HashSet::new();
        for c in &scored {
            if !c.source_id.is_empty() {
                unique_sources_set.insert(c.source_id.clone());
            }
        }
        let unique_sources = unique_sources_set.len();
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let total_results = scored.len();

        info!("results_aggregated total={} unique_sources={} completion={} time_ms={:.2}", 
              total_results, unique_sources, completion, elapsed);

        AggregatedResult {
            chunks: scored,
            total_results, // Wait, total_results could mean the whole set before truncate? Python code does `total_results=len(scored)` which means after truncate.
            unique_sources,
            vector_matches: parallel_result.total_vector_hits,
            graph_matches: parallel_result.total_graph_hits,
            completion_reached: completion,
            aggregation_time_ms: elapsed,
        }
    }

    fn check_completion(&self, chunks: &[ScoredChunk]) -> bool {
        if chunks.len() < self.min_chunks_for_completion {
            return false;
        }
        if chunks.len() >= self.max_results {
            return true;
        }
        let sum: f64 = chunks.iter().map(|c| c.final_score).sum();
        let avg = sum / chunks.len().max(1) as f64;
        avg >= self.min_avg_score
    }
}

struct ChunkAccumulator {
    node: SearchResult,
    vector_scores: Vec<f64>,
    graph_scores: Vec<f64>,
    matched_by: HashSet<String>,
}

impl ChunkAccumulator {
    fn new(node: SearchResult) -> Self {
        Self {
            node,
            vector_scores: vec![],
            graph_scores: vec![],
            matched_by: HashSet::new(),
        }
    }
}
