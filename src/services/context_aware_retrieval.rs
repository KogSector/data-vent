use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;

use crate::services::graph_algorithms::{GraphAlgorithms, PathResult};
use crate::services::intelligent_retriever::IntelligentRetriever;
use crate::services::parallel_search::ParallelSearchDispatcher;
use crate::services::query_decomposer::{QueryChunk, QueryDecomposer};
use crate::services::result_aggregator::{AggregatedResult, ResultAggregator};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextOptions {
    #[serde(default)]
    pub previous_chunk_ids: Vec<String>,
    #[serde(default = "default_focus_strategy")]
    pub focus_strategy: String, // "refine", "expand", "deep_dive"
    pub algorithm: Option<String>, // "bfs", "wcc", "betweenness", "pagerank"
}

fn default_focus_strategy() -> String {
    "refine".to_string()
}

pub struct ContextAwareRetriever {
    retriever: Arc<IntelligentRetriever>,
    decomposer: Arc<QueryDecomposer>,
    dispatcher: Arc<ParallelSearchDispatcher>,
    aggregator: Arc<ResultAggregator>,
    graph_algo: GraphAlgorithms,
}

impl ContextAwareRetriever {
    pub fn new(
        retriever: Arc<IntelligentRetriever>,
        decomposer: Arc<QueryDecomposer>,
        dispatcher: Arc<ParallelSearchDispatcher>,
        aggregator: Arc<ResultAggregator>,
        graph_algo: GraphAlgorithms,
    ) -> Self {
        Self {
            retriever,
            decomposer,
            dispatcher,
            aggregator,
            graph_algo,
        }
    }

    /// Stateless context-aware retrieval applying focus strategies
    pub async fn retrieve_with_context(
        &self,
        graph_name: &str,
        intent: &str,
        keywords: Vec<String>,
        context: ContextOptions,
        limit: usize,
    ) -> AggregatedResult {
        let start = std::time::Instant::now();
        info!(
            "context_retrieval_started strategy={} previous_seeds={} algorithm={:?}",
            context.focus_strategy,
            context.previous_chunk_ids.len(),
            context.algorithm
        );

        let decomp_res = self.decomposer.decompose(intent).await;
        let mut all_chunks = decomp_res.chunks;

        for kw in keywords {
            all_chunks.push(QueryChunk {
                text: kw.clone(),
                intent: "entity_lookup".to_string(),
                weight: 1.0,
                original_span: (0, 0),
                tokens: kw.split_whitespace().map(|s| s.to_string()).collect(),
            });
        }

        match context.focus_strategy.as_str() {
            "expand" => {
                // Expansion strategy: identify communities (WCC) or bridge nodes (Betweenness)
                if !context.previous_chunk_ids.is_empty() {
                    let wcc_map = self.graph_algo.get_wcc_components(graph_name, &context.previous_chunk_ids).await;
                    let bc_map = self.graph_algo.get_betweenness_scores(graph_name, &context.previous_chunk_ids, 64).await;

                    // Add seed exploration chunks
                    for id in &context.previous_chunk_ids {
                        let mut boost_weight = 0.8;
                        if let Some(&bc) = bc_map.get(id) {
                            if bc > 0.0 {
                                boost_weight += 0.2;
                            }
                        }
                        all_chunks.push(QueryChunk {
                            text: id.clone(),
                            intent: "community_expansion".to_string(),
                            weight: boost_weight,
                            original_span: (0, 0),
                            tokens: vec![id.clone()],
                        });
                    }
                    let _ = wcc_map;
                }
            }
            "deep_dive" => {
                // Deep dive strategy: expand relationships from previous seeds via BFS
                if !context.previous_chunk_ids.is_empty() {
                    let bfs_nodes = self.graph_algo.bfs_traversal(graph_name, &context.previous_chunk_ids, 3, limit * 2).await;
                    for node in bfs_nodes.iter().take(3) {
                        all_chunks.push(QueryChunk {
                            text: node.content.clone(),
                            intent: "deep_dive_context".to_string(),
                            weight: 0.7,
                            original_span: (0, 0),
                            tokens: node.content.split_whitespace().take(5).map(|s| s.to_string()).collect(),
                        });
                    }
                }
            }
            _ => {
                // "refine" strategy (default): focus on previous seeds with higher weight
                for id in &context.previous_chunk_ids {
                    all_chunks.push(QueryChunk {
                        text: id.clone(),
                        intent: "context_refinement".to_string(),
                        weight: 1.2,
                        original_span: (0, 0),
                        tokens: vec![id.clone()],
                    });
                }
            }
        }

        let search_res = self.dispatcher.dispatch(graph_name, all_chunks, &self.retriever).await;
        let mut agg_res = self.aggregator.aggregate(search_res, intent, limit);
        agg_res.aggregation_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        agg_res
    }

    /// Dedicated multi-hop path reasoning between two nodes
    pub async fn multi_hop_reasoning(
        &self,
        graph_name: &str,
        source_chunk_id: &str,
        target_chunk_id: &str,
        max_hops: usize,
        rel_types: Vec<String>,
    ) -> Vec<PathResult> {
        self.graph_algo
            .shortest_paths(graph_name, source_chunk_id, target_chunk_id, max_hops, 3, &rel_types)
            .await
    }
}
