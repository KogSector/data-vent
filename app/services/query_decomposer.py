"""
Data Vent - Query Decomposition Engine

Breaks user queries into meaningful semantic chunks for parallel search.
Uses lightweight rules-based NLP (no LLM calls) for low-latency decomposition.

Algorithm:
1. Preprocessing — normalize, clean input
2. Stop word removal
3. N-gram extraction — meaningful 1-3 word phrases
4. Entity recognition — detect technical terms, identifiers
5. Intent classification — entity_lookup, relationship_query, attribute_search
6. Weight assignment — importance scoring per chunk
"""

import re
import structlog
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set

logger = structlog.get_logger()


# ─── Stop words (common English + query filler words) ───────────────────────

STOP_WORDS: Set[str] = {
    # Articles & determiners
    "a", "an", "the", "this", "that", "these", "those",
    # Prepositions
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "into",
    "about", "between", "through", "during", "before", "after", "above",
    "below", "up", "down", "out", "off", "over", "under",
    # Conjunctions
    "and", "or", "but", "nor", "so", "yet", "both", "either", "neither",
    # Pronouns
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "who", "whom",
    # Auxiliary verbs
    "is", "am", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could",
    # Query filler
    "what", "how", "where", "when", "why", "which",
    "show", "tell", "give", "get", "find", "search",
    "please", "help", "need", "want", "like",
    "all", "any", "some", "each", "every", "no", "not",
    "just", "only", "also", "very", "really", "quite",
    "more", "most", "much", "many", "few", "less", "least",
    "then", "than", "too", "here", "there", "now",
}

# ─── Relationship indicators ────────────────────────────────────────────────

RELATIONSHIP_KEYWORDS: Set[str] = {
    "related", "connects", "linked", "associated", "depends",
    "references", "uses", "calls", "inherits", "implements",
    "contains", "belongs", "owns", "maps", "extends",
    "imports", "exports", "requires", "provides",
    "between", "relationship", "connection", "dependency",
    "parent", "child", "sibling", "ancestor", "descendant",
}

# ─── Action / attribute indicators ──────────────────────────────────────────

ATTRIBUTE_KEYWORDS: Set[str] = {
    "type", "name", "value", "status", "state", "count",
    "size", "length", "format", "version", "date", "time",
    "created", "updated", "modified", "deleted",
    "config", "configuration", "setting", "parameter",
    "property", "attribute", "field", "column",
    "description", "summary", "title", "label",
}


@dataclass
class QueryChunk:
    """A decomposed chunk of the original query."""

    text: str               # The chunk text (typically 1-3 meaningful words)
    intent: str             # entity_lookup | relationship_query | attribute_search | action_query
    weight: float           # Importance weight 0.0 - 1.0
    original_span: Tuple[int, int]  # (start, end) position in original query
    tokens: List[str] = field(default_factory=list)  # Individual tokens in this chunk


@dataclass
class DecompositionResult:
    """Result of query decomposition."""

    original_query: str
    chunks: List[QueryChunk]
    total_chunks: int
    decomposition_time_ms: float = 0.0


class QueryDecomposer:
    """
    Decomposes user queries into meaningful semantic chunks
    for parallel graph search.

    Uses lightweight rules-based NLP — no LLM dependency.
    Typical latency: < 5ms.
    """

    def __init__(
        self,
        max_chunks: int = 10,
        min_chunk_length: int = 2,
        max_ngram_size: int = 3,
        stop_words: Optional[Set[str]] = None,
    ):
        self.max_chunks = max_chunks
        self.min_chunk_length = min_chunk_length
        self.max_ngram_size = max_ngram_size
        self.stop_words = stop_words or STOP_WORDS

    async def decompose(self, query: str) -> DecompositionResult:
        """
        Break a user query into meaningful semantic chunks.

        Pipeline:
        1. Preprocess (normalize, clean)
        2. Extract entities (quoted strings, technical terms, identifiers)
        3. Extract n-grams from remaining text
        4. Classify intent for each chunk
        5. Assign weights
        6. Deduplicate and limit

        Returns DecompositionResult with ordered chunks.
        """
        import time
        start = time.perf_counter()

        if not query or not query.strip():
            return DecompositionResult(
                original_query=query or "",
                chunks=[],
                total_chunks=0,
            )

        # Step 1: Preprocess
        cleaned = self._preprocess(query)

        # Step 2: Extract entities (quoted strings, identifiers, technical terms)
        entities, remaining_text = self._extract_entities(cleaned, query)

        # Step 3: Extract meaningful n-grams from remaining text
        ngrams = self._extract_ngrams(remaining_text)

        # Step 4: Combine and classify intent
        all_chunks: List[QueryChunk] = []
        all_chunks.extend(entities)

        for text, span, tokens in ngrams:
            intent = self._classify_intent(tokens)
            all_chunks.append(QueryChunk(
                text=text,
                intent=intent,
                weight=0.0,  # assigned in next step
                original_span=span,
                tokens=tokens,
            ))

        # Step 5: Assign weights
        all_chunks = self._assign_weights(all_chunks, query)

        # Step 6: Deduplicate, sort by weight, limit
        all_chunks = self._deduplicate(all_chunks)
        all_chunks.sort(key=lambda c: c.weight, reverse=True)
        all_chunks = all_chunks[: self.max_chunks]

        elapsed_ms = (time.perf_counter() - start) * 1000

        result = DecompositionResult(
            original_query=query,
            chunks=all_chunks,
            total_chunks=len(all_chunks),
            decomposition_time_ms=round(elapsed_ms, 2),
        )

        logger.info(
            "query_decomposed",
            original=query[:100],
            chunks=len(all_chunks),
            time_ms=result.decomposition_time_ms,
        )

        return result

    # ─── Internal methods ───────────────────────────────────────────────

    def _preprocess(self, query: str) -> str:
        """Normalize and clean the query text."""
        text = query.strip()
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special chars except quotes, hyphens, underscores, dots, slashes
        text = re.sub(r"[^\w\s\"'\-_./]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_entities(
        self, cleaned: str, original: str
    ) -> Tuple[List[QueryChunk], str]:
        """
        Extract high-value entities from the query:
        - Quoted strings ("exact phrase")
        - snake_case or camelCase identifiers
        - Dotted paths (module.class.method)
        - UPPER_CASE constants
        """
        entities: List[QueryChunk] = []
        remaining = cleaned

        # 1. Quoted strings
        for match in re.finditer(r'"([^"]+)"', cleaned):
            text = match.group(1).strip()
            if len(text) >= self.min_chunk_length:
                start_pos = original.lower().find(text.lower())
                span = (start_pos, start_pos + len(text)) if start_pos >= 0 else (0, 0)
                entities.append(QueryChunk(
                    text=text,
                    intent="entity_lookup",
                    weight=1.0,  # Quoted = highest priority
                    original_span=span,
                    tokens=text.split(),
                ))
                remaining = remaining.replace(match.group(0), " ")

        # 2. Technical identifiers (snake_case, camelCase, dotted paths)
        for match in re.finditer(
            r"\b([a-zA-Z][a-zA-Z0-9]*(?:[._][a-zA-Z][a-zA-Z0-9]*)+)\b", remaining
        ):
            text = match.group(1)
            if len(text) >= self.min_chunk_length:
                start_pos = original.lower().find(text.lower())
                span = (start_pos, start_pos + len(text)) if start_pos >= 0 else (0, 0)
                entities.append(QueryChunk(
                    text=text,
                    intent="entity_lookup",
                    weight=0.95,
                    original_span=span,
                    tokens=[text],
                ))
                remaining = remaining.replace(text, " ", 1)

        # 3. UPPER_CASE constants
        for match in re.finditer(r"\b([A-Z][A-Z0-9_]{2,})\b", remaining):
            text = match.group(1)
            start_pos = original.find(text)
            span = (start_pos, start_pos + len(text)) if start_pos >= 0 else (0, 0)
            entities.append(QueryChunk(
                text=text,
                intent="entity_lookup",
                weight=0.9,
                original_span=span,
                tokens=[text],
            ))
            remaining = remaining.replace(text, " ", 1)

        remaining = re.sub(r"\s+", " ", remaining).strip()
        return entities, remaining

    def _extract_ngrams(
        self, text: str
    ) -> List[Tuple[str, Tuple[int, int], List[str]]]:
        """
        Extract meaningful 1-3 word n-grams after stop word removal.
        Returns list of (text, span, tokens).
        """
        words = text.lower().split()

        # Remove stop words but keep track of positions
        meaningful_words: List[Tuple[str, int]] = []
        for i, word in enumerate(words):
            # Strip punctuation from word edges
            clean_word = re.sub(r"^[^\w]+|[^\w]+$", "", word)
            if (
                clean_word
                and clean_word not in self.stop_words
                and len(clean_word) >= self.min_chunk_length
            ):
                meaningful_words.append((clean_word, i))

        if not meaningful_words:
            return []

        ngrams: List[Tuple[str, Tuple[int, int], List[str]]] = []

        # Generate n-grams from meaningful words
        # Prefer bigger n-grams (more context), then fill with unigrams
        used_indices: Set[int] = set()

        # Try trigrams first
        if self.max_ngram_size >= 3:
            for i in range(len(meaningful_words) - 2):
                w1, idx1 = meaningful_words[i]
                w2, idx2 = meaningful_words[i + 1]
                w3, idx3 = meaningful_words[i + 2]
                # Only group if they were adjacent or near-adjacent in original
                if idx2 - idx1 <= 2 and idx3 - idx2 <= 2:
                    tokens = [w1, w2, w3]
                    ngram_text = " ".join(tokens)
                    ngrams.append((ngram_text, (idx1, idx3), tokens))
                    used_indices.update({i, i + 1, i + 2})

        # Then bigrams for words not yet used
        if self.max_ngram_size >= 2:
            for i in range(len(meaningful_words) - 1):
                if i in used_indices and i + 1 in used_indices:
                    continue
                w1, idx1 = meaningful_words[i]
                w2, idx2 = meaningful_words[i + 1]
                if idx2 - idx1 <= 2:
                    tokens = [w1, w2]
                    ngram_text = " ".join(tokens)
                    ngrams.append((ngram_text, (idx1, idx2), tokens))
                    used_indices.update({i, i + 1})

        # Finally unigrams for remaining words
        for i, (word, idx) in enumerate(meaningful_words):
            if i not in used_indices:
                ngrams.append((word, (idx, idx), [word]))

        return ngrams

    def _classify_intent(self, tokens: List[str]) -> str:
        """Classify the intent of a chunk based on its tokens."""
        token_set = set(t.lower() for t in tokens)

        # Check for relationship indicators
        if token_set & RELATIONSHIP_KEYWORDS:
            return "relationship_query"

        # Check for attribute indicators
        if token_set & ATTRIBUTE_KEYWORDS:
            return "attribute_search"

        # Default: entity lookup (most common for graph search)
        return "entity_lookup"

    def _assign_weights(
        self, chunks: List[QueryChunk], original_query: str
    ) -> List[QueryChunk]:
        """
        Assign importance weights to chunks.

        Scoring factors:
        - Token count (longer phrases = more specific = higher weight)
        - Position in query (earlier = more important)
        - Intent type bonus
        - Entity detection bonus (already set for entities)
        """
        query_len = max(len(original_query), 1)

        for chunk in chunks:
            if chunk.weight > 0:
                # Already weighted (entities from _extract_entities)
                continue

            weight = 0.5  # Base weight

            # Token count bonus (multi-word = more specific)
            token_count = len(chunk.tokens)
            if token_count >= 3:
                weight += 0.2
            elif token_count == 2:
                weight += 0.1

            # Position bonus (earlier in query = slightly more important)
            if chunk.original_span[0] >= 0:
                position_ratio = 1.0 - (chunk.original_span[0] / query_len)
                weight += position_ratio * 0.1

            # Intent bonus
            if chunk.intent == "relationship_query":
                weight += 0.05
            elif chunk.intent == "entity_lookup":
                weight += 0.1

            # Word length bonus (longer words tend to be more specific)
            avg_word_len = sum(len(t) for t in chunk.tokens) / max(len(chunk.tokens), 1)
            if avg_word_len > 6:
                weight += 0.1

            chunk.weight = min(weight, 1.0)

        return chunks

    def _deduplicate(self, chunks: List[QueryChunk]) -> List[QueryChunk]:
        """Remove duplicate or overlapping chunks."""
        seen_texts: Set[str] = set()
        unique: List[QueryChunk] = []

        for chunk in chunks:
            normalized = chunk.text.lower().strip()
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique.append(chunk)

        return unique
