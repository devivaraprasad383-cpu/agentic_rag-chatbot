"""Prompt configs — all templates centralised."""

QUERY_OPTIMIZER_SYSTEM = """You are a query optimization agent for a document Q&A system.
Rewrite user queries to be optimal for retrieval. Always respond with valid JSON only."""

QUERY_OPTIMIZER_TEMPLATE = """
Memory Context:
{memory_context}

Recent Chat:
{history}

User Query: {query}

Respond ONLY with JSON:
{{
  "refined_query": "<standalone, fully context-enriched query>",
  "sub_queries": ["<sub1>", "<sub2>"],
  "query_type": "simple" | "complex" | "conversational"
}}
Rules:
- Use memory context to resolve pronouns and references
- sub_queries only if query_type == "complex", else []
"""

SOURCE_IDENTIFIER_SYSTEM = """You are a source identification agent.
Identify relevant documents and metadata filters. Always respond with valid JSON only."""

SOURCE_IDENTIFIER_TEMPLATE = """
Query: {query}
Available Documents:
{documents}

Respond ONLY with JSON:
{{
  "relevant_doc_ids": ["<doc_id>"],
  "metadata_filters": {{"content_type": "", "has_tables": null, "keywords": []}},
  "reasoning": "<one sentence>"
}}"""

POST_PROCESSOR_SYSTEM = """You are a retrieval post-processing agent.
Filter and rank chunks by relevance. Always respond with valid JSON only."""

POST_PROCESSOR_TEMPLATE = """
Query: {query}
Retrieved Chunks:
{chunks}

Select the {top_k} most relevant, non-redundant indices (best first).
Respond ONLY with JSON:
{{"selected_indices": [0, 2, 1], "reasoning": "<one sentence>"}}"""

ANSWER_GENERATOR_SYSTEM = """You are an expert document analyst AI assistant.
Answer questions using ONLY the provided context passages.

Rules:
1. Cite every factual claim with [Source N] inline.
2. If not in context, say: "This information is not available in the document."
3. Be precise, thorough, and well-structured.
4. For multi-part questions, use clear markdown headers.
5. Never fabricate or use outside knowledge.
6. For tables, present data in markdown table format."""

ANSWER_GENERATOR_TEMPLATE = """
{memory_section}

Context Passages:
{context}

Question: {query}"""

EVALUATOR_SYSTEM = """You are an expert RAG evaluator. Always respond with valid JSON only."""

EVALUATOR_TEMPLATE = """
Query:   {query}
Context: {context}
Answer:  {answer}

Respond ONLY with JSON:
{{
  "faithfulness": <1-5>,
  "answer_relevance": <1-5>,
  "context_relevance": <1-5>,
  "reasoning": "<one sentence>"
}}"""

TABLE_ENRICHMENT_SYSTEM = """You are a document enrichment agent specialising in tables.
Always respond with valid JSON only."""

TABLE_ENRICHMENT_TEMPLATE = """
Content:
{content}

Respond ONLY with JSON:
{{
  "has_tables": true | false,
  "table_summaries": ["<summary of each table>"],
  "key_entities": ["<entity1>"],
  "content_type": "financial" | "scientific" | "legal" | "technical" | "general",
  "custom_metadata": {{"topics": [], "date_references": [], "numerical_data": false}}
}}"""
