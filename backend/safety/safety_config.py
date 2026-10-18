"""
Safety Configuration — all thresholds, rules, and patterns.
Centralised so they're easy to tune without touching node logic.
"""

# ── Faithfulness gate ──────────────────────────────────────────────
MIN_FAITHFULNESS_SCORE    = 3    # block answers scoring below this
MIN_ANSWER_RELEVANCE      = 3    # block answers scoring below this

# ── Retrieval gate ────────────────────────────────────────────────
MIN_CHUNKS_REQUIRED       = 1    # need at least this many chunks
MIN_AVG_CHUNK_SCORE       = 0.20 # minimum avg cosine similarity

# ── Input guard ───────────────────────────────────────────────────
MAX_QUERY_LENGTH          = 1000  # characters
MIN_QUERY_LENGTH          = 3     # characters

# Prompt injection patterns (fast regex, no LLM needed)
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"you\s+are\s+now\s+(a|an)\s+\w+",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*you",
    r"<\s*system\s*>",
    r"forget\s+(everything|all)\s+(you\s+know|above)",
    r"act\s+as\s+(if\s+you\s+are|a)\s+",
    r"jailbreak",
    r"dan\s+mode",
]

# Out-of-scope patterns — queries clearly not about the document
OUT_OF_SCOPE_PATTERNS = [
    r"write\s+(me\s+)?(a\s+)?(poem|song|story|essay|code|email)",
    r"what\s+is\s+the\s+weather",
    r"tell\s+me\s+a\s+joke",
    r"who\s+is\s+(the\s+)?president",
    r"translate\s+(this|to)\s+",
    r"what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+",   # math questions
]

# ── Output guard fallback messages ────────────────────────────────
BLOCKED_HALLUCINATION_MSG = (
    "⚠️ **Answer blocked by safety system.**\n\n"
    "The generated answer contained claims that could not be "
    "verified against the document. Please rephrase your question "
    "or ask about a specific section of the document."
)

BLOCKED_RELEVANCE_MSG = (
    "⚠️ **Answer blocked — relevance too low.**\n\n"
    "The system could not generate a sufficiently relevant answer "
    "for your query. Try rephrasing or asking a more specific question."
)

BLOCKED_INJECTION_MSG = (
    "⚠️ **Query blocked by safety system.**\n\n"
    "Your query appears to contain instructions that could manipulate "
    "the AI system. Please ask a genuine question about the document."
)

BLOCKED_OUT_OF_SCOPE_MSG = (
    "⚠️ **Out of scope.**\n\n"
    "This question doesn't appear to be about the uploaded document. "
    "Please ask something specific to the document content."
)

BLOCKED_NO_CONTEXT_MSG = (
    "⚠️ **Insufficient context found.**\n\n"
    "The retrieval system could not find relevant passages for your query. "
    "Try rephrasing with different keywords, or check if this topic "
    "is covered in the document."
)

BLOCKED_EMPTY_QUERY_MSG = (
    "⚠️ **Invalid query.**\n\n"
    "Please enter a valid question about the document."
)
