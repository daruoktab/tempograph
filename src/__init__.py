# src/__init__.py
"""
Temporal RAG System
===================
Sistem RAG dengan Temporal Knowledge Graph untuk chatbot berbahasa Indonesia.

Modules:
- config: Configuration management
- embedders: Multiple embedding model support (Gemini, HuggingFace)
- llm_providers: Multiple LLM provider support (Gemini, OpenRouter, HuggingFace local)
- ingestion: Pipeline untuk ingest conversation ke knowledge graph
- retrieval: Agentic retrieval dengan temporal filtering
- evaluation: Metrics dan evaluator untuk RAG performance
- graph_client: Neo4j/Graphiti wrapper
"""

__version__ = "0.1.0"
__author__ = "Daru"
