# src/ingestion/__init__.py
"""
Ingestion Pipeline Module
=========================
Pipeline untuk ingest conversation dataset ke Temporal Knowledge Graph.
"""

from .episode_ingester import EpisodeIngester

__all__ = ["EpisodeIngester"]
