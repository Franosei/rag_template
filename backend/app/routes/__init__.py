
"""
API Routes Package

This package contains all API endpoint definitions organized by
functionality. Each module defines a FastAPI router with related endpoints.

Modules:
    documents: Document upload, listing, and deletion endpoints
    chat: Chat/question answering endpoints with agentic RAG
"""

# Import routers for easy access
from app.routes.documents import router as documents_router
from app.routes.chat import router as chat_router

__all__ = ["documents_router", "chat_router"]