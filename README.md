# rag_template
A ready-to-use Retrieval-Augmented Generation (RAG) system template designed for both developers and non-developers. Simply clone the project, add your own documents, and deploy an AI assistant that can search, understand, and answer questions using your data.


rag-system/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py
│   │   │
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── document_processor.py
│   │   │   ├── embedding_service.py
│   │   │   ├── vector_store.py
│   │   │   ├── llm_service.py
│   │   │   └── rag_agent.py 
│   │   │
│   │   ├── routes/ 
│   │   │   ├── __init__.py
│   │   │   ├── documents.py
│   │   │   └── chat.py
│   │   │
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── file_handlers.py
│   │
│   ├── requirements.txt
│   ├── .env.example
│   └── .env                                     (create this - gitignored)
│
├── frontend/
│   ├── public/
│   │   └── index.html
│   │
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.js
│   │   └── index.css
│   │
│   ├── package.json
│   └── package-lock.json                        (auto-generated)
│
├── data/
│   ├── uploads/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── vector_db/
│       └── .gitkeep
│
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── nginx.conf                               (optional - for frontend)
│
├── docs/
│   ├── USER_GUIDE.md
│   ├── DEPLOYMENT.md
│   └── AGENTIC_RAG.md                          ⭐ NEW - Explains agentic features
│
├── scripts/
│   ├── setup.sh
│   ├── reset_db.sh                             (optional)
│   └── backup.sh                               (optional)
│
├── tests/                                       (optional)
│   ├── test_embedding.py
│   ├── test_rag.py
│   └── test_api.py
│
├── .gitignore
├── .env.example                                 (root level)
├── README.md
├── QUICKSTART.md
├── docker-compose.yml
└── LICENSE                                      (optional)