# SHL Assessment Recommendation Chatbot

## Overview

This project is a FastAPI-based conversational recommendation system for SHL assessments.

The system:
- recommends SHL assessments based on recruiter needs
- supports multi-turn conversations
- handles clarification questions
- compares assessments
- refuses off-topic queries
- uses semantic search with Sentence Transformers + FAISS

---

## Features

### 1. Semantic Retrieval
Uses sentence embeddings instead of keyword matching.

### 2. Conversational Context
Supports multi-turn recruiter conversations.

### 3. Clarification Handling
Requests additional details for vague queries.

### 4. Assessment Comparison
Supports comparison requests like:
- "Compare OPQ and GSA"

### 5. Off-topic Refusal
Rejects unrelated questions.

### 6. Conversation Completion
Detects conversation-ending responses.

---

## Tech Stack

- FastAPI
- Sentence Transformers
- FAISS
- Python

---

## API Endpoints

### Health Check

GET `/health`

Response:

```json
{
  "status": "ok"
}
```

---

### Chat Endpoint

POST `/chat`

Example request:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hiring Python backend developer"
    }
  ]
}
```

---

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run server:

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Retrieval Architecture

1. SHL catalog scraped into `catalog.json`
2. Sentence embeddings generated
3. FAISS index used for semantic similarity search
4. FastAPI serves conversational recommendations

---

## Author

Pooja