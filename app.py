from fastapi import FastAPI
from pydantic import BaseModel
from retriever import search_assessments

app = FastAPI(
    title="SHL Assessment Recommendation Chatbot"
)


# -----------------------------
# Request Schema
# -----------------------------
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def root():
    return {
        "message": "SHL chatbot backend is running"
    }


# -----------------------------
# Health Endpoint
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# -----------------------------
# Chat Endpoint
# -----------------------------
@app.post("/chat")
def chat(request: ChatRequest):

    messages = request.messages

    # Latest user message
    latest_message = messages[-1].content.lower()

    # Full conversation context
    full_conversation = " ".join(
        [msg.content for msg in messages]
    ).lower()

    # -----------------------------
    # Comparison Detection
    # -----------------------------
    comparison_words = [
        "difference",
        "compare",
        "vs",
        "versus"
    ]

    is_comparison = any(
        word in latest_message
        for word in comparison_words
    )

    # -----------------------------
    # Off-topic Refusal
    # -----------------------------
    off_topic_keywords = [
        "salary",
        "legal",
        "law",
        "politics",
        "weather",
        "recipe",
        "football",
        "movie"
    ]

    for word in off_topic_keywords:
        if word in latest_message:
            return {
                "reply": "I can only help with SHL assessment recommendations and related comparisons.",
                "recommendations": [],
                "end_of_conversation": False
            }

    # -----------------------------
    # Conversation Completion
    # -----------------------------
    completion_words = [
        "perfect",
        "thanks",
        "thank you",
        "great",
        "that's what we need",
        "done"
    ]

    end_conversation = any(
        word in latest_message
        for word in completion_words
    )

    if end_conversation:
        return {
            "reply": "Glad I could help with your SHL assessment selection.",
            "recommendations": [],
            "end_of_conversation": True
        }

    # -----------------------------
    # Comparison Response
    # -----------------------------
    if is_comparison:

        results = search_assessments(
            latest_message,
            top_k=2
        )

        comparison_text = []

        for item in results:

            comparison_text.append(
                f"• {item.get('name', '')}\n"
                f"{item.get('description', '')[:250]}"
            )

        return {
            "reply": "Here is a comparison between the requested SHL assessments:\n\n"
                     + "\n\n".join(comparison_text),
            "recommendations": [],
            "end_of_conversation": False
        }

    # -----------------------------
    # Clarification Logic
    # -----------------------------
    vague_words = [
        "assessment",
        "test",
        "hiring",
        "job",
        "role"
    ]

    if (
        len(latest_message.split()) <= 4
        or latest_message in vague_words
    ):
        return {
            "reply": "Could you share the role, required skills, seniority level, or assessment type you are looking for?",
            "recommendations": [],
            "end_of_conversation": False
        }

    # -----------------------------
    # Search Assessments
    # -----------------------------
    results = search_assessments(
        full_conversation,
        top_k=5
    )

    recommendations = []

    for item in results:

        recommendations.append({
            "name": item.get("name", ""),
            "url": item.get("link", ""),
            "test_type": ", ".join(item.get("keys", []))
        })

    # -----------------------------
    # No Results Found
    # -----------------------------
    if not recommendations:
        return {
            "reply": "I could not find relevant SHL assessments for that request.",
            "recommendations": [],
            "end_of_conversation": False
        }

    # -----------------------------
    # Final Response
    # -----------------------------
    return {
        "reply": f"I found {len(recommendations)} SHL assessments matching your requirements.",
        "recommendations": recommendations,
        "end_of_conversation": False
    }