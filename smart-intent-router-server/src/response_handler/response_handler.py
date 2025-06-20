def handle_response(raw_response: str, model: str, intent: str, language: str) -> dict:
    """
    Process and format the LLM response for the MCP client.
    """
    # Example: wrap everything in a consistent structure
    return {
        "model": model,
        "intent": intent,
        "language": language,
        "response": raw_response
    }