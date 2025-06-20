def handle_response(raw_response: str, model: str, intent: str, language: str) -> dict:
    """
    Process and format the LLM response for the MCP client.

    NOTE: This function must remain stateless and thread-safe.
    Do not use or modify any global/shared state, to ensure safe operation with multiple concurrent clients.

    TODO: Extend this function to include response post-processing, error normalization, or metadata enrichment as needed.

    Args:
        raw_response (str): The raw response string returned from LM Studio.
        model (str): The identifier of the LM Studio model used.
        intent (str): The classified intent of the original request.
        language (str): The detected language code of the original request.

    Returns:
        dict: A structured response object for the MCP client.
    """
    return {
        "model": model,
        "intent": intent,
        "language": language,
        "response": raw_response
    }
