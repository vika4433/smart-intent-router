# Smart Intent Router - Architecture Refactor Summary

## Overview

The Smart Intent Router has been refactored to properly separate client-facing REST APIs from internal MCP tools, creating a hybrid architecture that serves both web clients and MCP clients efficiently.

## Architecture Changes

### Before (All MCP Tools)
- All APIs were exposed as MCP tools
- No separation between client-facing and internal functionality
- Web clients had to use MCP protocol for everything

### After (Hybrid Architecture)
- **FastAPI REST endpoints**: Client-facing APIs for web applications
- **MCP tools**: Internal orchestration and AI model management
- **Real-time support**: WebSockets and Server-Sent Events for live feedback

## API Separation

### 🌐 Client-Facing FastAPI Endpoints

These are REST APIs that web clients should use:

#### **Core Routing**
- `POST /route_request` - Rule-based request routing
- `POST /route_request_ai` - AI-driven intelligent routing

#### **Conversation Management**
- `POST /conversations` - Create new conversation
- `GET /conversations/{user_id}` - Get user's conversations
- `GET /conversations/{conversation_id}/messages` - Get conversation messages
- `DELETE /conversations/{conversation_id}` - Delete conversation
- `DELETE /conversations/{conversation_id}/messages` - Clear conversation messages
- `POST /conversations/{conversation_id}/context` - Get conversation context

#### **Session Management**
- `POST /sessions` - Create new session
- `GET /sessions/{session_id}` - Check session status

#### **System Health**
- `GET /health` - System health check

#### **Real-time Communication**
- `GET /stream_response/{conversation_id}` - Server-Sent Events streaming
- `WebSocket /ws/{conversation_id}` - Bidirectional real-time communication

### 🔧 Internal MCP Tools

These are for internal orchestration and AI agents:

#### **Core AI Tools**
- `send_to_llm` - Direct LLM communication
- `classify_intent` - Intent classification with context
- `detect_language` - Language detection
- `select_llm_model` - Model selection based on intent/language

#### **Model Management**
- `get_models_from_config_tool` - Get models from config
- `get_models_from_lm_studio_tool` - Get real-time LM Studio models
- `get_model_tool` - Get specific model by name
- `get_orchestrator_model_tool` - Get orchestrator model
- `health_check` - Internal health check

#### **Conversation Context**
- `get_conversation_context` - Get conversation context for AI

### 📚 MCP Resources

- `resource://smart-intent-router/models` - Available models list
- Configuration, routing rules, system status resources

## Real-time Communication Support

FastAPI provides excellent support for real-time server-to-client communication:

### 1. **Server-Sent Events (SSE)**
```javascript
// Client-side JavaScript
const eventSource = new EventSource('/stream_response/conv_123?message=Hello');
eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

### 2. **WebSockets**
```javascript
// Client-side JavaScript
const ws = new WebSocket('ws://localhost:8000/ws/conv_123');
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
ws.send(JSON.stringify({
    type: 'user_message',
    message: 'Hello, AI!',
    user_id: 'user123'
}));
```

### 3. **StreamingResponse**
For streaming LLM responses chunk by chunk:
```python
async def stream_llm_response():
    for chunk in llm_response_chunks:
        yield f"data: {json.dumps(chunk)}\\n\\n"
```

## Server Configuration

### Dual Server Setup
The refactored server runs two services simultaneously:

1. **FastAPI Server** (Port 8000) - Client-facing REST APIs
2. **MCP Server** (Port 3001) - Internal tool orchestration

### Starting the Server
```bash
cd smart-intent-router-server
python src/mcp_server/server.py
```

This starts both servers:
- FastAPI server: `http://localhost:8000`
- MCP server: `http://localhost:3001`

## Client Integration

### Web Client Usage
Web clients should use the FastAPI endpoints:

```python
import requests

# Route a request
response = requests.post('http://localhost:8000/route_request', json={
    'message': 'Hello, how can you help me?',
    'user_id': 'user123'
})

# Create a session
session_response = requests.post('http://localhost:8000/sessions', json={
    'user_id': 'user123'
})

# Get health status
health = requests.get('http://localhost:8000/health')
```

### MCP Client Usage
AI agents and internal orchestration use MCP tools:

```python
# Example MCP client code
from mcp import ClientSession

async with ClientSession() as session:
    # Use internal tools
    result = await session.call_tool("send_to_llm", {
        "model_name": "gpt-4",
        "user_message": "Analyze this data",
        "system_prompt": "You are a data analyst"
    })
```

## Benefits of This Architecture

### 🚀 **Performance**
- REST APIs are faster for simple requests
- MCP tools optimized for complex AI orchestration
- Real-time communication for live feedback

### 🔒 **Security**
- Clear separation of public vs internal APIs
- FastAPI middleware for CORS, authentication
- MCP tools not exposed to external clients

### 📱 **Client Compatibility**
- Web browsers can use standard REST/WebSocket APIs
- AI agents can use MCP protocol for advanced features
- Mobile apps get simple HTTP endpoints

### 🔧 **Maintainability**
- Clear separation of concerns
- FastAPI for client-facing features
- MCP for AI orchestration
- Independent scaling of each service

## File Structure

```
smart-intent-router-server/
├── src/mcp_server/
│   └── server.py          # Hybrid FastAPI + MCP server
├── requirements.txt       # Updated with FastAPI deps
└── ...existing files...
```

## Dependencies Added

```bash
fastapi              # Web framework
uvicorn[standard]    # ASGI server
websockets          # WebSocket support
```

## Future Enhancements

1. **Authentication**: Add JWT/OAuth to FastAPI endpoints
2. **Rate Limiting**: Implement request throttling
3. **Caching**: Add Redis for session/conversation caching
4. **Monitoring**: Add metrics and logging for both servers
5. **Load Balancing**: Scale FastAPI and MCP servers independently

## Testing the Setup

1. **Test FastAPI endpoints**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Test MCP server**:
   ```bash
   # Use MCP client to connect to localhost:3001
   ```

3. **Test WebSocket**:
   ```javascript
   // Use browser console to test WebSocket connection
   const ws = new WebSocket('ws://localhost:8000/ws/test');
   ```

This refactored architecture provides the best of both worlds: simple REST APIs for web clients and powerful MCP tools for AI orchestration, with full support for real-time communication.
