# Smart Intent Router

A sophisticated AI request routing system that combines **FastAPI + AI Agent + MCP** in a unified architecture. The system features an intelligent AI agent that orchestrates request routing, FastAPI for modern REST endpoints, and Model Context Protocol (MCP) for advanced tool capabilities. It intelligently classifies user intents, detects languages, and routes requests to the most appropriate language model using either rule-based or AI agent-driven orchestration.

## 🚀 Features

### Core Routing Intelligence
- **AI Agent Orchestration**: Intelligent AI agent that makes context-aware routing decisions using LLM reasoning
- **Dual Routing Modes**: Rule-based routing and AI agent-driven orchestration for maximum flexibility
- **Intent Classification**: Automatically classifies user requests into categories (code, math, translation, creative writing, general)
- **Language Detection**: Supports multilingual input with automatic language detection
- **Dynamic LLM Selection**: Routes requests to the best model based on intent, language, and context

### Advanced Architecture Stack
- **FastAPI Foundation**: Modern REST API framework with async support and automatic documentation
- **AI Agent Core**: Intelligent LLM-based orchestrator that reasons about routing decisions
- **MCP Integration**: Model Context Protocol for advanced tool capabilities and dynamic discovery
- **Real-Time Model Discovery**: Dynamically fetches available models from LM Studio in real-time
- **Conversation Management**: Persistent chat sessions with context-aware responses

### User Experience
- **Conversation History**: All conversations stored in MongoDB for persistence across sessions
- **Context-Aware Responses**: Smart context retrieval maintains conversation flow and relevance
- **Web Interface**: Modern Streamlit-based chat interface with markdown rendering
- **Cross-Platform Support**: VS Code tasks and launch configurations for Windows, macOS, and Linux
- **LM Studio Integration**: Seamless integration with LM Studio for local model hosting

## 🏗️ Architecture

### **FastAPI + AI Agent + MCP** Architecture

The Smart Intent Router implements a cutting-edge **three-layer architecture**:

#### 🌐 **FastAPI Layer** (Port 8000)
Modern REST API foundation:
- High-performance async HTTP endpoints
- WebSocket support for real-time communication
- Server-Sent Events (SSE) for live streaming
- Automatic OpenAPI documentation
- CORS support for cross-origin requests

#### 🤖 **AI Agent Layer** 
Intelligent orchestration core:
- **LLM-powered reasoning** for routing decisions
- **Context-aware analysis** of user requests and conversation history
- **Dynamic tool discovery** and capability assessment
- **Adaptive routing strategies** based on model performance and availability
- **Multi-step orchestration** for complex routing scenarios

#### 🔧 **MCP Layer** (Port 3001)
Advanced tool ecosystem:
- Model Context Protocol for standardized tool interfaces
- Dynamic tool and resource discovery
- Advanced conversation context management
- Real-time model discovery from LM Studio
- Extensible plugin architecture

### Components

The Smart Intent Router consists of two main components working together:

#### 1. **Web Client** (`web-client/`)
- **Streamlit-based chat interface** with real-time messaging
- **Markdown rendering** with syntax highlighting for code blocks
- **Session management** with persistent conversation history
- **REST API client** for communication with FastAPI endpoints
- **Dual routing support** - can use both rule-based and AI-driven routing

#### 2. **Server** (`smart-intent-router-server/`)
A unified **FastAPI + AI Agent + MCP** server architecture:

##### 🌐 FastAPI REST Layer:
- **Core Routing**: `/route_request` (rule-based), `/route_request_ai` (AI agent-driven)
- **Conversation Management**: CRUD operations for conversations and messages
- **Session Management**: User session creation and tracking
- **Real-time Communication**: WebSocket and SSE endpoints
- **Health Monitoring**: System status and model availability

##### 🤖 AI Agent Layer:
- **Intelligent Orchestrator**: LLM-powered routing decisions with reasoning capabilities
- **Context Analysis**: Deep understanding of conversation flow and user intent
- **Dynamic Strategy Selection**: Adapts routing approach based on request complexity
- **Multi-Model Coordination**: Orchestrates multiple specialist models for complex tasks
- **Learning & Adaptation**: Improves routing decisions based on interaction patterns

##### 🔧 MCP Tool Layer:
- **Tool Ecosystem**: `send_to_llm`, `classify_intent`, `detect_language`
- **Model Management**: `get_models_from_lm_studio`, `get_orchestrator_model`
- **Dynamic Discovery**: Real-time tool and resource enumeration
- **Context Management**: Conversation history and context retrieval

##### Core Components:

- **AI Agent Orchestrator** (`mcp_server/server.py`)
  - **LLM-powered reasoning engine** for intelligent routing decisions
  - **Context-aware analysis** of conversation history and user intent
  - **Dynamic capability assessment** of available models and tools
  - **Multi-step orchestration** for complex routing scenarios
  - **Configurable reasoning prompts** via `ai_router.system_prompt`
  - **Automatic tool and resource discovery** for MCP clients
  - **Adaptive routing strategies** based on model performance and availability

- **Intent Classifier** (`intent_classifier/`)
  - Analyzes user messages to determine intent type
  - Supports: `code`, `math`, `translation`, `creative writing`, `general`
  - Uses keyword-based classification with LLM fallback
  - Context-aware classification for follow-up questions

- **Language Detector** (`language_detector/`)
  - Automatically detects the language of user input
  - Supports multiple languages for multilingual routing

- **LLM Selector** (`llm_selector/`)
  - Dynamically selects the best model based on intent, language, and context
  - Configurable model mappings and fallback strategies
  - Real-time model availability validation

- **Real-Time Model Discovery** (`lm_studio_proxy/`)
  - Fetches available models directly from LM Studio `/v1/models` endpoint
  - Combines real-time data with configuration metadata
  - Intelligent fallback to configuration when LM Studio unavailable
  - Enhanced model validation and health monitoring

- **Conversation Manager** (`utils/conversation_manager.py`)
  - Manages persistent chat sessions and conversation history
  - Context-aware message filtering and token counting
  - Smart context retrieval with conversation summaries

##### Supporting Components:

- **Session Manager** (`utils/session_manager.py`)
  - User session management and tracking
  - Session expiration and cleanup

- **LM Studio Proxy** (`lm_studio_proxy/`)
  - HTTP client for LM Studio API integration
  - Request/response handling for local model inference
  - Real-time model discovery and availability checking

- **Response Handler** (`response_handler/`)
  - Processes and formats LLM responses
  - Handles different response types and error cases

### **FastAPI + AI Agent + MCP** Integration

The system runs as a **unified server** with three integrated layers:

1. **FastAPI Server** (Port 8000) - REST API foundation
2. **AI Agent Core** - Intelligent reasoning and orchestration  
3. **MCP Server** (Port 3001) - Advanced tool ecosystem

**Benefits of This Architecture**:
- 🧠 **Intelligence**: AI agent provides human-like reasoning for routing decisions
- 🚀 **Performance**: FastAPI delivers high-performance REST endpoints
- 🔧 **Extensibility**: MCP enables rich tool ecosystem and dynamic capabilities
- 🔒 **Security**: Clear separation between public APIs and internal AI reasoning
- 📱 **Compatibility**: Web clients use REST, AI systems use MCP, agent coordinates both
- 🎯 **Precision**: AI agent learns and adapts routing strategies over time

## 💾 Conversation History Management

The Smart Intent Router automatically **persists all conversation history in MongoDB**, ensuring that your chat sessions are preserved across application restarts and providing seamless conversation continuity.

### Key Features:
- **Persistent Storage**: All messages, user sessions, and conversation metadata are stored in MongoDB
- **Session Continuity**: Resume conversations exactly where you left off, even after restarting the application
- **Context Preservation**: The system maintains conversation context for more coherent multi-turn dialogues
- **Automatic Management**: No manual intervention required - conversations are automatically saved and retrieved

### Database Collections:
- **`conversations`**: Stores conversation metadata and message history
- **`sessions`**: Manages user session data and tracking
- **`messages`**: Individual message storage with timestamps and metadata

### Privacy & Data Management:
- Conversation data is stored locally in your MongoDB instance by default
- Use the `delete_messages` MCP tool to clear conversation history when needed
- Configure retention policies in MongoDB for automatic data cleanup if desired

## 📋 Configuration Schema

The system is configured via `config/smart_intent_router_config.yaml`. This is a customizable template that you can modify based on your available models, supported languages, and specific requirements.

### Configuration Parameters

#### Intent Classification
- **`intents`**: Array of supported intent categories (e.g., `code`, `math`, `translation`, `creative_writing`, `general`)
- Users can add or modify intent types based on their use cases

#### Language Support  
- **`languages`**: Array of supported language codes (e.g., `en`, `es`, `fr`, `de`, `ru`)
- Add language codes for multilingual support

#### Model Configuration
Each model in the `models` array requires:

- **`model_name`**: Exact model name as it appears in LM Studio
- **`endpoint`**: API endpoint URL (typically `http://localhost:1234/v1/chat/completions`)
- **`context_length`**: Maximum token limit for the model
- **`weight`**: Priority value for model selection (higher = higher priority)
- **`supported_intents`**: Array of intents this model can handle
- **`supported_languages`**: Array of languages this model supports
- **`enabled`**: Boolean to enable/disable the model
- **`is_orchestrator`**: Boolean to mark as an orchestrator model (for AI-driven routing)

#### Orchestrator Model Selection
Models can be designated as orchestrators for AI-driven routing:
- **`is_orchestrator: true`**: Marks a model as capable of intelligent routing decisions
- **AI Router System Prompt**: Configurable instructions for how the orchestrator should behave
- **Automatic Selection**: The system automatically selects the first enabled orchestrator model
- **Fallback Strategy**: If no orchestrator is available, falls back to rule-based routing

#### Model Selection Logic
The system supports multiple routing strategies:

##### 1. **Rule-Based Routing** (Traditional)
When multiple models are available for the same intent and language:
1. **Weight-based selection**: The model with the highest `weight` value is automatically chosen
2. **Fallback order**: If weights are equal, models are prioritized by configuration order
3. **Default handling**: Falls back to `default_model` if no suitable match is found

##### 2. **AI Agent-Driven Routing** (New)
Uses an intelligent AI agent to make sophisticated routing decisions:
1. **AI Agent Selection**: Automatically selects the first enabled model with `is_orchestrator: true`
2. **Reasoning-Based Decisions**: The AI agent analyzes conversation context, intent complexity, and model capabilities using LLM reasoning
3. **Dynamic Tool Discovery**: The agent can discover and utilize available tools and models in real-time
4. **Multi-Step Orchestration**: Can break down complex requests into multiple routing steps
5. **Adaptive Learning**: Improves routing strategies based on interaction patterns and outcomes
6. **Intelligent Fallback**: Falls back to rule-based routing if no orchestrator agent is available

##### 3. **Real-Time Model Discovery**
The system now fetches available models directly from LM Studio:
- **Live Model List**: Queries LM Studio's `/v1/models` endpoint for currently loaded models
- **Configuration Enrichment**: Combines real-time data with configuration metadata
- **Intelligent Fallback**: Uses configuration-based model list when LM Studio is unavailable
- **Health Monitoring**: Tracks LM Studio connectivity and model availability status

#### System Templates
- **`system_templates`**: Custom system messages for each intent type
- Provides context-specific prompting for better responses

#### AI Router Configuration
- **`ai_router.system_prompt`**: System prompt for AI-driven routing
- Defines how the orchestrating LLM should behave and use tools/resources
- Customizable instructions for model selection and tool usage

#### Database Settings
- **MongoDB configuration** for persistent storage
- Configurable database and collection names
- Connection string for local or remote MongoDB instances

    
# AI Router configuration for AI agent behavior
ai_router:
  system_prompt: |
    You are an INTELLIGENT AI AGENT ORCHESTRATOR. Your job is to analyze user requests 
    and make intelligent routing decisions to the most appropriate specialist model.
    
    AI AGENT REASONING PROCESS:
    1. Analyze user intent and conversation context
    2. Assess available models and their capabilities  
    3. Make intelligent routing decision based on reasoning
    4. Execute routing via function calls:
       - classify_intent(message="user's exact message")
       - detect_language(prompt="user's exact message") 
       - get_models()
       - send_to_llm(model_name="selected_model", user_message="user's exact message")
    
system_templates:
  code: "You are a coding assistant..."  # Customize for your needs
  # Add templates for each intent
```

#### Intent Templates
- **`system_templates`**: Custom system messages for each intent type
- Provides context-specific prompting for better responses

#### AI Router Configuration
- **`ai_router.system_prompt`**: System prompt for AI-driven routing
- Defines how the orchestrating LLM should behave and use tools/resources
- Customizable instructions for model selection and tool usage

#### Database Settings
- **MongoDB configuration** for persistent storage
- Configurable database and collection names
- Connection string for local or remote MongoDB instances

## 🛠️ API Reference

### FastAPI REST Endpoints (Port 8000)

#### Core Routing
- **`POST /route_request`** - Rule-based request routing
- **`POST /route_request_ai`** - AI-driven intelligent routing with orchestrator model

#### Conversation Management
- **`POST /conversations`** - Create new conversation
- **`GET /conversations/{user_id}`** - Get user's conversations
- **`GET /conversations/{conversation_id}/messages`** - Get conversation messages
- **`DELETE /conversations/{conversation_id}`** - Delete conversation
- **`DELETE /conversations/{conversation_id}/messages`** - Clear conversation messages

#### Session Management
- **`POST /sessions`** - Create new user session
- **`GET /sessions/{session_id}`** - Check session status

#### Real-time Communication
- **`GET /stream_response/{conversation_id}`** - Server-Sent Events streaming
- **`WebSocket /ws/{conversation_id}`** - Bidirectional real-time communication

#### System Health
- **`GET /health`** - System health check with model availability

### MCP Tools (Port 3001)

#### AI Orchestration Tools
- **`send_to_llm`** - Direct LLM communication
- **`classify_intent`** - Intent classification with conversation context
- **`detect_language`** - Language detection
- **`get_models_from_lm_studio`** - Real-time model discovery from LM Studio
- **`get_models_from_config`** - Get models from configuration
- **`get_orchestrator_model`** - Get designated orchestrator model
- **`route_request_ai`** - AI-driven routing orchestration

#### System Management Tools
- **`health_check`** - Internal health monitoring with LM Studio connectivity
- **`get_conversation_context`** - Retrieve conversation context for AI

#### Legacy Tools (Maintained for Compatibility)
- **`route_request`** - Rule-based routing
- **`create_session`** - Create new user session
- **`create_conversation`** - Create new conversation
- **`get_conversations`** - Retrieve user conversations
- **`get_messages`** - Retrieve conversation messages
- **`delete_messages`** - Clear conversation history

## 🚦 Getting Started

### Prerequisites
- Python 3.8+
- MongoDB (local or remote)
- LM Studio (for local model hosting)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/vika4433/smart-intent-router.git
   cd smart-intent-router
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
   note: install python 3.11.2

3. **Download and configure models in LM Studio**
   - Download LM Studio from [lmstudio.ai](https://lmstudio.ai)
   - Install and open LM Studio
   - Download your preferred models from the model repository:
     - For coding: Qwen2.5-Coder, CodeLlama, or similar
     - For general tasks: Llama 3.2, Mistral, or similar  
     - For math: WizardMath or specialized math models
   - Start the LM Studio local server:
     - Go to "Local Server" tab in LM Studio
     - Load your downloaded model
     - Click "Start Server" (default port: 1234)
   - **Important**: Note the exact model names as they appear in LM Studio - you'll need these for configuration

4. **Set up MongoDB**
   - Install MongoDB locally or use MongoDB Atlas
   - For local installation: `brew install mongodb-community` (macOS) or follow [MongoDB installation guide](https://docs.mongodb.com/manual/installation/)
   - Start MongoDB service (default connection: `mongodb://localhost:27017`)

5. **Configure the system**
   - Edit `config/smart_intent_router_config.yaml` with your setup:
     - Update `model_name` fields with exact names from LM Studio
     - Verify endpoints (default: `http://localhost:1234/v1/chat/completions`)
     - Set model weights: higher values (e.g., 2.0) = higher priority
     - Configure supported intents and languages for each model
     - Update database connection string if needed

6. **Run the system**
   
   The system now runs both FastAPI and MCP servers simultaneously. You can start them in several ways:

   **Option A: Using VS Code (Recommended)**
   - Open the project in VS Code
   - Use the "Run Server and Web Client" compound launch configuration
   - This will start both servers and the web client automatically

   **Option B: Manual startup**
   ```bash
   # Start the hybrid server (FastAPI + MCP)
   cd smart-intent-router-server
   python src/mcp_server/server.py
   
   # In another terminal, start the web client
   cd web-client
   streamlit run src/app.py
   ```

   **Option C: Using terminal scripts**
   ```bash
   # Start server
   ./scripts/start_server.sh
   
   # Start client
   ./scripts/start_client.sh
   ```

7. **Access the application**
   - **Web Interface**: Open your browser to `http://localhost:8501`
   - **FastAPI Server**: Available at `http://localhost:8000`
   - **MCP Server**: Available at `http://localhost:3001` (for AI agents)
   - Start chatting with your intelligent router!

### Routing Modes

The system supports two routing modes that you can choose between:

#### 1. **Rule-Based Routing** (`/route_request`)
- Traditional intent → model mapping
- Fast and predictable
- Uses configuration rules and model weights
- Good for simple, well-defined routing scenarios

#### 2. **AI Agent-Driven Routing** (`/route_request_ai`)
- Uses an intelligent AI agent to make sophisticated routing decisions
- **Reasoning-based**: AI agent analyzes context and makes human-like decisions
- Context-aware and adaptive to conversation flow
- Can handle complex routing scenarios and multi-step orchestration
- Better for conversational AI, ambiguous requests, and learning from patterns

### Cross-Platform Development

The project includes VS Code configurations for seamless development on **Windows**, **macOS**, and **Linux**:

#### VS Code Launch Configurations
- **"Run FastAPI Server Only"** - Start just the server
- **"Run Web Client Only"** - Start just the Streamlit client  
- **"Run Server and Web Client"** - Compound launch for both

#### Cross-Platform Scripts
- **Windows**: PowerShell scripts (`.ps1`)
- **macOS/Linux**: Bash scripts (`.sh`)
- **Automatic Detection**: VS Code tasks automatically use the correct scripts

#### Pre-Launch Tasks
- **`kill-server-if-running`** - Stops any existing server process
- **`wait-for-server`** - Waits for server to be ready
- **`wait-for-server-long`** - Extended wait for complex startup scenarios

For detailed setup instructions, see `docs/setup_instructions.md`

## 🆕 Recent Enhancements

### AI Agent Orchestration
- **Intelligent AI Agent**: New AI agent that uses LLM reasoning for sophisticated routing decisions
- **Context-Aware Analysis**: Deep understanding of conversation flow and user intent patterns
- **Dynamic Tool Discovery**: MCP protocol support for real-time capability discovery
- **Configurable AI Agent**: Fully customizable system prompts for AI agent reasoning behavior
- **Multi-Step Orchestration**: AI agent can break down complex requests into multiple routing steps

### Real-Time Model Discovery  
- **Live Model Fetching**: Automatically discovers models currently loaded in LM Studio
- **Health Monitoring**: Real-time tracking of LM Studio connectivity and model availability
- **Intelligent Fallback**: Graceful degradation to configuration-based models when needed

### **FastAPI + AI Agent + MCP** Architecture
- **Unified Stack**: FastAPI for REST endpoints + AI Agent for intelligent reasoning + MCP for advanced tools
- **Three-Layer Integration**: Clear separation of concerns with seamless integration
- **AI-First Design**: AI agent at the core of routing decisions with human-like reasoning capabilities

### Developer Experience
- **Cross-Platform Support**: VS Code configurations work seamlessly on Windows, macOS, and Linux
- **Automated Setup**: Compound launch configurations start both server and client
- **Enhanced Testing**: Comprehensive test suites for orchestrator and configuration validation

## 📚 Documentation

- **Setup Instructions**: `docs/setup_instructions.md`
- **Developer Guide**: `docs/developer.md`
- **API Reference**: `docs/api_reference.md`
- **Architecture Overview**: `docs/architecture.md`

## 🔧 Technology Stack

- **Backend**: Python, **FastAPI** (REST endpoints), **AI Agent** (LLM-powered reasoning), **FastMCP** (Model Context Protocol)
- **Frontend**: Streamlit
- **Database**: MongoDB
- **AI Models**: LM Studio (local model hosting)
- **Communication**: REST APIs, WebSockets, Server-Sent Events, MCP protocol
- **Intelligence Layer**: AI Agent with LLM reasoning for adaptive routing decisions
- **Real-Time Features**: Live model discovery, streaming responses, bidirectional communication
- **Development**: VS Code with cross-platform task support (Windows/macOS/Linux)

