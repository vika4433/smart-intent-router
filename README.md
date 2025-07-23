# Smart Intent Router

A sophisticated AI request routing system built on the **Model Context Protocol (MCP)** that intelligently classifies user intents, detects languages, and routes requests to the most appropriate language model. Features a modern web interface and a modular server architecture with conversation management, multilingual support, and dynamic LLM selection.

## 🚀 Features

- **Intent Classification**: Automatically classifies user requests into categories (code, math, translation, creative writing, general)
- **Language Detection**: Supports multilingual input with automatic language detection
- **Dynamic LLM Selection**: Routes requests to the best model based on intent and language
- **Conversation Management**: Persistent chat sessions with context-aware responses
- **Conversation History**: All conversations are stored in MongoDB for persistence across sessions
- **Context-Aware Responses**: Smart context retrieval maintains conversation flow and relevance
- **Web Interface**: Modern Streamlit-based chat interface with markdown rendering
- **MCP Server**: Modular server architecture using Model Context Protocol
- **LM Studio Integration**: Seamless integration with LM Studio for local model hosting

## 🏗️ Architecture

### Components

The Smart Intent Router consists of two main components:

#### 1. **Web Client** (`web-client/`)
- **Streamlit-based chat interface** with real-time messaging
- **Markdown rendering** with syntax highlighting for code blocks
- **Session management** with persistent conversation history
- **Async MCP client** for communication with the backend server

#### 2. **MCP Server** (`smart-intent-router-server/`)
A modular backend server built on the Model Context Protocol with the following components:

##### Core Components:

- **Intent Classifier** (`intent_classifier/`)
  - Analyzes user messages to determine intent type
  - Supports: `code`, `math`, `translation`, `creative writing`, `general`
  - Uses keyword-based classification with LLM fallback

- **Language Detector** (`language_detector/`)
  - Automatically detects the language of user input
  - Supports multiple languages for multilingual routing

- **LLM Selector** (`llm_selector/`)
  - Dynamically selects the best model based on intent and language
  - Configurable model mappings and fallback strategies

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

- **Response Handler** (`response_handler/`)
  - Processes and formats LLM responses
  - Handles different response types and error cases

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

#### Model Selection Logic
When multiple models are available for the same intent and language:
1. **Weight-based selection**: The model with the highest `weight` value is automatically chosen
2. **Fallback order**: If weights are equal, models are prioritized by configuration order
3. **Default handling**: Falls back to `default_model` if no suitable match is found

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

### Example Configuration Structure
```yaml
# Example: Customizable template - modify as needed
intents: [code, math, translation, creative_writing, general]
languages: [en, es, fr]  # Add your supported languages

models:
  - model_name: "your-lm-studio-model-name"
    endpoint: "http://localhost:1234/v1/chat/completions"
    context_length: 4096
    weight: 2.0              # Higher weight = higher priority
    supported_intents: [code]
    supported_languages: [en]
    enabled: true
    
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

## 🛠️ MCP Tools

The server exposes the following MCP tools:

- **`route_request`**: Main routing function that processes user messages
- **`classify_intent`**: Classifies message intent
- **`detect_language`**: Detects message language
- **`select_llm_model`**: Selects appropriate model
- **`create_session`**: Creates new user session
- **`create_conversation`**: Creates new conversation
- **`get_conversations`**: Retrieves user conversations
- **`get_messages`**: Retrieves conversation messages
- **`delete_messages`**: Clears conversation history

## 🚦 Getting Started

### Prerequisites
- Python 3.8+
- MongoDB (local or remote)
- LM Studio (for local model hosting)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/smart-intent-router.git
   cd smart-intent-router
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

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
   ```bash
   # Start the MCP server
   python smart-intent-router-server/src/mcp_server/server.py
   
   # In another terminal, start the web client
   streamlit run web-client/src/app.py
   ```

7. **Access the application**
   - Open your browser to `http://localhost:8501`
   - Start chatting with your intelligent router!

### Model Weight Configuration Example
When you have multiple models that can handle the same intent, use weights to set priority:

```yaml
models:
  - model_name: "qwen2.5-coder-7b-instruct"
    weight: 2.0        # Higher priority for code tasks
    supported_intents: ["code"]
    
  - model_name: "llama-3.2-3b-instruct"  
    weight: 1.0        # Lower priority backup
    supported_intents: ["code", "general"]
    
  - model_name: "wizardmath-7b"
    weight: 3.0        # Highest priority for math
    supported_intents: ["math"]
```

**How weight selection works**: If a user asks a coding question, the system will choose `qwen2.5-coder-7b-instruct` (weight 2.0) over `llama-3.2-3b-instruct` (weight 1.0), even though both can handle code tasks.

For detailed setup instructions, see `docs/setup_instructions.md`

## 📚 Documentation

- **Setup Instructions**: `docs/setup_instructions.md`
- **Developer Guide**: `docs/developer.md`
- **API Reference**: `docs/api_reference.md`
- **Architecture Overview**: `docs/architecture.md`

## 🔧 Technology Stack

- **Backend**: Python, FastMCP (Model Context Protocol)
- **Frontend**: Streamlit
- **Database**: MongoDB
- **AI Models**: LM Studio (local model hosting)
- **Communication**: Async HTTP with MCP protocol

