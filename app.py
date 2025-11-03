import streamlit as st
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import re
import asyncio
import urllib.request
import urllib.error
from langchain_openai import AzureChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_core.callbacks import get_usage_metadata_callback, UsageMetadataCallbackHandler
from langchain_classic.memory import ConversationBufferMemory
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent


# Load environment variables from config file if it exists
def load_env_file(filepath):
    """Load environment variables from a file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Try to load from common config file locations
for config_file in ['config.env', '.env', 'config.txt']:
    load_env_file(config_file)



# Debug: Check if required environment variables are set
required_vars = ['AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT']
missing_vars = []
for var in required_vars:
    if not os.environ.get(var):
        missing_vars.append(var)

if missing_vars:
    print(f"⚠️ Missing required environment variables: {missing_vars}")
    print("Please check your config.env file or set these environment variables.")
else:
    print("✅ All required environment variables are set")

# Configuration settings
class AppConfig:
    def __init__(self):
        # Logging configuration
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.enable_conversation_log = os.environ.get("ENABLE_CONVERSATION_LOG", "true").lower() == "true"
        self.enable_detailed_logging = os.environ.get("ENABLE_DETAILED_LOGGING", "true").lower() == "true"
        self.log_file_path = os.environ.get("LOG_FILE_PATH", "ai_agent.log")
        self.conversation_log_path = os.environ.get("CONVERSATION_LOG_PATH", "conversation_log.json")
        
        # Performance settings
        self.streaming_delay = float(os.environ.get("STREAMING_DELAY", "0.02"))
        self.max_conversation_history = int(os.environ.get("MAX_CONVERSATION_HISTORY", "100"))
        
        # UI settings
        self.show_debug_info = os.environ.get("SHOW_DEBUG_INFO", "false").lower() == "true"
        self.enable_cost_alerts = os.environ.get("ENABLE_COST_ALERTS", "true").lower() == "true"
        self.cost_alert_threshold = float(os.environ.get("COST_ALERT_THRESHOLD", "1.0"))

# Initialize configuration
app_config = AppConfig()

# Set page config
st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="wide"
)

# Configure logging based on configuration
log_level = getattr(logging, app_config.log_level.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(app_config.log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

if "interaction_count" not in st.session_state:
    st.session_state.interaction_count = 0


# Load model configurations
def load_model_configs():
    """Load model configurations from JSON file"""
    try:
        with open('model_configs.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("model_configs.json not found. Using default configuration.")
        return {
            "models": {
                "gpt-4o": {
                    "name": "GPT-4o",
                    "deployment_name": "gpt-4o",
                    "api_version": "2025-01-01-preview",
                    "model_version": "2024-08-06",
                    "temperature": 0.0,
                    "supports_temperature": True,
                    "description": "Default GPT-4o model"
                }
            },
            "default_model": "gpt-4o"
        }
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing model_configs.json: {e}")
        raise

model_configs = load_model_configs()

# -------------------- MCP helpers --------------------
def load_mcp_config_file():
    """Locate and load an MCP config file (mcp.json). Returns (config_dict, path_str) or (None, None)."""
    # Env override
    env_path = os.environ.get("MCP_CONFIG_PATH")
    candidate_paths = []
    if env_path:
        candidate_paths.append(Path(env_path))
    # Look next to this app and at repo root
    app_dir = Path(__file__).resolve().parent
    repo_root = app_dir.parent
    candidate_paths.extend([app_dir / "mcp.json", repo_root / "mcp.json"]) 

    for p in candidate_paths:
        try:
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    return json.load(f), str(p)
        except Exception:
            continue
    return None, None

def _find_http_server_port(py_file_path: Path):
    """Best-effort parse of FastMCP .run(... port=NNNN ...) in a python file."""
    try:
        content = py_file_path.read_text(encoding="utf-8")
    except Exception:
        return None
    m = re.search(r"mcp\.run\([^)]*port\s*=\s*(\d+)", content)
    return int(m.group(1)) if m else None

def _list_mcp_tools_from_file(py_file_path: Path):
    """Collect tool function names following @mcp.tool decorators in a file."""
    tools = []
    try:
        lines = py_file_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return tools
    for i, line in enumerate(lines):
        if line.strip().startswith("@mcp.tool"):
            # Look ahead for the next async def
            for j in range(i + 1, min(i + 10, len(lines))):
                m = re.match(r"\s*async\s+def\s+([a-zA-Z_][a-zA-Z0-9_]*)\(", lines[j])
                if m:
                    tools.append(m.group(1))
                    break
    return tools

def discover_repo_mcp_servers():
    """Discover local MCP HTTP servers in sibling directories and extract ports and tools."""
    app_dir = Path(__file__).resolve().parent
    repo_root = app_dir.parent
    candidates = [
        ("aws-news", repo_root / "streamable-HTTP-server" / "main.py"),
        ("entra", repo_root / "streamable-HTTP-Entra-MCP" / "main.py"),
        ("aws-org", repo_root / "streamable-HTTP-aws-org-mcp" / "main.py"),
    ]
    servers = []
    for name, path in candidates:
        if path.exists():
            servers.append({
                "name": name,
                "path": str(path),
                "port": _find_http_server_port(path),
                "tools": _list_mcp_tools_from_file(path),
            })
    return servers

def check_http_health(port: int):
    """Call /health on localhost:port if possible. Returns dict with status info."""
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=1.5) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            try:
                data = json.loads(body)
            except Exception:
                data = {"raw": body}
            return {"reachable": True, "status": resp.status, "data": data}
    except Exception as e:
        return {"reachable": False, "error": str(e)}

def _redact_env(env_dict: dict | None) -> dict | None:
    if not env_dict:
        return None
    redacted = {}
    for k, v in env_dict.items():
        sval = str(v)
        if len(sval) > 8:
            redacted[k] = sval[:4] + "***" + sval[-2:]
        else:
            redacted[k] = "***"
    return redacted

def _build_mcp_client_config_from_json(config: dict) -> dict:
    """Translate mcp.json style config into MultiServerMCPClient mapping."""
    mapping: dict = {}
    servers = config.get("mcpServers") or config.get("servers") or {}
    for name, entry in servers.items():
        # Honor explicit transport if present
        transport = entry.get("transport")
        if entry.get("url") and (transport is None or transport in {"http", "streamable_http", "streamable-http"}):
            mapping[name] = {
                "transport": "streamable_http",
                "url": entry.get("url"),
            }
        elif entry.get("command"):
            mapping[name] = {
                "transport": "stdio",
                "command": entry.get("command"),
                "args": entry.get("args", []),
                "env": entry.get("env", {}),
            }
        # else: ignore unrecognized
    return mapping

def _build_mcp_client_config_from_discovered(servers: list[dict]) -> dict:
    mapping: dict = {}
    for srv in servers:
        name = srv.get("name")
        port = srv.get("port")
        if name and port:
            mapping[name] = {
                "transport": "streamable_http",
                "url": f"http://127.0.0.1:{port}/mcp",
            }
    return mapping

def _merge_mappings(primary: dict, secondary: dict) -> dict:
    out = dict(secondary)
    out.update(primary)
    return out

# -------------------- System prompt & Agent helpers --------------------
def load_system_prompt() -> str:
    """Load system prompt text from file specified by SYSTEM_PROMPT_PATH or default system_prompt.md next to app.py."""
    app_dir = Path(__file__).resolve().parent
    default_path = app_dir / "system_prompt.md"
    path_str = os.environ.get("SYSTEM_PROMPT_PATH", str(default_path))
    try:
        return Path(path_str).read_text(encoding="utf-8").strip()
    except Exception:
        # Minimal safe default
        return "You are a helpful agent. Use tools when helpful and avoid hallucinations."

@st.cache_resource
def get_mcp_tools_from_json():
    """Return LangChain tools loaded via langchain-mcp-adapters from mcp.json only."""
    config, _ = load_mcp_config_file()
    if not config:
        return []
    mapping = _build_mcp_client_config_from_json(config)
    if not mapping:
        return []
    async def _load(mapping: dict):
        client = MultiServerMCPClient(mapping)
        return await client.get_tools()
    try:
        return asyncio.run(_load(mapping))
    except RuntimeError:
        # In case an event loop is already running (some Streamlit environments)
        return []

@st.cache_resource
def get_agent(selected_model_key=None):
    """Create and cache an agent that uses tools from mcp.json and an external system prompt."""
    llm_inst = get_llm(selected_model_key)
    tools = get_mcp_tools_from_json()
    if not tools:
        return None
    try:
        agent = create_agent(
            model=llm_inst,
            tools=tools,
            system_prompt=load_system_prompt(),
        )
        return agent
    except Exception:
        return None

# Initialize the LLM
@st.cache_resource
def get_llm(selected_model_key=None):
    """
    Initialize Azure OpenAI LLM based on selected model configuration.
    """
    # Get selected model or use default
    if selected_model_key is None:
        selected_model_key = st.session_state.get("selected_model", model_configs["default_model"])
    
    if selected_model_key not in model_configs["models"]:
        logger.warning(f"Model {selected_model_key} not found, using default model")
        selected_model_key = model_configs["default_model"]
    
    model_config = model_configs["models"][selected_model_key]
    
    # Create LLM configuration from model config
    try:
        azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        azure_api_key = os.environ["AZURE_OPENAI_API_KEY"]
    except KeyError as e:
        logger.error(f"Missing required environment variable: {e}")
        st.error(f"❌ Missing required environment variable: {e}")
        st.error("Please check your config.env file or set the environment variables.")
        st.stop()
    
    llm_config = {
        "azure_endpoint": azure_endpoint,
        "azure_deployment": model_config["deployment_name"],
        "openai_api_version": model_config["api_version"],
        "max_tokens": None,
        "timeout": 30,  # Add timeout to prevent hanging
        "max_retries": 2,
        "model_version": model_config["model_version"],
    }
    
    # Add temperature only if the model supports it
    if model_config.get("supports_temperature", True):
        llm_config["temperature"] = model_config.get("temperature", 0.0)
        logger.info(f"LLM initialized with temperature: {llm_config['temperature']}")
    else:
        logger.info("LLM initialized without temperature (not supported by this model)")
    
    try:
        llm = AzureChatOpenAI(**llm_config)
        logger.info(f"LLM initialized successfully: {model_config['name']}")
        
        # Test connection with a simple call
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            test_response = llm.invoke(test_messages)
            logger.info("✅ Connection test successful")
        except Exception as test_e:
            logger.warning(f"⚠️ Connection test failed: {test_e}")
            logger.info("This might be due to network issues or API configuration")
        
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM {model_config['name']}: {str(e)}")
        logger.error(f"Endpoint: {azure_endpoint}")
        logger.error(f"Deployment: {model_config['deployment_name']}")
        logger.error(f"API Version: {model_config['api_version']}")
        raise e

# Initialize session state for model selection
if "selected_model" not in st.session_state:
    st.session_state.selected_model = model_configs["default_model"]

# Test connection status
@st.cache_data
def test_azure_connection():
    """Test Azure OpenAI connection and return status"""
    try:
        # Get the current model configuration
        model_config = model_configs["models"][st.session_state.selected_model]
        
        # Create a test LLM instance
        test_llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=model_config["deployment_name"],
            openai_api_version=model_config["api_version"],
            model_version=model_config["model_version"],
            timeout=10,  # Short timeout for testing
            max_retries=1
        )
        
        # Test with a simple message
        test_messages = [{"role": "user", "content": "test"}]
        test_response = test_llm.invoke(test_messages)
        
        return True, "Connected"
    except Exception as e:
        return False, str(e)

# App title
st.title("🤖 AI Agent")
st.markdown("A simple AI assistant powered by Azure OpenAI")

# Helper function to log interaction
def log_interaction(user_input, assistant_response, cost, prompt_tokens, completion_tokens, total_tokens, response_metadata=None, usage_metadata=None):
    """Log the interaction to file and console with comprehensive details"""
    
    # Get current model info
    current_model_config = model_configs["models"][st.session_state.selected_model]
    
    # Process response metadata if available
    azure_metadata = {}
    if response_metadata:
        try:
            # Extract key information from Azure response metadata
            azure_metadata = {
                "model_name": getattr(response_metadata, 'model_name', None),
                "system_fingerprint": getattr(response_metadata, 'system_fingerprint', None),
                "finish_reason": getattr(response_metadata, 'finish_reason', None),
                "service_tier": getattr(response_metadata, 'service_tier', None),
                "token_usage_details": {
                    "input_tokens": getattr(response_metadata, 'input_tokens', prompt_tokens),
                    "output_tokens": getattr(response_metadata, 'output_tokens', completion_tokens),
                    "total_tokens": getattr(response_metadata, 'total_tokens', total_tokens),
                    "input_token_details": getattr(response_metadata, 'input_token_details', {}),
                    "output_token_details": getattr(response_metadata, 'output_token_details', {})
                },
                "content_filter_results": getattr(response_metadata, 'content_filter_results', {}),
                "prompt_filter_results": getattr(response_metadata, 'prompt_filter_results', [])
            }
        except Exception as e:
            logger.warning(f"Could not parse response metadata: {e}")
            azure_metadata = {"error": str(e)}
    
    # Process usage metadata from modern callback
    modern_usage_metadata = {}
    if usage_metadata:
        try:
            modern_usage_metadata = {
                "input_tokens": usage_metadata.get('input_tokens', prompt_tokens),
                "output_tokens": usage_metadata.get('output_tokens', completion_tokens),
                "total_tokens": usage_metadata.get('total_tokens', total_tokens),
                "input_token_details": usage_metadata.get('input_token_details', {}),
                "output_token_details": usage_metadata.get('output_token_details', {})
            }
        except Exception as e:
            logger.warning(f"Could not parse usage metadata: {e}")
            modern_usage_metadata = {"error": str(e)}
    
    interaction_data = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "assistant_response": assistant_response,
        "cost": cost,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "interaction_id": st.session_state.interaction_count,
        "model_used": {
            "name": current_model_config["name"],
            "deployment_name": current_model_config["deployment_name"],
            "api_version": current_model_config["api_version"],
            "model_version": current_model_config["model_version"],
            "temperature": current_model_config.get("temperature", "N/A"),
            "supports_temperature": current_model_config.get("supports_temperature", True)
        },
        "response_metadata": {
            "response_length": len(assistant_response),
            "response_word_count": len(assistant_response.split()),
            "has_formatting": any(marker in assistant_response for marker in ['**', '*', '-', '1.', '2.', '3.']),
            "contains_code": '```' in assistant_response or '`' in assistant_response
        },
        "azure_response_metadata": azure_metadata,
        "modern_usage_metadata": modern_usage_metadata
    }
    
    # Log to file with full response
    logger.info(f"Interaction {st.session_state.interaction_count}: Cost ${cost:.6f} (Prompt: {prompt_tokens}, Completion: {completion_tokens})")
    logger.info(f"Model: {current_model_config['name']}")
    logger.info(f"User: {user_input}")
    logger.info(f"Assistant (Full Response): {assistant_response}")
    logger.info(f"Response Length: {len(assistant_response)} characters, {len(assistant_response.split())} words")
    
    # Log Azure metadata if available
    if azure_metadata:
        logger.info(f"Azure Model: {azure_metadata.get('model_name', 'N/A')}")
        logger.info(f"System Fingerprint: {azure_metadata.get('system_fingerprint', 'N/A')}")
        logger.info(f"Finish Reason: {azure_metadata.get('finish_reason', 'N/A')}")
        logger.info(f"Service Tier: {azure_metadata.get('service_tier', 'N/A')}")
        
        # Log content filter results
        content_filter = azure_metadata.get('content_filter_results', {})
        if content_filter:
            logger.info("Content Filter Results:")
            for category, result in content_filter.items():
                if isinstance(result, dict):
                    filtered = result.get('filtered', False)
                    severity = result.get('severity', 'unknown')
                    logger.info(f"  {category}: filtered={filtered}, severity={severity}")
    
    # Log modern usage metadata if available
    if modern_usage_metadata:
        logger.info("Modern Usage Metadata:")
        logger.info(f"  Input Tokens: {modern_usage_metadata.get('input_tokens', 'N/A')}")
        logger.info(f"  Output Tokens: {modern_usage_metadata.get('output_tokens', 'N/A')}")
        logger.info(f"  Total Tokens: {modern_usage_metadata.get('total_tokens', 'N/A')}")
        
        input_details = modern_usage_metadata.get('input_token_details', {})
        output_details = modern_usage_metadata.get('output_token_details', {})
        if input_details or output_details:
            logger.info("  Token Details:")
            if input_details:
                logger.info(f"    Input: {input_details}")
            if output_details:
                logger.info(f"    Output: {output_details}")
    
    logger.info("-" * 80)
    
    # Save detailed log to JSON file (if enabled)
    if app_config.enable_conversation_log:
        try:
            logs = []
            if os.path.exists(app_config.conversation_log_path):
                try:
                    with open(app_config.conversation_log_path, 'r') as f:
                        content = f.read().strip()
                        if content:  # Only try to parse if file has content
                            logs = json.loads(content)
                        else:
                            logger.info("Empty log file found, starting fresh")
                except json.JSONDecodeError as e:
                    logger.warning(f"Corrupted JSON in {app_config.conversation_log_path}, starting fresh: {e}")
                    logs = []
            
            logs.append(interaction_data)
            
            # Limit conversation history if configured
            if len(logs) > app_config.max_conversation_history:
                logs = logs[-app_config.max_conversation_history:]
                logger.info(f"Trimmed conversation log to {app_config.max_conversation_history} entries")
            
            with open(app_config.conversation_log_path, 'w') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Successfully saved interaction to {app_config.conversation_log_path}")
        except Exception as e:
            logger.error(f"Failed to save interaction log: {e}")
    else:
        logger.info("Conversation logging disabled")

# Model selection in sidebar
with st.sidebar:
    st.header("Model Selection")
    
    # Create model options for selectbox
    model_options = {}
    for key, config in model_configs["models"].items():
        model_options[f"{config['name']} ({key})"] = key
    
    # Model selector
    selected_model_display = st.selectbox(
        "Choose Model:",
        options=list(model_options.keys()),
        index=list(model_options.values()).index(st.session_state.selected_model),
        help="Select the AI model to use for conversations"
    )
    
    # Update selected model if changed
    new_selected_model = model_options[selected_model_display]
    if new_selected_model != st.session_state.selected_model:
        st.session_state.selected_model = new_selected_model
        # Clear cache to force LLM reinitialization
        get_llm.clear()
        st.rerun()
    
    # Show model info
    current_model_config = model_configs["models"][st.session_state.selected_model]
    st.info(f"**{current_model_config['name']}**\n\n{current_model_config['description']}")
    
    if current_model_config.get("supports_temperature", True):
        st.success("✅ Temperature supported")
    else:
        st.warning("⚠️ Temperature not supported")

"""Main content tabs"""
tabs = st.tabs(["Chat", "MCP"])

# Get the current LLM instance (used in Chat tab)
llm = get_llm(st.session_state.selected_model)

with tabs[0]:
    # Chat interface
    use_agent = st.toggle("Use MCP Agent", value=True, help="Use tools from mcp.json with an agent and external system prompt")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to ask?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Add to memory
        st.session_state.memory.chat_memory.add_user_message(prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            # Get conversation history from memory
            conversation_history = st.session_state.memory.chat_memory.messages
            
            # Prepare messages for the LLM with full conversation context
            messages = [("system", "You are a helpful assistant. You have access to the full conversation history.")]
            
            # Add conversation history
            for msg in conversation_history:
                if hasattr(msg, 'content'):
                    role = "human" if msg.__class__.__name__ == "HumanMessage" else "assistant"
                    messages.append((role, msg.content))
            
            # Get response with full metadata using modern LangChain approach
            response_container = st.empty()
            
            # Use both callbacks for comprehensive tracking
            with get_openai_callback() as cb_openai, get_usage_metadata_callback() as cb_usage:
                if use_agent:
                    # Build conversation history for the agent
                    hist = []
                    for msg in conversation_history:
                        if hasattr(msg, 'content'):
                            role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
                            hist.append({"role": role, "content": msg.content})
                    hist.append({"role": "user", "content": prompt})

                    agent = get_agent(st.session_state.selected_model)
                    if agent is None:
                        full_response = "MCP agent is not available (no tools from mcp.json)."
                        class _Obj: pass
                        full_response_obj = _Obj()
                        setattr(full_response_obj, 'response_metadata', None)
                        setattr(full_response_obj, 'usage_metadata', None)
                    else:
                        try:
                            result = asyncio.run(agent.ainvoke({"messages": hist}))
                            msgs = result.get("messages", []) if isinstance(result, dict) else []
                            full_response = (msgs[-1].content if msgs else "") or str(result)
                            # Synthetic object to keep existing logging path
                            class _Obj: pass
                            full_response_obj = _Obj()
                            setattr(full_response_obj, 'response_metadata', None)
                            setattr(full_response_obj, 'usage_metadata', None)
                        except Exception as e:
                            full_response = f"Agent error: {e}"
                            class _Obj: pass
                            full_response_obj = _Obj()
                            setattr(full_response_obj, 'response_metadata', None)
                            setattr(full_response_obj, 'usage_metadata', None)
                else:
                    # Direct LLM response (no tools)
                    full_response_obj = llm.invoke(messages)
                    full_response = full_response_obj.content
                
                # Display the response with streaming effect
                response_container.markdown("")
                for i in range(len(full_response) + 1):
                    response_container.markdown(full_response[:i] + "▌")
                    import time
                    time.sleep(app_config.streaming_delay)  # Configurable delay for streaming effect
                
                # Remove the cursor and display final response
                response_container.markdown(full_response)
                
                # Extract comprehensive metadata
                response_metadata = None
                usage_metadata = None
                
                # Get response metadata from the response object
                if hasattr(full_response_obj, 'response_metadata'):
                    response_metadata = full_response_obj.response_metadata
                    logger.info("Found response_metadata on full_response_obj")
                elif hasattr(full_response_obj, 'usage_metadata'):
                    usage_metadata = full_response_obj.usage_metadata
                    logger.info("Found usage_metadata on full_response_obj")
                
                # Get usage metadata from the modern callback
                if cb_usage.usage_metadata:
                    logger.info(f"Usage metadata from callback: {cb_usage.usage_metadata}")
                    # Extract the first (and likely only) model's usage data
                    for model_name, usage_data in cb_usage.usage_metadata.items():
                        usage_metadata = usage_data
                        break
                
                # Debug logging (if enabled)
                if app_config.enable_detailed_logging:
                    logger.info(f"Full response object type: {type(full_response_obj)}")
                    logger.info(f"Response metadata: {response_metadata}")
                    logger.info(f"Usage metadata: {usage_metadata}")
                    logger.info(f"OpenAI callback cost: ${cb_openai.total_cost:.6f}")
                
                # Update session state with cost tracking
                st.session_state.total_cost += cb_openai.total_cost
                st.session_state.interaction_count += 1
                
                # Log the interaction with comprehensive metadata
                log_interaction(prompt, full_response, cb_openai.total_cost, 
                              cb_openai.prompt_tokens, cb_openai.completion_tokens, 
                              cb_openai.total_tokens, response_metadata, usage_metadata)
        
        # Add assistant response to chat history and memory
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.memory.chat_memory.add_ai_message(full_response)

with tabs[1]:
    st.subheader("MCP Configuration")
    config, config_path = load_mcp_config_file()
    if config:
        st.success(f"Loaded mcp.json from: {config_path}")
        # Show a redacted view to avoid exposing secrets
        preview = json.loads(json.dumps(config))  # deep copy
        try:
            servers = preview.get("mcpServers") or {}
            for _, v in servers.items():
                if isinstance(v, dict) and "env" in v:
                    v["env"] = _redact_env(v.get("env"))
        except Exception:
            pass
        st.json(preview)
    else:
        st.warning("No mcp.json found. Set MCP_CONFIG_PATH or add mcp.json next to app.py.")
        uploaded = st.file_uploader("Optionally upload an mcp.json to preview", type=["json"])
        if uploaded is not None:
            try:
                st.json(json.load(uploaded))
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    # Servers from mcp.json (detailed list)
    if config:
        st.markdown("---")
        st.subheader("Servers from mcp.json")
        raw_servers = (config.get("mcpServers") or config.get("servers") or {})
        if not raw_servers:
            st.info("No servers defined in mcp.json")
        else:
            for name, entry in raw_servers.items():
                transport = entry.get("transport")
                url = entry.get("url")
                command = entry.get("command")
                args = entry.get("args")
                cols = st.columns([3,3,6,4])
                with cols[0]:
                    st.markdown(f"**{name}**")
                with cols[1]:
                    if url:
                        st.markdown("Transport: `streamable_http`")
                    elif command:
                        st.markdown("Transport: `stdio`")
                    else:
                        st.markdown(f"Transport: `{transport or 'unknown'}`")
                with cols[2]:
                    if url:
                        st.markdown(f"URL: `{url}`")
                    elif command:
                        st.markdown(f"Command: `{command}`\n\nArgs: `{(args or [])}`")
                    else:
                        st.markdown("—")
                with cols[3]:
                    # Health only for HTTP
                    if url:
                        try:
                            # Try to parse a port and check /health if localhost
                            m = re.match(r"https?://(?:127\.0\.0\.1|localhost):(\d+)/", url)
                            if m:
                                health = check_http_health(int(m.group(1)))
                                if health.get("reachable"):
                                    st.success("🟢 healthy")
                                else:
                                    st.error("🔴 unreachable")
                            else:
                                st.caption("Health check only for localhost URLs")
                        except Exception:
                            st.caption("Health check unavailable")

    st.markdown("---")
    st.subheader("MCP Client (langchain-mcp-adapters)")
    client_config = {}
    if config:
        client_config = _build_mcp_client_config_from_json(config)

    if not client_config:
        st.info("No MCP servers configured. Provide a valid mcp.json.")
    else:
        st.caption("Client configuration (redacted env):")
        redacted_view = {}
        for k, v in client_config.items():
            if isinstance(v, dict) and "env" in v:
                vv = dict(v)
                vv["env"] = _redact_env(v.get("env"))
                redacted_view[k] = vv
            else:
                redacted_view[k] = v
        st.json(redacted_view)

        if st.button("List MCP tools (all servers)"):
            try:
                async def _fetch_tools():
                    client = MultiServerMCPClient(client_config)
                    return await client.get_tools()
                tools_info = asyncio.run(_fetch_tools())
                # tools_info is typically a list of tool schemas per server
                st.success("Fetched tools from MCP servers")
                st.json(tools_info)
            except Exception as e:
                st.error(f"Failed to fetch tools: {e}")

        # Per-server tool listing (use sessions)
        if st.button("List tools per server"):
            try:
                async def _fetch_per_server(mapping: dict):
                    client = MultiServerMCPClient(mapping)
                    results = {}
                    for name in mapping.keys():
                        try:
                            async with client.session(name) as session:
                                tools = await load_mcp_tools(session)
                                results[name] = [t.name for t in tools]
                        except Exception as inner:
                            results[name] = {"error": str(inner)}
                    return results
                per = asyncio.run(_fetch_per_server(client_config))
                st.success("Fetched tools per server")
                st.json(per)
            except Exception as e:
                st.error(f"Failed to fetch tools per server: {e}")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.session_state.total_cost = 0.0
        st.session_state.interaction_count = 0
        logger.info("Chat cleared by user")
        st.rerun()
    
    st.markdown("---")
    st.header("Session Statistics")
    
    # Cost tracking
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Cost", f"${st.session_state.total_cost:.6f}")
    with col2:
        st.metric("Interactions", st.session_state.interaction_count)
    
    # Average cost per interaction
    if st.session_state.interaction_count > 0:
        avg_cost = st.session_state.total_cost / st.session_state.interaction_count
        st.metric("Avg Cost/Interaction", f"${avg_cost:.6f}")
    
    # Token usage info
    if app_config.enable_conversation_log and os.path.exists(app_config.conversation_log_path):
        try:
            with open(app_config.conversation_log_path, 'r') as f:
                content = f.read().strip()
                if content:
                    logs = json.loads(content)
                    if logs:
                        total_prompt_tokens = sum(log.get("prompt_tokens", 0) for log in logs)
                        total_completion_tokens = sum(log.get("completion_tokens", 0) for log in logs)
                        st.metric("Total Prompt Tokens", f"{total_prompt_tokens:,}")
                        st.metric("Total Completion Tokens", f"{total_completion_tokens:,}")
                else:
                    st.metric("Total Prompt Tokens", "0")
                    st.metric("Total Completion Tokens", "0")
        except (json.JSONDecodeError, Exception) as e:
            st.warning("Could not load conversation log")
            st.metric("Total Prompt Tokens", "0")
            st.metric("Total Completion Tokens", "0")
    elif not app_config.enable_conversation_log:
        st.metric("Total Prompt Tokens", "N/A")
        st.metric("Total Completion Tokens", "N/A")
    
    st.markdown("---")
    st.header("Connection Status")
    
    # Test connection
    try:
        is_connected, status_msg = test_azure_connection()
        if is_connected:
            st.success("🟢 Connected to Azure OpenAI")
        else:
            st.error(f"🔴 Connection Failed: {status_msg}")
    except Exception as e:
        st.error(f"🔴 Connection Error: {str(e)}")
    
    st.markdown("---")
    st.header("Current Model Info")
    current_model_config = model_configs["models"][st.session_state.selected_model]
    st.markdown(f"**Model:** {current_model_config['name']}")
    st.markdown(f"**Deployment:** {current_model_config['deployment_name']}")
    st.markdown(f"**API Version:** {current_model_config['api_version']}")
    st.markdown("**Provider:** Azure OpenAI")
    
    # Show temperature status
    if current_model_config.get("supports_temperature", True):
        st.markdown(f"**Temperature:** {current_model_config.get('temperature', 'Not set')}")
    else:
        st.markdown("**Temperature:** Not supported")
        st.info("ℹ️ This model doesn't support temperature parameter")
    
    # Memory info
    st.markdown("---")
    st.header("Memory")
    memory_messages = len(st.session_state.memory.chat_memory.messages)
    st.metric("Messages in Memory", memory_messages)
    
    if st.button("View Memory"):
        st.text_area("Conversation Memory", 
                    value=str(st.session_state.memory.chat_memory.messages), 
                    height=200)
    
    # Log files info
    st.markdown("---")
    st.header("Logs")
    if os.path.exists(app_config.log_file_path):
        log_size = os.path.getsize(app_config.log_file_path)
        st.metric("Log File Size", f"{log_size} bytes")
    
    if app_config.enable_conversation_log and os.path.exists(app_config.conversation_log_path):
        try:
            with open(app_config.conversation_log_path, 'r') as f:
                content = f.read().strip()
                if content:
                    logs = json.loads(content)
                    st.metric("Logged Interactions", len(logs))
                else:
                    st.metric("Logged Interactions", 0)
        except (json.JSONDecodeError, Exception) as e:
            st.metric("Logged Interactions", 0)
    elif not app_config.enable_conversation_log:
        st.info("Conversation logging disabled")
    
    # Configuration controls
    st.markdown("---")
    st.header("Configuration")
    
    # Logging controls
    st.subheader("Logging Settings")
    enable_conversation_log = st.checkbox(
        "Enable Conversation Log", 
        value=app_config.enable_conversation_log,
        help="Save detailed conversation logs to JSON file"
    )
    
    enable_detailed_logging = st.checkbox(
        "Enable Detailed Logging", 
        value=app_config.enable_detailed_logging,
        help="Include comprehensive metadata in logs"
    )
    
    # Update configuration if changed
    if enable_conversation_log != app_config.enable_conversation_log:
        app_config.enable_conversation_log = enable_conversation_log
        st.rerun()
    
    if enable_detailed_logging != app_config.enable_detailed_logging:
        app_config.enable_detailed_logging = enable_detailed_logging
        st.rerun()
    
    # Performance settings
    st.subheader("Performance Settings")
    streaming_delay = st.slider(
        "Streaming Delay (seconds)", 
        min_value=0.0, 
        max_value=0.1, 
        value=app_config.streaming_delay, 
        step=0.01,
        help="Delay between characters in streaming effect"
    )
    
    if streaming_delay != app_config.streaming_delay:
        app_config.streaming_delay = streaming_delay
    
    # Cost alerts
    if app_config.enable_cost_alerts:
        st.subheader("Cost Alerts")
        if st.session_state.total_cost > app_config.cost_alert_threshold:
            st.warning(f"⚠️ Session cost exceeded ${app_config.cost_alert_threshold:.2f}")
    
    # Debug info
    if app_config.show_debug_info:
        st.markdown("---")
        st.header("Debug Information")
        st.json({
            "log_level": app_config.log_level,
            "conversation_log_enabled": app_config.enable_conversation_log,
            "detailed_logging_enabled": app_config.enable_detailed_logging,
            "streaming_delay": app_config.streaming_delay,
            "max_conversation_history": app_config.max_conversation_history
        })