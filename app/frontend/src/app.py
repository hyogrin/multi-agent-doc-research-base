from fastapi import FastAPI
import gradio as gr
import requests
import os
import time
import json
from datetime import datetime
from i18n.locale_msg_front import UI_TEXT, EXAMPLE_PROMPTS


# Load environment variables
API_URL = os.getenv("API_URL", "http://localhost:8000/chat")
DEEP_SEARCH_API_URL = os.getenv("DEEP_SEARCH_API_URL", "http://localhost:8000/deep_search")

AUTH_USERNAME = os.getenv("AUTH_USERNAME") or os.getenv("GRADIO_USERNAME")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD") or os.getenv("GRADIO_PASSWORD")
auth_credentials = [(AUTH_USERNAME, AUTH_PASSWORD)] if AUTH_USERNAME and AUTH_PASSWORD else None

# Define the agent modes and example prompts
AGENT_MODES = {
    "query_rewrite": {"name": "Query Rewrite", "description": "GPT will respond rewriting your query"},
    "plan_execute": {"name": "Plan & Execute", "description": "GPT will plan & execute when the queries are complex"},
    "search_engine": {"name": "Search Engine", "description": "select your search engine"},
}

SEARCH_ENGINES = {
    "Bing Search": "bing_search_crawling",
    "Grounding Search": "grounding_bing_crawling",
    "Grounding Gen": "grounding_bing"
}

# Internationalization constants
SUPPORTED_LANGUAGES = {
    "en-US": "English",
    "ko-KR": "한국어"
}

class ChatMessage:
    def __init__(self, role, content, timestamp=None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now().strftime("%H:%M")


def stream_chat_with_api(message, chat_history, agent_mode, search_engine_choice, language="ko-KR", max_tokens=4000, temperature=0.7):
    """
    Stream-enabled version of the chat_with_api function that yields partial updates
    to enable real-time UI updates as tokens are received.
    """
    if not message or message.strip() == "":
        yield "", chat_history
        return
    
    # Format the conversation history for the API
    prev_messages = []
    for msg in chat_history[-10:]:  # Limit to last 10 messages for context
        prev_messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
    # Add the new message
    formatted_messages = prev_messages + [{"role": "user", "content": message}]
    
    # Fixed agent mode handling - agent_mode is now a list
    query_rewrite = "query_rewrite" in agent_mode if isinstance(agent_mode, list) else False
    plan_execute = "plan_execute" in agent_mode if isinstance(agent_mode, list) else False
    
    # Prepare the API payload - search_engine_choice should now be a string value
    payload = {
        "messages": formatted_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "query_rewrite": query_rewrite,
        "plan_execute": plan_execute,
        "search_engine": search_engine_choice,  # This should now be a string like "grounding_bing"
        "stream": True,
        "locale": language
    }
    
    # Debug logging
    print(f"API Payload: query_rewrite={query_rewrite}, plan_execute={plan_execute}, search_engine={search_engine_choice}, max_tokens={max_tokens}, temperature={temperature}, language={language}")
    print(f"Agent mode state: {agent_mode}")
    
    # Update UI - Add user message to history
    chat_history.append(ChatMessage("user", message))
    
    # Add an empty assistant message for streaming content with localized loading message
    loading_message = UI_TEXT[language]["connecting_api"]
    chat_history.append(ChatMessage("assistant", loading_message))
    
    # Yield an initial update to show the user message and loading indicator
    yield "", chat_history
    
    try:
        # Set up session with retry capability
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Decide which API URL to use based on plan_execute and search_engine
        # If search_engine is grounding_bing, always use API_URL
        if search_engine_choice == "grounding_bing":
            api_url_to_use = API_URL
        else:
            api_url_to_use = DEEP_SEARCH_API_URL if plan_execute else API_URL

        # Make request with stream=True
        response = session.post(
            api_url_to_use, 
            json=payload, 
            timeout=(5, 120),
            stream=True,
            headers={"Accept": "text/event-stream"}
        )
        
        print(f"Response status: {response.status_code}, Content-Type: {response.headers.get('Content-Type', 'unknown')}")
        
        if response.status_code == 200:
            # Clear loading indicator with localized message
            content_type = response.headers.get('Content-Type', '')
            searching_message = UI_TEXT[language]["searching_response"]
            chat_history[-1].content = searching_message
            
            yield "", chat_history
            
            if 'text/event-stream' in content_type:
                # Process Server-Sent Events (SSE)
                buffer = ""
                
                print("Starting SSE processing loop...")
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    # Decode the line
                    line = line.decode('utf-8')
                    print(f"SSE line received: {line}")  # Debugging
                    
                    # Skip SSE comments and empty lines
                    if line.startswith(':') or not line.strip():
                        continue
                    
                    # Handle SSE format (data: prefix)
                    if line.startswith('data: '):
                        line = line[6:].strip()  # Remove the 'data: ' prefix
                            
                        # Status message handling
                        if line.startswith('### '):
                            chat_history[-1].content = f"⟳ {line[4:]}"  # Replace with loading indicator
                            yield "", chat_history
                    else:
                        # Regular content - accumulate
                        if chat_history[-1].content.startswith("⟳"):
                            # Replace the status message with actual content
                            chat_history[-1].content = line
                        else:
                            # Append to existing content with proper line breaks
                            if chat_history[-1].content:
                                # Apply formatting rules for line breaks
                                if line.startswith(('•', '-', '#', '1.', '2.', '3.')) or chat_history[-1].content.endswith(('.', '!', '?', ':')):
                                    chat_history[-1].content += "\n\n" + line
                                else:
                                    chat_history[-1].content += "\n" + line
                            else:
                                chat_history[-1].content = line
                                
                    # Yield update for UI refresh
                    yield "", chat_history

            else:

                # Handle regular non-streaming response
                print("Not a chunked response, trying to process as regular response")
                try:
                    chunks = []
                    for chunk in response.iter_content(chunk_size=None):
                        if chunk:
                            chunks.append(chunk)
                            
                    if chunks:
                        response_text = b''.join(chunks).decode('utf-8', errors='replace')
                        
                        # Try to parse as JSON first
                        try:
                            response_data = json.loads(response_text)
                            if isinstance(response_data, dict) and "content" in response_data:
                                chat_history[-1].content = response_data["content"]
                                print(f"Parsed JSON response with content: {response_data['content'][:30]}...")
                            else:
                                chat_history[-1].content = response_text
                                print("JSON response without content field, using raw text")
                        except json.JSONDecodeError:
                            # Not valid JSON, just use as text
                            chat_history[-1].content = response_text
                            print("Not a valid JSON response, using raw text")
                            
                        # Yield the final response
                        yield "", chat_history
                    else:
                        print("No content received from response")
                        chat_history[-1].content = "No response received from server."
                        yield "", chat_history
                except Exception as e:
                    print(f"Error processing response: {type(e).__name__}: {str(e)}")
                    chat_history[-1].content = f"Error processing response: {str(e)}"
                    yield "", chat_history
        else:
            content_type = response.headers.get('Content-Type', '')
            if content_type.startswith("text/html"):
                chat_history[-1].content = f"Server Error: Received HTML response (status {response.status_code}).\nPlease check if the API server is running and the endpoint is correct."
            else:
                chat_history[-1].content = f"Error: {response.status_code} - {response.text}"
            yield "", chat_history
            
    except requests.exceptions.Timeout:
        print("Request timed out")
        chat_history[-1].content = "Error: Request timed out. The server took too long to respond."
        yield "", chat_history
    except requests.exceptions.ConnectionError:
        print("Connection error")
        chat_history[-1].content = "Error: Connection failed. Please check if the API server is running."
        yield "", chat_history
    except requests.exceptions.ChunkedEncodingError:
        print("Chunked encoding error - connection interrupted")
        chat_history[-1].content = "Error: Connection interrupted while receiving data from the server."
        yield "", chat_history
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {type(e).__name__}: {str(e)}")
        chat_history[-1].content = f"Error connecting to the API: {str(e)}"
        yield "", chat_history
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        chat_history[-1].content = "Error: Received invalid JSON from the server."
        yield "", chat_history
    except Exception as e:
        print(f"Unexpected error in stream_chat_with_api: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()  # Print detailed stack trace for debugging
        chat_history[-1].content = f"Error: {str(e)}"
        yield "", chat_history
    
    # Final yield with either complete response or error message
    print("Streaming completed")
    yield "", chat_history


def format_chat_history(chat_history):
    """Converts the chat_history into the format expected by Gradio's chatbot component"""
    formatted_history = []
    for msg in chat_history:
        if msg.role == "user":
            formatted_history.append((msg.content, None))
        else:
            formatted_history.append((None, msg.content))
    return formatted_history



def main():
    gr.set_static_paths(paths=["public/"])
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), css="""
        /* Prevent system font loading errors */
        @font-face {
            font-family: 'system-ui';
            src: local('system-ui'), local('-apple-system'), local('BlinkMacSystemFont');
            font-display: swap;
        }
        
        @font-face {
            font-family: 'ui-sans-serif';
            src: local('ui-sans-serif'), local('system-ui'), local('-apple-system');
            font-display: swap;
        }
        .container { max-width: 800px; margin: auto; padding-top: 1.5rem; }
        .agent-mode-container { margin-bottom: 2rem; padding: 1rem 0; }
        .mode-card { 
            border: 1px solid #e5e7eb;
            padding: 1rem; 
            border-radius: 8px; 
            background: white; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.05); 
            transition: all 0.2s;
            margin: 0 0.5rem;
        }
        .mode-card:hover { transform: translateY(-2px); box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .mode-switch { margin-top: 10px; }
        .search-engine-selector { margin-top: 2px; }
        /* Fixed prompt container height */
    .prompt-container { 
        display: flex; 
        flex-direction: column;
        gap: 1rem; 
        margin-bottom: 1.5rem;
        height: auto;
        min-height: fit-content;
    }
    
    
        .prompt-card { padding: 1rem; border-radius: 8px; background: white; box-shadow: 0 2px 6px rgba(0,0,0,0.1); flex: 1; min-width: 200px; transition: transform 0.2s; }
        .prompt-card:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .chat-container { height: 500px; overflow-y: auto; padding: 1rem; border-radius: 8px; background: #f9f9f9; margin-bottom: 1rem; }
        .input-container { display: flex; gap: 0.5rem; }
        .msg-user { text-align: right; margin-bottom: 0.5rem; }
        .msg-bot { text-align: left; margin-bottom: 0.5rem; }
        footer { margin-top: 2rem; text-align: center; color: #666; }
        .title { font-weight: 600; margin-bottom: 0.25rem; }
        .description { font-size: 0.875rem; color: #666; }
        .input-textbox { width: 100%; padding: 0.5rem; border-radius: 8px; border: 1px solid #e5e7eb; }
        .submit-button { background-color: #4f46e5; color: white; border-radius: 8px; padding: 0.5rem 1rem; }
        #status-indicator { 
            text-align: center; 
            color: #4f46e5; 
            font-size: 0.9rem;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
            min-height: 24px;
        }
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        .processing { 
            animation: pulse 1.5s ease-in-out infinite;
        }
        #clear-button {
            background-color: #f3f4f6;
            color: #4b5563;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 0.375rem 0.75rem;
            font-size: 0.875rem;
            transition: all 0.2s;
            text-align: center;
        }
        #clear-button:hover {
            background-color: #e5e7eb;
            color: #374151;
        }
        /* Override system font references with web-safe alternatives */
        * {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
        }
        
        /* Disabled search engine styling */
        .disabled-notice {
            font-size: 0.75rem;
            color: #9ca3af;
            font-style: italic;
            margin-top: 0.5rem;
        }

        /* Example button styling */
        .example-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s ease;
            width: 100%;
            margin-top: 8px;
        }
        
        .example-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        .example-button:active {
            transform: translateY(0);
            box-shadow: 0 2px 6px rgba(102, 126, 234, 0.3);
        }
        
        .language-toggle {
            background: #6f42c1;
            color: white;
            border: none;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }
        .language-toggle:hover {
            background: #5a2d91;
        }
        
        
    """) as demo:
        # Language state management
        language_state = gr.State("ko-KR")  # Default to Korean
        
        # Language toggle function
        def toggle_language(current_lang):
            new_lang = "en-US" if current_lang == "ko-KR" else "ko-KR"
            return new_lang
        
        # Function to get localized UI content
        def get_ui_content(lang):
            return UI_TEXT[lang]
        
        # Function to get localized example prompts
        def get_example_prompts(lang):
            return EXAMPLE_PROMPTS[lang]
        
        # Language toggle button
        with gr.Row():
            with gr.Column(scale=10):
                title_md = gr.Markdown(UI_TEXT["en-US"]["title"])
            with gr.Column(scale=2):
                language_toggle_btn = gr.Button(
                    UI_TEXT["ko-KR"]["language_toggle"],
                    elem_classes="language-toggle"
                )
        
        agent_mode_state = gr.State(["query_rewrite", "plan_execute"])  # Initialize with both enabled
        search_engine_state = gr.State(list(SEARCH_ENGINES.values())[0])  
        chat_history = gr.State([])

        select_agent_mode_md = gr.Markdown(UI_TEXT["ko-KR"]["select_agent_mode"])

        # Use cards and toggle switches for better UX
        with gr.Group(elem_classes="agent-mode-container"):
            with gr.Row(equal_height=True):
                # Query Rewrite Card
                with gr.Column(scale=1, elem_classes="mode-card"):
                    query_rewrite_title_md = gr.Markdown(UI_TEXT["ko-KR"]["query_rewrite_title"])
                    query_rewrite_desc_md = gr.Markdown(UI_TEXT["ko-KR"]["query_rewrite_desc"])
                    query_rewrite_switch = gr.Checkbox(
                        value=True,
                        label=UI_TEXT["en-US"]["enable_label"],
                        elem_classes="mode-switch"
                    )
                
                # Plan & Execute Card
                with gr.Column(scale=1, elem_classes="mode-card"):
                    plan_execute_title_md = gr.Markdown(UI_TEXT["ko-KR"]["planning_title"])
                    plan_execute_desc_md = gr.Markdown(UI_TEXT["ko-KR"]["planning_desc"])
                    plan_execute_switch = gr.Checkbox(
                        value=True,  # Set to True by default
                        label=UI_TEXT["en-US"]["enable_label"],
                        elem_classes="mode-switch"
                    )
                
                # Search Engine Card
                with gr.Column(scale=1, elem_classes="mode-card"):
                    search_engine_title_md = gr.Markdown(UI_TEXT["ko-KR"]["search_engine_title"])
                    search_engine_desc_md = gr.Markdown(UI_TEXT["ko-KR"]["search_engine_desc"])
                    
                    search_engine_choice_ui = gr.Radio(
                        choices=list(SEARCH_ENGINES.keys()),
                        value=list(SEARCH_ENGINES.keys())[0],
                        label="",
                        elem_classes="search-engine-selector"
                    )
                    
                    

            # Function to update agent mode based on toggle switches - allow both to be enabled
            def update_agent_mode(query_rewrite, plan_execute):
                modes = []
                if query_rewrite:
                    modes.append("query_rewrite")
                if plan_execute:
                    modes.append("plan_execute")
                return modes
            
            # Remove the functions that made switches mutually exclusive
            # No longer need update_query_rewrite and update_plan_execute functions
            
            # Connect the switches to update the agent mode without affecting each other
            query_rewrite_switch.change(
                update_agent_mode,
                inputs=[query_rewrite_switch, plan_execute_switch],
                outputs=[agent_mode_state],
                queue=False
            )
            
            plan_execute_switch.change(
                update_agent_mode,
                inputs=[query_rewrite_switch, plan_execute_switch],
                outputs=[agent_mode_state],
                queue=False
            )
            
            # Fixed search engine change handler with Google Search prevention
            def update_search_engine_state(choice_display_name):
                """Convert display name to actual API value, but prevent Google Search selection"""
                if choice_display_name == "Google Search":
                    # Return the current first available option instead
                    fallback_choice = list(SEARCH_ENGINES.keys())[0]
                    api_value = SEARCH_ENGINES[fallback_choice]
                    print(f"Google Search disabled, falling back to: '{fallback_choice}' -> '{api_value}'")
                    return api_value, fallback_choice
                else:
                    api_value = SEARCH_ENGINES[choice_display_name] 
                    print(f"Search engine changed: '{choice_display_name}' -> '{api_value}'")
                    return api_value, choice_display_name
            
            # Connect the search engine change handler with fallback UI update
            def handle_search_engine_change(choice_display_name):
                """Handle search engine change and update UI if needed"""
                api_value, actual_choice = update_search_engine_state(choice_display_name)
                return api_value, actual_choice
            
            search_engine_choice_ui.change(
                fn=handle_search_engine_change,
                inputs=[search_engine_choice_ui],
                outputs=[search_engine_state, search_engine_choice_ui],
                queue=False
            )
        
        with gr.Row(elem_classes="chat-container", height=800):
           
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                height=800,
                avatar_images=(None, "public/images/azure-ai-search.png")
            )
        
        with gr.Row(elem_classes="input-container"):
            with gr.Column(scale=10):
                message_input = gr.Textbox(
                    show_label=False,
                    container=False,
                    autofocus=False,
                    elem_classes="input-textbox",
                    elem_id="message-input-box"  # Add a unique ID
                )
            with gr.Column(scale=1):
                submit_button = gr.Button(UI_TEXT["ko-KR"]["send_button"], elem_classes="submit-button")

        # Status indicator and clear chat button
        with gr.Row():
            with gr.Column(scale=8):
                status_indicator = gr.Markdown("", elem_id="status-indicator")
            with gr.Column(scale=2):
                clear_button = gr.Button(UI_TEXT["ko-KR"]["clear_chat_button"], elem_id="clear-button")

        def process_stream(message, history, agent_mode, search_engine, language):
            """
            Fixed streaming function that properly handles inputs and outputs
            """
            if not message or message.strip() == "":
                return "", history, format_chat_history(history)
            
            # Create a generator object to track updates with language parameter
            generator = stream_chat_with_api(message, history, agent_mode, search_engine, language)
            
            # Process each update from the generator
            for _, updated_history in generator:
                # Convert history to chatbot format
                chatbot_ui = format_chat_history(updated_history)
                
                # Yield: empty message input, updated history, updated chatbot UI
                yield "", updated_history, chatbot_ui
        
        # Fix the submit button handlers
        def show_processing_status(language):
            """Function to show processing status"""
            return UI_TEXT[language]["processing_message"]

        def clear_processing_status():
            """Function to clear processing status"""
            return ""

        def clear_message_input():
            """Function to clear the message input"""
            return ""
        
        submit_action = submit_button.click(
            fn=show_processing_status,  # Show processing status
            inputs=[language_state],
            outputs=[status_indicator],
            queue=False,
            api_name=None
        ).then(
            fn=process_stream,  # Use our wrapper function that updates both history and UI
            inputs=[message_input, chat_history, agent_mode_state, search_engine_state, language_state],
            outputs=[message_input, chat_history, chatbot],  # Note: now directly updating the chatbot UI
            api_name="chat_stream",  # Name for the streaming API endpoint
            queue=True
        ).then(
            fn=clear_processing_status,  # Clear status indicator
            inputs=[],
            outputs=[status_indicator],
            queue=False,
            api_name=None
        )
        
        # Also trigger on Enter key
        message_input.submit(
            fn=show_processing_status,  # Set processing status
            inputs=[language_state],
            outputs=[status_indicator],
            queue=False,
            api_name=None
        ).then(
            fn=process_stream,  # Use the same wrapper function for consistent behavior
            inputs=[message_input, chat_history, agent_mode_state, search_engine_state, language_state],
            outputs=[message_input, chat_history, chatbot],  # Update all three components
            api_name=None,  # Don't create duplicate API endpoint
            queue=True
        ).then(
            fn=clear_processing_status,  # Clear status indicator
            inputs=[],
            outputs=[status_indicator],
            queue=False,
            api_name=None,
        )
        
        # Function to clear the chat history
        def clear_chat():
            return [], []
        
        # Connect clear button to reset the chat history
        clear_button.click(
            clear_chat,
            inputs=[],
            outputs=[chatbot, chat_history],
            queue=False,
            api_name=None
        )

        try_prompts_md = gr.Markdown(UI_TEXT["ko-KR"]["try_prompts"])

        with gr.Group(elem_classes="prompt-container"):
            # Create dynamic example prompts section that can be updated
            with gr.Row(equal_height=True):
                # First row of prompt cards
                with gr.Column(scale=1, elem_classes="prompt-card"):
                    prompt1_title = gr.Markdown(f"#### {EXAMPLE_PROMPTS['ko-KR']['question_Microsoft']['title']}")
                    prompt1_desc = gr.Markdown(f"{EXAMPLE_PROMPTS['ko-KR']['question_Microsoft']['description']}")
                    prompt1_btn = gr.Button(
                        f"Try {EXAMPLE_PROMPTS['ko-KR']['question_Microsoft']['title']}", 
                        elem_classes="example-button"
                    )
                
                with gr.Column(scale=1, elem_classes="prompt-card"):
                    prompt2_title = gr.Markdown(f"#### {EXAMPLE_PROMPTS['ko-KR']['product_info']['title']}")
                    prompt2_desc = gr.Markdown(f"{EXAMPLE_PROMPTS['ko-KR']['product_info']['description']}")
                    prompt2_btn = gr.Button(
                        f"Try {EXAMPLE_PROMPTS['ko-KR']['product_info']['title']}", 
                        elem_classes="example-button"
                    )
                
                with gr.Column(scale=1, elem_classes="prompt-card"):
                    prompt3_title = gr.Markdown(f"#### {EXAMPLE_PROMPTS['ko-KR']['recommendation']['title']}")
                    prompt3_desc = gr.Markdown(f"{EXAMPLE_PROMPTS['ko-KR']['recommendation']['description']}")
                    prompt3_btn = gr.Button(
                        f"Try {EXAMPLE_PROMPTS['ko-KR']['recommendation']['title']}", 
                        elem_classes="example-button"
                    )
            
            with gr.Row(equal_height=True):
                # Second row of prompt cards
                with gr.Column(scale=1, elem_classes="prompt-card"):
                    prompt4_title = gr.Markdown(f"#### {EXAMPLE_PROMPTS['ko-KR']['comparison']['title']}")
                    prompt4_desc = gr.Markdown(f"{EXAMPLE_PROMPTS['ko-KR']['comparison']['description']}")
                    prompt4_btn = gr.Button(
                        f"Try {EXAMPLE_PROMPTS['ko-KR']['comparison']['title']}", 
                        elem_classes="example-button"
                    )
                with gr.Column(scale=1, elem_classes="prompt-card"):
                    prompt5_title = gr.Markdown(f"#### {EXAMPLE_PROMPTS['ko-KR']['support_questions']['title']}")
                    prompt5_desc = gr.Markdown(f"{EXAMPLE_PROMPTS['ko-KR']['support_questions']['description']}")
                    prompt5_btn = gr.Button(
                        f"Try {EXAMPLE_PROMPTS['ko-KR']['support_questions']['title']}", 
                        elem_classes="example-button"
                    )

                with gr.Column(scale=1, elem_classes="prompt-card"):
                    prompt6_title = gr.Markdown(f"#### {EXAMPLE_PROMPTS['ko-KR']['tools']['title']}")
                    prompt6_desc = gr.Markdown(f"{EXAMPLE_PROMPTS['ko-KR']['tools']['description']}")
                    prompt6_btn = gr.Button(
                        f"Try {EXAMPLE_PROMPTS['ko-KR']['tools']['title']}", 
                        elem_classes="example-button"
                    )
        
        # Create dynamic button click handlers that respond to current language
        def get_current_prompt(lang, category):
            """Get current prompt text for a category in the specified language"""
            return EXAMPLE_PROMPTS[lang][category]["prompt"]
        
        # Set up button click handlers that use current language state
        prompt1_btn.click(
            fn=lambda lang: get_current_prompt(lang, "question_Microsoft"),
            inputs=[language_state],
            outputs=[message_input],
            queue=False
        )
        
        prompt2_btn.click(
            fn=lambda lang: get_current_prompt(lang, "product_info"),
            inputs=[language_state],
            outputs=[message_input],
            queue=False
        )
        prompt3_btn.click(
            fn=lambda lang: get_current_prompt(lang, "recommendation"),
            inputs=[language_state],
            outputs=[message_input],
            queue=False
        )
        
        prompt4_btn.click(
            fn=lambda lang: get_current_prompt(lang, "comparison"),
            inputs=[language_state],
            outputs=[message_input],
            queue=False
        )
        
        prompt5_btn.click(
            fn=lambda lang: get_current_prompt(lang, "support_questions"),
            inputs=[language_state],
            outputs=[message_input],
            queue=False
        )
        
        prompt6_btn.click(
            fn=lambda lang: get_current_prompt(lang, "tools"),
            inputs=[language_state],
            outputs=[message_input],
            queue=False
        )
        
        # Enhanced language toggle functionality that updates everything
        def handle_language_toggle(current_lang):
            new_lang = toggle_language(current_lang)
            ui_text = get_ui_content(new_lang)
            example_prompts = get_example_prompts(new_lang)
            
            # Return comprehensive UI updates including all prompt cards
            return [
                new_lang,  # language_state
                ui_text["title"],  # title_md
                ui_text["select_agent_mode"],  # select_agent_mode_md
                ui_text["query_rewrite_title"],  # query_rewrite_title_md
                ui_text["query_rewrite_desc"],  # query_rewrite_desc_md
                ui_text["plan_execute_title"],  # plan_execute_title_md
                ui_text["plan_execute_desc"],  # plan_execute_desc_md
                ui_text["search_engine_title"],  # search_engine_title_md
                ui_text["search_engine_desc"],  # search_engine_desc_md
                ui_text["send_button"],  # submit_button
                ui_text["clear_chat_button"],  # clear_button
                ui_text["try_prompts"],  # try_prompts_md
                ui_text["language_toggle"],  # language_toggle_btn
                # Update all prompt cards
                f"#### {example_prompts['question_Microsoft']['title']}",  # prompt1_title
                example_prompts['question_Microsoft']['description'],  # prompt1_desc
                f"Try {example_prompts['question_Microsoft']['title']}",  # prompt1_btn
                f"#### {example_prompts['product_info']['title']}",  # prompt2_title
                example_prompts['product_info']['description'],  # prompt2_desc
                f"Try {example_prompts['product_info']['title']}",  # prompt2_btn
                f"#### {example_prompts['recommendation']['title']}",  # prompt3_title
                example_prompts['recommendation']['description'],  # prompt3_desc
                f"Try {example_prompts['recommendation']['title']}",  # prompt3_btn
                f"#### {example_prompts['comparison']['title']}",  # prompt4_title
                example_prompts['comparison']['description'],  # prompt4_desc
                f"Try {example_prompts['comparison']['title']}",  # prompt4_btn
                f"#### {example_prompts['support_questions']['title']}",  # prompt5_title
                example_prompts['support_questions']['description'],  # prompt5_desc
                f"Try {example_prompts['support_questions']['title']}",  # prompt5_btn
                f"#### {example_prompts['tools']['title']}",  # prompt6_title
                example_prompts['tools']['description'],  # prompt6_desc
                f"Try {example_prompts['tools']['title']}",  # prompt6_btn
            ]
        
        # Connect comprehensive language toggle
        language_toggle_btn.click(
            fn=handle_language_toggle,
            inputs=[language_state],
            outputs=[
                language_state,
                title_md,
                select_agent_mode_md,
                query_rewrite_title_md,
                query_rewrite_desc_md,
                plan_execute_title_md,
                plan_execute_desc_md,
                search_engine_title_md,
                search_engine_desc_md,
                submit_button,
                clear_button,
                try_prompts_md,
                language_toggle_btn,
                # All prompt card components
                prompt1_title,
                prompt1_desc,
                prompt1_btn,
                prompt2_title,
                prompt2_desc,
                prompt2_btn,
                prompt3_title,
                prompt3_desc,
                prompt3_btn,
                prompt4_title,
                prompt4_desc,
                prompt4_btn,
                prompt5_title,
                prompt5_desc,
                prompt5_btn,
                prompt6_title,
                prompt6_desc,
                prompt6_btn,
            ],
            queue=False
        )
                        
        
    demo.queue(
        default_concurrency_limit=30,
        max_size=50,
        api_open=False
    ).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        max_threads=30,
        share=True,
        auth=auth_credentials
    )

if __name__ == "__main__":
    main()
