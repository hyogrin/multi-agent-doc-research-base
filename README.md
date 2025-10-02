# Multi-Agent Doc Research
An AI-powered chatbot that leverages planning and search to provide accurate and context-aware responses to user queries. Built with FastAPI, Azure OpenAI, and Gradio, this project demonstrates advanced techniques for enhancing LLM applications with modular design, evaluation support, and flexible UI options.

![multi-agent-doc-research-architecture-Page-1.jpg](/images/multi-agent-doc-research-architecture-Page-1.jpg)


### 🧠 Semantic Kernel Integration
- The chatbot now incorporates [Semantic Kernel](https://github.com/microsoft/semantic-kernel), Microsoft's open-source orchestration SDK for LLM apps.
- Enables more intelligent planning and contextual understanding, resulting in richer, more accurate responses.
- Supports planner-based execution and native function calling for complex multi-step tasks.


### 🔍 Verbose Mode
- Introduced **verbose mode** for improved debugging and traceability.
- Logs include:
  - Raw input/output data
  - API call history
  - Function invocation details
- Helps track down issues and optimize prompt behavior.

### 🎨 UI Framework
- Now supports the following UI framework:
  - [Chainlit](https://github.com/Chainlit/chainlit) – great for interactive prototyping
  
### 🔁 Query Rewrite
- A module that reformulates user queries to improve response quality and informativeness.  
- Helps the LLM better understand the user’s intent and generate more accurate, context-aware answers.

### 🧭 Plan & Execute
- Implements planning techniques to **enrich search keywords** based on the original query context.  
- Automatically decomposes **complex questions into sub-queries**, searches them, and returns synthesized context to the chatbot.  
- Boosts performance in multi-intent or multi-hop question scenarios.

# Project Structure

The project is organized into two main parts:

- `backend`: Contains the FastAPI server and all backend functionality
- `frontend`: Contains the frontend UI

