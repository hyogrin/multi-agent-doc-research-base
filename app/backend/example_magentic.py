#!/usr/bin/env python3
"""
Example usage of MagenticExecutor with Microsoft Agent Framework.

This example demonstrates how to use the Magentic orchestration pattern
for intelligent multi-agent research.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

async def main():
    """
    Example: Using MagenticExecutor for research
    """
    from services_afw.plan_search_orchestrator_afw import PlanSearchOrchestratorAFW
    from config.config import Settings
    from model.models import ChatMessage
    
    print("=" * 80)
    print("MagenticExecutor Example - Intelligent Research Orchestration")
    print("=" * 80)
    
    # Initialize settings
    settings = Settings()
    
    # Create orchestrator
    orchestrator = PlanSearchOrchestratorAFW(settings)
    
    # Example research question
    messages = [
        ChatMessage(
            role="user",
            content="Compare the advantages and disadvantages of microservices vs monolithic architecture for enterprise applications. Include considerations for scalability, deployment, and maintenance."
        )
    ]
    
    print("\n📋 Research Question:")
    print(messages[0].content)
    print("\n" + "=" * 80)
    print("\n🎯 Starting Magentic orchestration...\n")
    
    # Generate response with Magentic pattern
    response_text = ""
    async for chunk in orchestrator.generate_response(
        messages=messages,
        research=True,
        planning=True,
        multi_agent_type="afw_magentic",  # ✅ Use Magentic pattern
        stream=True,
        locale="en-US",
        include_web_search=False,  # Disable external searches for quick demo
        include_ai_search=False,
        include_ytb_search=False,
        verbose=True
    ):
        # Print streaming output
        if chunk.startswith("data: "):
            # Progress message
            progress = chunk[6:].strip()
            if progress and not progress.startswith("__TTFT"):
                print(progress)
        else:
            # Content
            print(chunk, end="", flush=True)
            response_text += chunk
    
    print("\n\n" + "=" * 80)
    print("✅ Magentic orchestration completed!")
    print("=" * 80)
    
    print(f"\n📊 Response length: {len(response_text)} characters")
    
    print("\n💡 Key Features Demonstrated:")
    print("  ✅ Intelligent orchestrator coordination")
    print("  ✅ Specialized ResearchAnalyst and ResearchWriter agents")
    print("  ✅ Streaming progress and results")
    print("  ✅ Adaptive multi-round collaboration")
    print("  ✅ Structured markdown output")


if __name__ == "__main__":
    # Check required environment variables
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file")
        exit(1)
    
    # Run the example
    asyncio.run(main())
