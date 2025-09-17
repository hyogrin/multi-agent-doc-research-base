"""
Test Script for Enhanced Document Upload System

This script provides basic testing functionality for the upload system.
Run this to validate your setup before full integration.
"""

import os
import asyncio
import tempfile
import logging
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_enhanced_upload_plugin():
    """Test the enhanced upload plugin functionality."""
    try:
        from services_sk.enhanced_upload_doc_plugin import EnhancedUploadDocPlugin
        
        print("🧪 Testing Enhanced Upload Plugin...")
        
        # Initialize plugin
        upload_plugin = EnhancedUploadDocPlugin()
        print("✅ Plugin initialized successfully")
        
        # Create a test file
        test_content = """
        This is a test document for upload validation.
        
        Page 1 Content:
        This document contains sample text to test the chunking functionality.
        It includes multiple paragraphs and sections to validate proper processing.
        
        Page 2 Content:
        Additional content to test page-based chunking.
        The system should create intelligent chunks from this content.
        
        Summary:
        This test document validates the upload and processing pipeline.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            test_file_path = f.name
        
        print(f"📄 Created test file: {test_file_path}")
        
        # Test document upload
        print("🚀 Testing document upload...")
        result = await upload_plugin.upload_documents(
            file_paths=[test_file_path],
            document_type="TEST_DOCUMENT",
            company="Test Company",
            industry="Testing",
            report_year="2024",
            force_upload=True
        )
        
        print("📊 Upload Results:")
        print(f"  Status: {result['status']}")
        print(f"  Total Files: {result['total_files']}")
        print(f"  Successful: {result['successful_uploads']}")
        print(f"  Failed: {result['failed_uploads']}")
        
        if result['successful_uploads'] > 0:
            print("✅ Upload test PASSED")
            
            # Test document existence check
            print("\n🔍 Testing document existence check...")
            check_result = await upload_plugin.check_documents_exist([Path(test_file_path).name])
            print(f"  Check Status: {check_result['status']}")
            print(f"  File Exists: {check_result['results'][Path(test_file_path).name]['exists']}")
            
        else:
            print("❌ Upload test FAILED")
            
        # Clean up
        os.unlink(test_file_path)
        print("🧹 Cleaned up test file")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all dependencies are installed")
    except Exception as e:
        print(f"❌ Test Error: {e}")
        logger.exception("Upload plugin test failed")


async def test_file_upload_manager_plugin():
    """Test the file upload manager plugin for chat integration."""
    try:
        from services_sk.file_upload_manager_plugin import FileUploadManagerPlugin
        
        print("\n🧪 Testing File Upload Manager Plugin...")
        
        # Initialize plugin
        manager_plugin = FileUploadManagerPlugin()
        print("✅ Manager plugin initialized successfully")
        
        # Test upload request handling
        print("🤖 Testing upload request detection...")
        result1 = await manager_plugin.handle_upload_request(
            user_message="I want to upload some documents",
            session_id="test_session"
        )
        
        print(f"  Status: {result1['status']}")
        print(f"  Action Required: {result1['action_required']}")
        
        if result1['status'] == 'upload_requested':
            print("✅ Upload request detection PASSED")
        else:
            print("❌ Upload request detection FAILED")
        
        # Test capabilities explanation
        print("\n📖 Testing capabilities explanation...")
        result2 = await manager_plugin.explain_upload_capabilities()
        
        print(f"  Status: {result2['status']}")
        print(f"  Has capabilities info: {'capabilities' in result2}")
        
        if result2['status'] == 'info':
            print("✅ Capabilities explanation PASSED")
        else:
            print("❌ Capabilities explanation FAILED")
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure the file upload manager plugin is available")
    except Exception as e:
        print(f"❌ Test Error: {e}")
        logger.exception("Manager plugin test failed")


def test_api_endpoint():
    """Test the API endpoint (requires server to be running)."""
    try:
        import requests
        
        print("\n🧪 Testing API Endpoint...")
        
        # Test health check first
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is running")
            else:
                print("❌ Server health check failed")
                return
        except requests.exceptions.ConnectionError:
            print("❌ Server not running. Start the server with: uvicorn main:app --reload")
            return
        
        # Create test file for upload
        test_content = "Test document content for API validation."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            test_file_path = f.name
        
        # Test file upload endpoint
        print("📤 Testing file upload endpoint...")
        
        with open(test_file_path, 'rb') as file:
            files = {'files': ('test_document.txt', file, 'text/plain')}
            data = {
                'document_type': 'TEST_DOCUMENT',
                'company': 'Test Company',
                'industry': 'Testing',
                'report_year': '2024',
                'force_upload': True
            }
            
            response = requests.post(
                "http://localhost:8000/upload_files",
                files=files,
                data=data,
                timeout=30
            )
        
        print(f"  Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"  Message: {result['message']}")
            print("✅ API endpoint test PASSED")
        else:
            print(f"  Error: {response.text}")
            print("❌ API endpoint test FAILED")
        
        # Clean up
        os.unlink(test_file_path)
        
    except ImportError:
        print("❌ requests library not available. Install with: pip install requests")
    except Exception as e:
        print(f"❌ API test error: {e}")


def check_environment():
    """Check if required environment variables are set."""
    print("🔧 Checking Environment Configuration...")
    
    required_vars = [
        "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        "AZURE_DOCUMENT_INTELLIGENCE_API_KEY",
        "AZURE_AI_SEARCH_ENDPOINT", 
        "AZURE_AI_SEARCH_API_KEY",
        "AZURE_AI_SEARCH_INDEX_NAME",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            print(f"  ✅ {var}: Set")
    
    if missing_vars:
        print(f"\n❌ Missing environment variables:")
        for var in missing_vars:
            print(f"  • {var}")
        print("\nPlease set these variables before running tests.")
        return False
    else:
        print("✅ All required environment variables are set")
        return True


async def run_all_tests():
    """Run all tests in sequence."""
    print("🚀 Enhanced Document Upload System - Test Suite")
    print("=" * 60)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please configure required variables.")
        return
    
    # Run plugin tests
    await test_enhanced_upload_plugin()
    await test_file_upload_manager_plugin()
    
    # Test API endpoint (optional - requires server)
    test_api_endpoint()
    
    print("\n🎉 Test Suite Complete!")
    print("\nNext Steps:")
    print("1. If all tests pass, integrate with Chainlit frontend")
    print("2. Follow UPLOAD_INTEGRATION_GUIDE.md for frontend setup")
    print("3. Test end-to-end flow with real documents")


def main():
    """Main test runner."""
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        logger.exception("Test suite failed")


if __name__ == "__main__":
    main()
