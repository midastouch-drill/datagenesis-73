"""
AI Configuration Routes
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

from ..services.ai_service import ai_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ai", tags=["ai-config"])

class AIConfigRequest(BaseModel):
    provider: str
    model: str
    api_key: str
    endpoint: Optional[str] = None

@router.post("/configure")
async def configure_ai_model(config: AIConfigRequest, request: Request):
    """Configure AI model with user-provided credentials"""
    
    try:
        # Log configuration attempt (without sensitive data)
        logger.info(f"🔧 Configuring AI: {config.provider} - {config.model}")
        
        # Validate provider
        supported_providers = ['gemini', 'openai', 'anthropic', 'ollama']
        if config.provider not in supported_providers:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported provider: {config.provider}. Supported: {supported_providers}"
            )
        
        # Configure the AI service
        success = await ai_service.configure(
            provider=config.provider,
            model=config.model,
            api_key=config.api_key,
            endpoint=config.endpoint
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to configure AI service"
            )
        
        # Test the configuration
        health_result = await ai_service.health_check()
        
        return {
            "status": "success",
            "message": f"Successfully configured {config.provider} with model {config.model}",
            "provider": config.provider,
            "model": config.model,
            "endpoint": config.endpoint,
            "health": health_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ AI configuration error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def get_ai_health():
    """Get AI service health status"""
    try:
        if not ai_service.is_initialized:
            return {
                "status": "not_configured",
                "message": "AI service not configured",
                "provider": None,
                "model": None
            }
        
        health_result = await ai_service.health_check()
        return health_result
        
    except Exception as e:
        logger.error(f"❌ AI health check error: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "provider": ai_service.current_provider,
            "model": ai_service.current_model
        }

@router.get("/status")
async def get_ai_status():
    """Get current AI configuration status"""
    return {
        "is_configured": ai_service.is_initialized,
        "provider": ai_service.current_provider,
        "model": ai_service.current_model,
        "endpoint": ai_service.endpoint if ai_service.current_provider == 'ollama' else None
    }

@router.get("/providers")
async def get_supported_providers():
    """Get list of supported AI providers and their models"""
    return {
        "providers": {
            "gemini": {
                "name": "Google Gemini",
                "models": [
                    "gemini-1.5-flash",
                    "gemini-1.5-pro", 
                    "gemini-2.0-flash-exp",
                    "gemini-1.0-pro"
                ],
                "requires_api_key": True,
                "api_key_format": "AIzaSy..."
            },
            "openai": {
                "name": "OpenAI GPT",
                "models": [
                    "gpt-4",
                    "gpt-4-turbo",
                    "gpt-3.5-turbo",
                    "gpt-4o",
                    "gpt-4o-mini"
                ],
                "requires_api_key": True,
                "api_key_format": "sk-..."
            },
            "anthropic": {
                "name": "Anthropic Claude",
                "models": [
                    "claude-3-sonnet-20240229",
                    "claude-3-haiku-20240307",
                    "claude-3-opus-20240229",
                    "claude-3-5-sonnet-20241022"
                ],
                "requires_api_key": True,
                "api_key_format": "sk-ant-..."
            },
            "ollama": {
                "name": "Ollama (Local)",
                "models": [
                    "llama3.2:3b",  # ~2GB RAM - Recommended for 4GB systems
                    "phi3:3.8b",    # ~2.3GB RAM - Good for coding
                    "llama3:8b",    # ~5.9GB RAM - Requires 8GB+ system
                    "mistral:7b",   # ~4.1GB RAM - Requires 6GB+ system
                    "codellama:7b", # ~4.1GB RAM - Requires 6GB+ system
                    "llama2:7b",    # ~4.1GB RAM - Requires 6GB+ system
                    "llama3:70b",   # ~40GB RAM - Requires high-end systems
                    "custom"
                ],
                "requires_api_key": False,
                "requires_endpoint": True,
                "default_endpoint": "http://localhost:11434",
                "memory_requirements": {
                    "llama3.2:3b": "~2GB RAM",
                    "phi3:3.8b": "~2.3GB RAM", 
                    "llama3:8b": "~5.9GB RAM",
                    "mistral:7b": "~4.1GB RAM",
                    "codellama:7b": "~4.1GB RAM",
                    "llama2:7b": "~4.1GB RAM",
                    "llama3:70b": "~40GB RAM"
                }
            }
        }
    }

@router.post("/test-connection")
async def test_ai_connection():
    """Test the current AI configuration with a simple prompt"""
    try:
        if not ai_service.is_initialized:
            raise HTTPException(
                status_code=400,
                detail="AI service not configured"
            )
        
        # For Ollama, use a simpler test to avoid complex generation
        if ai_service.current_provider == 'ollama':
            # Import the Ollama service to test simple completion
            from ..services.ollama_service import OllamaService
            ollama_service = OllamaService(ai_service.endpoint or 'http://localhost:11434')
            ollama_service.configure_model(ai_service.current_model, ai_service.endpoint)
            await ollama_service.initialize()
            
            # Test with simple completion
            test_response = await ollama_service.generate_completion("Say 'Hello DataGenesis!' if you can respond.")
            
            return {
                "status": "success",
                "message": "Ollama connection test successful",
                "provider": ai_service.current_provider,
                "model": ai_service.current_model,
                "test_result": {
                    "response_received": bool(test_response),
                    "response_length": len(test_response) if test_response else 0,
                    "sample_response": test_response[:100] + "..." if len(test_response) > 100 else test_response
                }
            }
        else:
            # For other providers, use schema generation test
            test_result = await ai_service.generate_schema_from_natural_language(
                description="A simple user profile with name and email",
                domain="general",
                data_type="tabular"
            )
            
            return {
                "status": "success",
                "message": "AI connection test successful",
                "provider": ai_service.current_provider,
                "model": ai_service.current_model,
                "test_result": {
                    "fields_generated": len(test_result.get('schema', {})),
                    "detected_domain": test_result.get('detected_domain', 'unknown')
                }
            }
        
    except Exception as e:
        logger.error(f"❌ AI connection test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))