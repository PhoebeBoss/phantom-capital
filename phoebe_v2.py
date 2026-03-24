#!/usr/bin/env python3
"""
PHOEBE v2 - OpenClaw Orchestrator
Rebuilt on open-source LLMs (no Anthropic dependency)
Uses: Together AI, Groq, Ollama, vLLM
"""

import asyncio
import json
import logging
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('phoebe')


class LLMProvider(Enum):
    """Available open-source LLM providers"""
    GROQ = "groq"  # Free, fast
    TOGETHER = "together"  # Free tier available
    OLLAMA = "ollama"  # Local (best privacy)
    OPENROUTER = "openrouter"  # Multi-model aggregator


@dataclass
class PhoebConfig:
    """Phoebe configuration"""
    provider: LLMProvider = LLMProvider.GROQ
    model: str = "llama-3.3-70b-versatile"  # Free on Groq
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    zeabur_api_key: str = os.getenv("ZEABUR_API_KEY", "")
    api_key: str = ""  # Provider-specific API key
    temperature: float = 0.7
    max_tokens: int = 1024


class GroqLLM:
    """Groq LLM integration (free tier, no API key required for public models)"""
    
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"
        self.api_key = os.getenv("GROQ_API_KEY", "")
        
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set. Get free key: https://console.groq.com/keys")
    
    async def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate text using Groq API"""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.7
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data['choices'][0]['message']['content']
                else:
                    logger.error(f"Groq API error: {response.status_code}")
                    return f"Error: {response.status_code}"
        
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            return f"Error: {str(e)}"


class TogetherAILLM:
    """Together AI integration (free tier available)"""
    
    def __init__(self, model: str = "meta-llama/Llama-2-70b-chat-hf"):
        self.model = model
        self.api_key = os.getenv("TOGETHER_API_KEY", "")
        
        if not self.api_key:
            logger.warning("TOGETHER_API_KEY not set. Get free key: https://www.together.ai/")
    
    async def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate text using Together AI"""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.together.xyz/inference",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": 0.7
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data['output']['choices'][0]['text']
                else:
                    logger.error(f"Together AI error: {response.status_code}")
                    return f"Error: {response.status_code}"
        
        except Exception as e:
            logger.error(f"Together AI generation failed: {e}")
            return f"Error: {str(e)}"


class OllamaLLM:
    """Ollama local LLM (100% private, runs on your hardware)"""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    async def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate text using local Ollama"""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "num_predict": max_tokens,
                        "temperature": 0.7,
                        "stream": False
                    },
                    timeout=120.0  # Local can be slower
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data['response']
                else:
                    logger.error(f"Ollama error: {response.status_code}")
                    return f"Error: {response.status_code}"
        
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"Error: {str(e)}"


class Phoebe:
    """Phoebe AI Agent - OpenClaw Orchestrator"""
    
    def __init__(self, config: PhoebConfig):
        self.config = config
        self.llm = self._init_llm()
        self.memory_file = os.path.expanduser("~/.openclaw/MEMORY.md")
        self.memory = self._load_memory()
        self.services = {}
        
        logger.info(f"Phoebe initialized with {config.provider.value}")
        logger.info(f"Model: {config.model}")
    
    def _init_llm(self):
        """Initialize LLM provider"""
        if self.config.provider == LLMProvider.GROQ:
            return GroqLLM(self.config.model)
        elif self.config.provider == LLMProvider.TOGETHER:
            return TogetherAILLM(self.config.model)
        elif self.config.provider == LLMProvider.OLLAMA:
            return OllamaLLM(self.config.model)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
    
    def _load_memory(self) -> Dict[str, Any]:
        """Load memory from disk"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    content = f.read()
                return {
                    "last_updated": datetime.now().isoformat(),
                    "content": content,
                    "status": "loaded"
                }
        except Exception as e:
            logger.warning(f"Could not load memory: {e}")
        
        return {
            "last_updated": datetime.now().isoformat(),
            "content": "# Phoebe's Memory\n## Status: Online",
            "status": "initialized"
        }
    
    def _save_memory(self):
        """Save memory to disk"""
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, 'w') as f:
                f.write(self.memory["content"])
            self.memory["last_updated"] = datetime.now().isoformat()
        except Exception as e:
            logger.error(f"Could not save memory: {e}")
    
    async def think(self, prompt: str) -> str:
        """Generate response from LLM"""
        logger.info(f"Thinking about: {prompt[:50]}...")
        
        # Build context with memory
        context = f"""You are Phoebe, an autonomous AI agent managing Phantom Capital.
        
Current context:
{self.memory['content']}

User request: {prompt}

Respond concisely and directly. Take action if needed."""
        
        response = await self.llm.generate(context, self.config.max_tokens)
        logger.info(f"Generated response: {response[:100]}...")
        return response
    
    async def handle_telegram_message(self, message: str) -> str:
        """Handle incoming Telegram message"""
        logger.info(f"Telegram message: {message}")
        
        # Simple command parsing
        if message.startswith("/status"):
            return self._get_status()
        elif message.startswith("/memory"):
            return self.memory["content"][:500]
        elif message.startswith("/help"):
            return self._get_help()
        else:
            # Let LLM handle it
            return await self.think(message)
    
    def _get_status(self) -> str:
        """Get current status"""
        return f"""
Phoebe Status Report
===================
Provider: {self.config.provider.value}
Model: {self.config.model}
Memory: {self.memory['status']}
Last Updated: {self.memory['last_updated']}
Services: {len(self.services)} active
        """.strip()
    
    def _get_help(self) -> str:
        """Get help message"""
        return """
Phoebe Commands
===============
/status - Show current status
/memory - View memory/context
/help - Show this help
/restart - Restart service
/deploy - Deploy new service
/logs - View recent logs

Or just send a message and I'll respond!
        """.strip()
    
    async def run_telegram_bot(self):
        """Run Telegram bot interface"""
        if not self.config.telegram_token:
            logger.warning("TELEGRAM_BOT_TOKEN not set, skipping bot startup")
            return
        
        try:
            from telegram import Update
            from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
        except ImportError:
            logger.error("python-telegram-bot not installed")
            logger.error("Install with: pip install python-telegram-bot")
            return
        
        app = Application.builder().token(self.config.telegram_token).build()
        
        async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await update.message.reply_text("Phoebe online! Send commands or messages.")
        
        async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            response = await self.handle_telegram_message(update.message.text)
            await update.message.reply_text(response)
        
        app.add_handler(CommandHandler("start", start_handler))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
        
        logger.info("Starting Telegram bot...")
        await app.run_polling()


async def health_check_server(phoebe: 'Phoebe'):
    """Simple health check HTTP server for Zeabur"""
    try:
        from aiohttp import web
        
        async def health(request):
            return web.json_response({
                "status": "healthy",
                "provider": phoebe.config.provider.value,
                "model": phoebe.config.model,
                "timestamp": datetime.now().isoformat()
            })
        
        app = web.Application()
        app.router.add_get('/health', health)
        app.router.add_get('/', health)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 8000)
        await site.start()
        
        logger.info("Health check server running on :8000")
    except Exception as e:
        logger.warning(f"Could not start health check server: {e}")


async def main():
    """Main entry point"""
    # Auto-detect best provider
    provider = LLMProvider.GROQ  # Default: fast, free
    
    # Check for API keys to determine provider
    if os.getenv("GROQ_API_KEY"):
        provider = LLMProvider.GROQ
        logger.info("Using Groq (GROQ_API_KEY detected)")
    elif os.getenv("TOGETHER_API_KEY"):
        provider = LLMProvider.TOGETHER
        logger.info("Using Together AI (TOGETHER_API_KEY detected)")
    elif os.getenv("OLLAMA_BASE_URL"):
        provider = LLMProvider.OLLAMA
        logger.info("Using Ollama (OLLAMA_BASE_URL detected)")
    
    config = PhoebConfig(provider=provider)
    phoebe = Phoebe(config)
    
    logger.info("="*50)
    logger.info("PHOEBE v2 INITIALIZED")
    logger.info(f"Provider: {config.provider.value}")
    logger.info(f"Model: {config.model}")
    logger.info(f"Status: ONLINE")
    logger.info("="*50)
    
    # Run health check server and Telegram bot concurrently
    await asyncio.gather(
        health_check_server(phoebe),
        phoebe.run_telegram_bot(),
        return_exceptions=True
    )


if __name__ == "__main__":
    asyncio.run(main())
