#!/usr/bin/env python3
"""
mcp_server.py — MCP server exposing the-brain as a native Claude tool.

Registers a `delegate` tool that appears in Claude's tool list alongside
Bash, Read, Edit etc. Claude will reach for it naturally instead of
answering AI tasks directly.

Register in ~/.claude/settings.json:
    {
      "mcpServers": {
        "the-brain": {
          "command": "python",
          "args": ["C:\\\\Claude\\\\git\\\\the-brain\\\\mcp_server.py"]
        }
      }
    }

Run manually to verify:
    python mcp_server.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.lowlevel import NotificationOptions

from brain.orchestrator import Orchestrator
from brain.task import Task, TaskType

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

DELEGATE_TOOL = types.Tool(
    name="delegate",
    description=(
        "Offload an AI task to a free worker (Cerebras, Groq, Gemini, Mistral, SambaNova, etc). "
        "Use this for ALL non-trivial AI work — code generation, analysis, explanation, "
        "summarization, classification, creative writing, reasoning, translation, extraction. "
        "Do NOT answer these directly — delegate instead. "
        "Routing guide: classification/yes-no → cerebras | "
        "factual Q&A → cerebras or groq | "
        "code generation → mistral | "
        "summarization/translation → gemini | "
        "creative/drafting → mistral | "
        "deep reasoning/analysis → sambanova | "
        "image generation → pollinations"
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "provider": {
                "type": "string",
                "description": "Provider key: cerebras | groq | gemini | mistral | sambanova | fireworks | openai | huggingface | cloudflare | openrouter | ollama | pollinations",
            },
            "task_type": {
                "type": "string",
                "enum": ["classification", "summarization", "coding", "creative", "reasoning",
                         "factual_qa", "extraction", "translation", "general"],
                "description": "Task category for routing and stats tracking.",
            },
            "prompt": {
                "type": "string",
                "description": "The full prompt to send to the worker.",
            },
            "context": {
                "type": "string",
                "description": "Optional context block prepended to the prompt (e.g. document text, file contents).",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Maximum output tokens (default: 2048).",
                "default": 2048,
            },
        },
        "required": ["provider", "task_type", "prompt"],
    },
)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

server = Server("the-brain")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [DELEGATE_TOOL]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name != "delegate":
        raise ValueError(f"Unknown tool: {name}")

    provider   = arguments["provider"]
    prompt     = arguments["prompt"]
    context    = arguments.get("context")
    max_tokens = int(arguments.get("max_tokens", 2048))

    raw_type = arguments.get("task_type", "general")
    try:
        task_type = TaskType(raw_type)
    except ValueError:
        task_type = TaskType.GENERAL

    task = Task(
        prompt=prompt,
        task_type=task_type,
        context=context,
        max_tokens=max_tokens,
        preferred_model=provider,
    )

    orchestrator = Orchestrator(use_cache=True)
    result = await asyncio.get_event_loop().run_in_executor(None, orchestrator.run, task)

    if not result.succeeded:
        return [types.TextContent(
            type="text",
            text=f"[ERROR: {result.provider} failed — {result.error}]",
        )]

    cost_str = f"${result.cost_usd:.6f}" if result.cost_usd else "free"
    header = (
        f"[{result.provider} / {result.model} | "
        f"{result.tokens_used} tokens | "
        f"{result.latency_ms:.0f}ms | {cost_str}]"
    )

    return [types.TextContent(
        type="text",
        text=f"{header}\n\n{result.content}",
    )]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="the-brain",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
