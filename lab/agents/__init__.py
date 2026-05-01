"""Agentes Claude do Lab — reviewers e coders com contexto especializado.

Cada agente carrega automaticamente o `.claude/agents/*.md` correto baseado no
`target_file` ou `touched_domains` da proposta, injetando as constraints do
projeto (FT2Font, sem ProcessPool, etc.) no system prompt.

Se ANTHROPIC_API_KEY não estiver disponível ou o pacote `anthropic` não estiver
instalado, todos os agentes caem silenciosamente para a implementação rule-based
equivalente — o Lab nunca quebra por ausência de LLM.
"""
from __future__ import annotations

from lab.agents.base import get_agent_prompt, is_claude_available
from lab.agents.coder_agent import ClaudeSDKCoder
from lab.agents.reviewer_agent import ClaudeReviewerAgent

__all__ = [
    "ClaudeSDKCoder",
    "ClaudeReviewerAgent",
    "get_agent_prompt",
    "is_claude_available",
]
