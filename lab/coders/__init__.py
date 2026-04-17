"""Coders do Lab — transformam Proposal em PatchProposal (diff + rationale).

Tres backends plugaveis:

- `ollama_coder.OllamaCoder` — default, 100% local via Ollama (Qwen2.5-Coder-7B
  ou DeepSeek-Coder-V2). Custo zero.
- `claude_code_coder.ClaudeCodeCoder` — opt-in, invoca `claude -p` CLI. Custo
  aparece apenas quando o usuario clica "Gerar patch" na UI.
- `lab.agents.coder_agent.ClaudeSDKCoder` — usa Anthropic SDK diretamente com
  contexto do agente especialista (.claude/agents/*.md). Requer ANTHROPIC_API_KEY.

Todos os coders retornam `PatchProposal(patch_unified_diff, files_affected,
rationale)` — NUNCA aplicam patch automaticamente. A aplicacao final eh
responsabilidade do Rust/UI apos aprovacao humana.
"""
from __future__ import annotations

from lab.coders.base import Coder, PatchProposal, build_local_patch_from_hint

__all__ = [
    "Coder",
    "PatchProposal",
    "build_local_patch_from_hint",
]
