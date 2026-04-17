"""ClaudeSDKCoder — gera patches via Claude API com contexto do agente especialista.

Fluxo:
1. Tenta `build_local_patch_from_hint` (sem LLM, determinístico).
2. Se não resolver, carrega o .claude/agents/*.md correto para o target_file.
3. Lê o conteúdo atual do arquivo alvo (até 400 linhas).
4. Chama Claude com system=agente + user=proposta+arquivo.
5. Extrai o diff unificado da resposta e retorna PatchProposal.

Fallback: se ANTHROPIC_API_KEY ausente ou `anthropic` não instalado, retorna
PatchProposal com error preenchido (nunca lança exceção).
"""
from __future__ import annotations

import datetime
import re
from pathlib import Path

from lab.agents.base import (
    find_repo_root,
    get_agent_prompt,
    is_claude_available,
    make_client,
    read_file_slice,
)
from lab.coders.base import PatchProposal, build_local_patch_from_hint
from lab.coders.ollama_coder import _extract_diff_from_response  # reutiliza parser

MODEL = "claude-sonnet-4-5"
MAX_FILE_LINES = 400
MAX_TOKENS = 4096

_DEFAULT_SYSTEM = """\
Você é um engenheiro sênior especialista em Python trabalhando no projeto TraduzAi \
— uma aplicação desktop de tradução automática de mangá.

Sua tarefa exclusiva: gerar unified diffs precisos para corrigir problemas detectados \
automaticamente no pipeline. Sempre use o formato:

```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ ... @@
 context
-linha removida
+linha adicionada
```

Após o diff, adicione uma seção "Rationale:" de 2–4 frases explicando a mudança.

Constraints invioláveis do projeto:
- NÃO usar PIL ImageFont.truetype() — segfault 0xc0000005 no Windows
- NÃO usar ProcessPoolExecutor — BrokenProcessPool no Windows
- NÃO usar ThreadPoolExecutor com a mesma face FreeType — segfault
- NÃO usar matplotlib TextPath/TextToPath — falha com acentos
- USAR matplotlib.ft2font.FT2Font para rendering
- Execução serial obrigatória no typesetter
"""


class ClaudeSDKCoder:
    """Gera patches via Anthropic SDK usando o agente especialista correto."""

    coder_id = "claude_sdk"

    def propose_patch(self, proposal: dict, repo_root: Path) -> PatchProposal:
        proposal_id = str(proposal.get("proposal_id", "unknown"))

        # 1. Patch local determinístico (sem LLM)
        local = build_local_patch_from_hint(proposal, repo_root)
        if local is not None:
            return local

        # 2. Verificar disponibilidade
        if not is_claude_available():
            return PatchProposal(
                proposal_id=proposal_id,
                patch_unified_diff="",
                author="claude_sdk_coder",
                confidence=0.0,
                error=(
                    "ANTHROPIC_API_KEY não configurada ou pacote 'anthropic' não instalado. "
                    "Configure a variável de ambiente e reinicie o app."
                ),
            )

        try:
            return self._call_claude(proposal, repo_root, proposal_id)
        except Exception as exc:
            return PatchProposal(
                proposal_id=proposal_id,
                patch_unified_diff="",
                author="claude_sdk_coder",
                confidence=0.0,
                error=str(exc)[:500],
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call_claude(
        self, proposal: dict, repo_root: Path, proposal_id: str
    ) -> PatchProposal:
        target_file = str(proposal.get("target_file", ""))
        touched_domains = list(proposal.get("touched_domains") or [])

        # Carrega o agente especialista (system prompt)
        system = get_agent_prompt(target_file, touched_domains) or _DEFAULT_SYSTEM

        # Lê o arquivo alvo
        file_content = ""
        if target_file:
            path = repo_root / target_file
            if path.exists():
                file_content = read_file_slice(path, MAX_FILE_LINES)

        user_msg = _build_coder_prompt(proposal, file_content, target_file)

        client = make_client()
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text if response.content else ""
        diff = _extract_diff_from_response(raw)
        rationale = _extract_rationale(raw)
        confidence = _estimate_confidence(diff, raw)

        return PatchProposal(
            proposal_id=proposal_id,
            patch_unified_diff=diff,
            files_affected=[target_file] if (diff and target_file) else [],
            rationale=rationale,
            author="claude_sdk_coder",
            confidence=confidence,
            model_used=MODEL,
            generated_at_iso=datetime.datetime.now(datetime.UTC).isoformat(),
            error="" if diff else "Claude não produziu um diff válido na resposta.",
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_coder_prompt(
    proposal: dict, file_content: str, target_file: str
) -> str:
    title = str(proposal.get("title", ""))
    motivation = str(proposal.get("motivation", ""))
    change_kind = str(proposal.get("change_kind", "logic_fix"))
    issue_type = str(proposal.get("issue_type", ""))
    hint = proposal.get("local_patch_hint") or {}

    findings_text = "\n".join(
        f"- [{f.get('severity','?')}] {f.get('body','')}"
        for f in (proposal.get("review_findings") or [])[:5]
        if f.get("body")
    ) or "(sem findings detalhados)"

    file_section = ""
    if file_content and target_file:
        file_section = (
            f"\n\n## Conteúdo atual de `{target_file}` "
            f"(primeiras {MAX_FILE_LINES} linhas)\n\n"
            f"```python\n{file_content}\n```"
        )

    return f"""\
## Proposta de melhoria: {title}

**Issue type:** `{issue_type}`
**Change kind:** `{change_kind}`
**Arquivo alvo:** `{target_file}`

**Motivação:**
{motivation}

**Findings dos critics:**
{findings_text}

**Local patch hint:**
```json
{hint}
```{file_section}

## Sua tarefa

1. Gere um unified diff completo dentro de um bloco ```diff ... ```.
2. Após o bloco diff, escreva "Rationale:" seguido de 2–4 frases explicando a lógica.
3. Se precisar tocar mais de um arquivo, inclua todos os hunks no mesmo diff.
4. Não remova testes existentes.
5. Respeite todas as constraints do seu system prompt.
"""


def _extract_rationale(text: str) -> str:
    """Extrai a seção Rationale após o bloco diff."""
    # Remove blocos diff fenced
    after = re.sub(r"```diff[\s\S]*?```", "", text, flags=re.IGNORECASE).strip()
    # Procura "Rationale:" explícito
    m = re.search(r"Rationale:\s*([\s\S]+)", after, re.IGNORECASE)
    if m:
        return m.group(1).strip()[:600]
    return after[:400] if after else ""


def _estimate_confidence(diff: str, raw: str) -> float:
    if not diff:
        return 0.0
    score = 0.45
    if "---" in diff and "+++" in diff:
        score += 0.20
    if "@@" in diff:
        score += 0.15
    if re.search(r"^\+[^+]", diff, re.MULTILINE):  # tem linhas adicionadas
        score += 0.10
    if len(raw) > 150:
        score += 0.10
    return round(min(1.0, score), 3)


__all__ = ["ClaudeSDKCoder"]
