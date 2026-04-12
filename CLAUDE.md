# CLAUDE.md — Instruções para o Claude Code

## Sobre o Projeto
TraduzAi é um app desktop Tauri v2 (React + TypeScript frontend, Rust backend, Python sidecar) para tradução automática de mangá/manhwa/manhua EN→PT-BR.

## Stack
- **Frontend:** React 19 + TypeScript + Tailwind CSS + Zustand + Framer Motion
- **Backend:** Rust (Tauri v2 core)
- **Pipeline IA:** Python 3.12 (PaddleOCR, Claude Haiku API, OpenCV inpainting, matplotlib FT2Font typesetting)
- **Comunicação:** Tauri IPC (invoke) + Python sidecar via stdout JSON lines

## Arquitetura chave
- O frontend (src/) comunica com o Rust (src-tauri/) via `invoke()`
- O Rust spawna o Python sidecar (pipeline/) como processo filho
- O Python emite JSON lines no stdout que o Rust lê e repassa como eventos Tauri
- Créditos são gerenciados localmente com verificação server-side futura
- Free tier: 2 capítulos/semana (~40 páginas), reseta toda segunda-feira

## Padrões de código
- Frontend: componentes funcionais React, hooks, Zustand para state
- Rust: async/await com tokio, serde para serialização
- Python: type hints, docstrings, lazy imports para startup rápido
- Todas as mensagens de UI em português brasileiro
- Tema escuro (dark mode) é o padrão e único tema

## Caminhos importantes
- `src/lib/tauri.ts` — Todas as bindings Tauri (invoke calls)
- `src/lib/stores/appStore.ts` — State global (projeto, pipeline, créditos)
- `src-tauri/src/commands/` — Commands Rust expostos ao frontend
- `src-tauri/src/commands/settings.rs` — Restart do app (mata sidecars antes)
- `pipeline/main.py` — Entry point do pipeline Python
- `pipeline/translator/translate.py` — Prompt do Claude Haiku e chamada API
- `pipeline/typesetter/renderer.py` — Typesetting com FT2Font (medição + rendering)
- `pipeline/vision_stack/runtime.py` — OCR, classificação de balões, detecção de estilo
- `fonts/font-map.json` — Mapeamento tipo de texto → fonte

## Como testar
```bash
npm run tauri dev          # Dev mode completo
cd pipeline && python main.py config.json  # Pipeline isolado
```

## Decisões de design
- Processamento 100% local (exceto tradução via API e contexto via AniList)
- Nenhuma imagem é enviada a servidores — apenas texto extraído
- Formato project.json aberto e reimportável
- Sem funcionalidade de compartilhamento (segurança legal)
- 1 crédito = 1 página (modelo simples e transparente)

## Restrições técnicas importantes (typesetting)
- **NÃO usar PIL `ImageFont.truetype()`** para fontes — causa segfault 0xc0000005 no Windows com fontes de mangá (OTF/TTF)
- **NÃO usar ProcessPoolExecutor** — Windows `spawn` re-executa main.py nos workers → `BrokenProcessPool`
- **NÃO usar ThreadPoolExecutor** — FreeType (PIL) não é thread-safe para a mesma face → segfault
- **NÃO usar matplotlib TextPath** para rendering — falha com acentos (AttributeError) e acúmulo de chamadas causa segfault 0xc0000005
- **NÃO usar TextToPath singleton** — cache interno do matplotlib conflita com FT2Font criados separadamente → segfault
- **NÃO habilitar glow/gradient/shadow com TextPath** — efeitos multiplicam chamadas FreeType → crash em balões texturizados
- **Usar `matplotlib.ft2font.FT2Font`** para rendering — bitmap direto com anti-aliasing, suporta Unicode/acentos, sem PIL
- **Usar estimativa matemática para `getbbox()`** — `len(text) * size * 0.55` evita centenas de chamadas FreeType durante binary search
- Execução serial obrigatória no typesetting (sem paralelismo)
- Todos os balões (brancos e texturizados) recebem mesmo tratamento de estilo no typesetting (mesma fonte, force_upper)
- Inpainting é o único passo que diferencia branco de texturizado
- `restart_app()` deve matar processos sidecar Python antes de reiniciar (evita processos zombis)
