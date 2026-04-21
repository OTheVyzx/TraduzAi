# TraduzAi - Contexto Enxuto

> Ultima atualizacao: 2026-04-21 (v0.56.x)
> Objetivo: Controle profissional, GPU Nativa e Suporte a Manhwas.

## Snapshot do projeto
- App desktop Tauri v2 para traducao automatica de manga/manhwa/manhua EN -> PT-BR.
- Pipeline IA: Python 3.12 (Matplotlib FT2Font para renderização estável).

## Estado atual
- **Detecção de Hardware Real**: Implementada comunicação via sidecar para detecção real de GPU CUDA (RTX 4060 reconhecida).
- **Otimização para Long Strips**: Detector agora ajusta dinamicamente a resolução (até 3072px) para não perder balões pequenos em Manhwas verticais.
- **Estabilização Visual**: Refatoração do `EditorCanvas` para eliminar flickering (pisca-pisca) e remoção de avisos do linter via `AnimContainer`.
- **Modo Manual Profissional**: Ferramentas de OCR e Tradução pontual ("Photoshop-style") operando com GPU estável.

## Últimas mudanças
### 21/04/2026 - Hardware & Performance (v0.56.x)
- **Backend Rust**: Substituídos placeholders de hardware por chamadas reais de sistema via Python.
- **Detector**: Patch de resolução dinâmica no YOLO/CTD para imagens com proporção vertical extrema.
- **Build**: Corrigidos erros de sintaxe JSX e duplicidade de imports que impediam a compilação do Vite.
- **UI/UX**: Refinamento das animações e tempo de resposta na troca de páginas.
