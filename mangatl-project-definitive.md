# MangáTL — Tradutor Desktop de Mangá/Manhwa/Manhua por IA

> Ferramenta local de tradução automática. Sem distribuição. Sem servidores. Seu mangá, sua tradução, seu arquivo.

---

## 1. POSICIONAMENTO DO PRODUTO

### 1.1 O que é
MangáTL é um **aplicativo desktop** de tradução e edição automática de mangá/manhwa/manhua. Funciona como ferramenta local — igual a um editor de imagem com IA embutida.

### 1.2 O que NÃO é
- ❌ Não é plataforma de distribuição
- ❌ Não tem biblioteca pública
- ❌ Não tem sistema de compartilhamento
- ❌ Não gera links de acesso
- ❌ Não armazena conteúdo em servidor
- ❌ Não incentiva pirataria

### 1.3 Analogia legal
Funciona como Photoshop, GIMP, ou Google Translate: processa conteúdo que o usuário fornece e exporta o resultado. O usuário é 100% responsável pelos arquivos que utiliza.

### 1.4 Concorrência direta — Koharu
O Koharu (github.com/mayocream/koharu) é um tradutor de mangá em Rust/Tauri que faz pipeline similar (OCR → inpainting → tradução → typesetting). Está na versão 0.40, é open source (GPL v3), e roda localmente.

**Onde o MangáTL será superior:**
| Aspecto | Koharu | MangáTL |
|---------|--------|---------|
| Foco de tradução | JP→EN (primário) | EN→PT-BR (primário, com contexto) |
| Tradução contextual | LLM genérico | Claude com sinopse + personagens + glossário |
| Público-alvo | Técnico/dev | Leitor casual brasileiro |
| UX | Funcional mas técnica | Polida, pensada pro leitor de mangá |
| Formato de projeto | Interno | project.json aberto e editável |
| Glossário por obra | Não | Sim (consistência entre capítulos) |
| Comunidade | Global/EN | Focada no BR |

---

## 2. FLUXO PRINCIPAL DO APP

```
1. Usuário abre o app
2. Importa arquivos (.zip, .cbz ou pasta com imagens)
   OU importa projeto anterior (.zip com project.json)
3. Tela de configuração:
   - Nome da obra (autocomplete via AniList)
   - Idioma: EN→PT-BR (padrão)
   - Qualidade: Rápida / Normal / Alta
4. Botão "Traduzir" inicia o pipeline:
   ├── [1] Extração e validação
   ├── [2] OCR (detecção de texto nos balões)
   ├── [3] Busca de contexto (sinopse, personagens)
   ├── [4] Tradução contextualizada (Claude API)
   ├── [5] Inpainting (remoção do texto original)
   └── [6] Typesetting (texto traduzido aplicado)
5. Preview em tempo real (página por página)
6. Usuário pode editar textos manualmente (futuro V3)
7. Exporta resultado em .zip
8. FIM — nenhum compartilhamento
```

---

## 3. FORMATO DE PROJETO — project.json

### 3.1 Estrutura do ZIP exportado
```
MinhaTradução.zip
├── images/          # Imagens originais (limpas, sem texto)
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── translated/      # Imagens finais traduzidas
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
└── project.json     # Metadados completos do projeto
```

### 3.2 Estrutura do project.json
```json
{
  "versao": "1.0",
  "app": "mangatl",
  "obra": "Solo Leveling",
  "capitulo": 42,
  "idioma_origem": "en",
  "idioma_destino": "pt-BR",
  "contexto": {
    "sinopse": "Sung Jin-Woo é o caçador mais fraco...",
    "genero": ["action", "fantasy"],
    "personagens": ["Sung Jin-Woo", "Cha Hae-In"],
    "glossario": {
      "Hunter": "Caçador",
      "Gate": "Portal",
      "S-Rank": "Rank-S"
    }
  },
  "paginas": [
    {
      "numero": 1,
      "arquivo_original": "images/001.jpg",
      "arquivo_traduzido": "translated/001.jpg",
      "textos": [
        {
          "id": "t1",
          "bbox": [120, 45, 380, 120],
          "tipo": "fala",
          "original": "I alone can level up.",
          "traduzido": "Somente eu posso subir de nível.",
          "confianca_ocr": 0.97,
          "estilo": {
            "fonte": "AnimeAce",
            "tamanho": 16,
            "cor": "#FFFFFF",
            "contorno": "#000000",
            "contorno_px": 2,
            "bold": false,
            "italico": false,
            "rotacao": 0,
            "alinhamento": "center"
          }
        },
        {
          "id": "t2",
          "bbox": [400, 200, 580, 250],
          "tipo": "sfx",
          "original": "BOOM",
          "traduzido": "BOOM",
          "confianca_ocr": 0.85,
          "estilo": {
            "fonte": "MangaBold",
            "tamanho": 28,
            "cor": "#FF3333",
            "contorno": "#000000",
            "contorno_px": 3,
            "bold": true,
            "italico": false,
            "rotacao": -15,
            "alinhamento": "center"
          }
        }
      ]
    }
  ],
  "estatisticas": {
    "total_paginas": 24,
    "total_textos": 156,
    "tempo_processamento_seg": 180,
    "creditos_usados": 24,
    "data_criacao": "2026-03-30T14:30:00Z"
  }
}
```

### 3.3 Reimportação de projetos
- Se o ZIP contém `project.json` → abre como projeto editável
- Se não contém → trata como nova obra
- Permite: continuar tradução, editar textos, reprocessar com melhorias, alterar glossário

---

## 4. STACK TÉCNICA — DECISÕES DE CUSTO-BENEFÍCIO

### 4.1 Framework: Tauri v2 (React + Rust)

**Por que Tauri e não Electron:**
- Bundle de ~3-10MB (vs 150MB+ do Electron)
- Usa WebView nativo do OS (não embarca Chromium)
- Backend em Rust = seguro, rápido, leve
- Suporte a Windows, macOS, Linux
- Frontend em React (já domina)
- Auditoria de segurança independente (Radically Open Security)

**Por que não Rust puro como Koharu:**
- Koharu é escrito 100% em Rust — exige expertise em Rust
- Tauri permite usar React/TypeScript no frontend (desenvolvimento mais rápido)
- O pipeline pesado (OCR, inpainting) será em Python (mais bibliotecas de ML disponíveis)
- Tauri serve como ponte: React (UI) ↔ Rust (sistema) ↔ Python (IA)

### 4.2 Pipeline de processamento

| Etapa | Ferramenta | Onde roda | Custo |
|-------|-----------|-----------|-------|
| OCR | PaddleOCR | Local (CPU/GPU) | GRÁTIS |
| Detecção de balões | PP-DocLayoutV3 ou YOLO | Local (CPU/GPU) | GRÁTIS |
| Busca de contexto | AniList GraphQL API | Internet | GRÁTIS |
| Tradução | Claude Haiku API | Internet (API) | ~$0.002/pg |
| Inpainting | LaMA / AOT-GAN | Local (GPU) | GRÁTIS |
| Typesetting | Pillow + fontes locais | Local (CPU) | GRÁTIS |

**Custo real por página: ~R$0.012** (apenas a chamada de API para tradução)
**Custo por capítulo de 20 páginas: ~R$0.24**

Isso é drasticamente mais barato que o modelo web porque OCR, inpainting e typesetting rodam 100% na máquina do usuário.

### 4.3 Processamento local — Requisitos

**Mínimo:**
- CPU: qualquer x64 moderno (i5 8ª gen ou equivalente)
- RAM: 8GB
- GPU: não necessária (modo CPU, mais lento)
- Disco: 2GB para modelos de IA
- Internet: necessária apenas para tradução (API) e contexto

**Recomendado:**
- CPU: i7/Ryzen 7 ou superior
- RAM: 16GB
- GPU: NVIDIA com 4GB+ VRAM (CUDA) ou Apple Silicon (Metal)
- Disco: SSD com 5GB livre
- Internet: banda larga

**Modo CPU vs GPU:**
- COM GPU (NVIDIA CUDA): ~5 seg/página (total pipeline)
- SEM GPU (CPU only): ~25-40 seg/página
- Modelos são baixados no primeiro uso (~1.5GB)

### 4.4 Comunicação Tauri ↔ Python

```
[React Frontend (Tauri WebView)]
        │ invoke()
        ▼
[Rust Backend (Tauri Core)]
        │ Command::new("python")
        │ ou sidecar bundled
        ▼
[Python Sidecar]
  ├── paddleocr
  ├── claude API client
  ├── lama-cleaner
  └── pillow typesetter
        │ stdout JSON
        ▼
[Rust Backend] → [React Frontend]
```

O Python será empacotado como "sidecar" do Tauri — um executável standalone (via PyInstaller/Nuitka) que vem junto com o app. O usuário não precisa instalar Python.

---

## 5. ARQUITETURA DO SISTEMA

```
mangatl/
├── src/                        # Frontend React (Tauri WebView)
│   ├── App.tsx                 # Raiz
│   ├── pages/
│   │   ├── Home.tsx            # Tela inicial, importar/criar projeto
│   │   ├── Setup.tsx           # Config: nome da obra, idioma, qualidade
│   │   ├── Processing.tsx      # Pipeline com progresso real
│   │   ├── Preview.tsx         # Visualização antes/depois
│   │   ├── Export.tsx          # Exportação de resultado
│   │   ├── Settings.tsx        # Configurações do app
│   │   └── Credits.tsx         # Compra de créditos
│   ├── components/
│   │   ├── ui/                 # Design system (botões, inputs, cards)
│   │   ├── ImageViewer.tsx     # Comparação original vs traduzido
│   │   ├── ProgressPipeline.tsx # Barra de progresso por etapa
│   │   ├── TextEditor.tsx      # Editor de textos (futuro)
│   │   ├── GlossaryPanel.tsx   # Glossário da obra
│   │   └── ProjectCard.tsx     # Card de projeto recente
│   ├── lib/
│   │   ├── tauri.ts            # Bindings Tauri invoke
│   │   ├── stores/             # Zustand state management
│   │   └── api.ts              # Claude API + Woovi/Stripe
│   └── styles/                 # Tailwind + tema dark manga
│
├── src-tauri/                  # Backend Rust (Tauri Core)
│   ├── src/
│   │   ├── main.rs             # Entry point
│   │   ├── commands/           # Commands expostos ao frontend
│   │   │   ├── project.rs      # Criar, abrir, salvar projeto
│   │   │   ├── pipeline.rs     # Iniciar/cancelar pipeline
│   │   │   ├── export.rs       # Exportar ZIP
│   │   │   ├── credits.rs      # Gerenciar créditos
│   │   │   └── settings.rs     # Configurações
│   │   ├── pipeline/
│   │   │   └── sidecar.rs      # Comunicação com Python sidecar
│   │   └── utils/
│   │       ├── zip.rs          # Manipulação de ZIP/CBZ
│   │       └── fs.rs           # File system helpers
│   ├── Cargo.toml
│   └── tauri.conf.json
│
├── pipeline/                   # Python Sidecar (IA)
│   ├── main.py                 # Entry point do sidecar
│   ├── ocr/
│   │   ├── detector.py         # PaddleOCR + detecção de balões
│   │   └── analyzer.py         # Análise de estilo (cor, fonte, efeitos)
│   ├── translator/
│   │   ├── context.py          # Busca contexto (AniList API)
│   │   ├── translate.py        # Claude Haiku API
│   │   └── glossary.py         # Gerenciamento de glossário
│   ├── inpainter/
│   │   ├── lama.py             # LaMA inpainting
│   │   └── mask.py             # Geração de máscaras
│   ├── typesetter/
│   │   ├── renderer.py         # Renderização de texto (Pillow)
│   │   ├── fonts.py            # Font matching e seleção
│   │   └── layout.py           # Cálculo de layout em balões
│   ├── models/                 # Modelos de IA (baixados no 1º uso)
│   │   └── .gitkeep
│   ├── requirements.txt
│   └── build.py                # Script para criar sidecar (PyInstaller)
│
├── fonts/                      # Fontes de mangá livres/licenciadas
│   ├── speech/
│   │   ├── AnimeAce.ttf
│   │   ├── CCWildWords.ttf
│   │   └── MangaTemple.ttf
│   ├── narration/
│   │   ├── CCAstroCity.ttf
│   │   └── DigitalStrip.ttf
│   ├── sfx/
│   │   ├── BadaBoom.ttf
│   │   └── ComicBold.ttf
│   └── font-map.json           # Mapeamento tipo_texto → fonte padrão
│
├── assets/                     # Ícones, imagens do app
├── docs/                       # Documentação
├── package.json                # Dependencies do frontend
└── README.md
```

---

## 6. PIPELINE DE TRADUÇÃO — DETALHAMENTO

### 6.1 Etapa 1: Extração e Validação
- Aceita: .zip, .cbz, pasta com JPG/PNG/WEBP
- Descompacta e ordena páginas numericamente
- Detecta se é projeto existente (project.json presente)
- Gera thumbnails para navegação rápida

### 6.2 Etapa 2: OCR + Detecção de Balões
**Ferramenta: PaddleOCR (local, gratuito)**
- Detecta regiões de texto + bounding boxes
- Classifica tipo: fala, narração, SFX, pensamento
- Extrai: texto, posição, direção (horizontal/vertical)
- Confiança do OCR salva no project.json

**Análise de estilo visual:**
- Cor do texto: sampling de pixels dentro da bbox
- Tamanho: altura da bbox convertida em pt
- Contorno: detecção de borda (gradiente de cor)
- Bold/itálico: análise de espessura dos strokes
- Rotação: ângulo da bbox (se SFX)

### 6.3 Etapa 3: Contexto da Obra
**Busca automática via AniList GraphQL API (grátis):**
```graphql
query {
  Media(search: "Solo Leveling", type: MANGA) {
    title { english romaji }
    description
    genres
    characters(sort: ROLE) {
      nodes { name { full } }
    }
  }
}
```
- Extrai: sinopse, gênero, lista de personagens
- Gera glossário inicial automático
- Usuário pode editar/complementar glossário

### 6.4 Etapa 4: Tradução (Claude Haiku API)
**Único componente que usa internet/API**

```
System prompt:
"Você é um tradutor profissional de mangá especializado em EN→PT-BR.

OBRA: {titulo}
SINOPSE: {sinopse}
GÊNERO: {generos}
PERSONAGENS: {personagens}
GLOSSÁRIO: {glossario}

REGRAS:
- Traduza para português brasileiro natural e fluido
- Mantenha honoríficos quando presentes (san, kun, chan, sensei)
- SFX: traduza quando fizer sentido, mantenha original se universal
- Use o glossário para manter consistência de termos
- Cada tradução deve caber em ~{max_chars} caracteres
- Mantenha o tom adequado ao gênero ({genero})
- Para falas, use linguagem natural de diálogo
- Para narração, use tom mais formal
- Responda APENAS com o JSON de traduções"

User message:
"Traduza os seguintes textos da página {num}:
{lista de textos com bbox e tipo}

Contexto das páginas anteriores:
{textos traduzidos das últimas 3 páginas}"
```

**Custo estimado por página:**
- Input: ~300 tokens (prompt + contexto + textos)
- Output: ~150 tokens (traduções)
- Claude Haiku: $0.25/MTok input, $1.25/MTok output
- **Custo real: ~$0.00026/página = R$0.0016/página**
- **20 páginas = R$0.032 de custo real**

Isso é tão barato que dá pra oferecer um trial generoso.

### 6.5 Etapa 5: Inpainting (Remoção de Texto)
**Ferramenta: LaMA (local, GPU)**
- Gera máscara a partir das bounding boxes do OCR
- Dilata máscara em 3px (cobrir bordas)
- LaMA preenche a região removida reconstruindo o fundo
- Resultado salvo em /images/ (imagem limpa)

### 6.6 Etapa 6: Typesetting (Renderização de Texto)
**Ferramenta: Pillow/PIL (local, CPU)**
- Seleciona fonte baseada no tipo de texto + font-map.json
- Calcula tamanho do texto para caber na bbox
- Auto word-wrap respeitando largura do balão
- Aplica: cor, contorno (stroke), rotação, alinhamento
- Centraliza texto verticalmente no balão
- Anti-aliasing para qualidade
- Resultado salvo em /translated/

---

## 7. MONETIZAÇÃO — MODELO FREEMIUM

### 7.1 Custo real operacional
O único custo variável é a API Claude Haiku para tradução:
- R$0.0016 por página (custo para nós)
- R$0.032 por capítulo de 20 páginas

Todo o resto (OCR, inpainting, typesetting) é local e gratuito.

### 7.2 Modelo de créditos: 1 página = 1 crédito

**Free tier (sem pagar nada):**
- 10 páginas/dia grátis (resetam a cada 24h)
- Suficiente para experimentar, traduzir ~1 capítulo a cada 2 dias
- Todas as funcionalidades disponíveis
- Sem marca d'água

**Pacotes de créditos (one-time purchase):**

| Pacote | Preço | Créditos | R$/página | Capítulos (~20pg) |
|--------|-------|----------|-----------|-------------------|
| Starter | R$5 | 100 | R$0.05 | 5 |
| Pack | R$15 | 400 | R$0.038 | 20 |
| Mega | R$35 | 1.000 | R$0.035 | 50 |
| Ultra | R$60 | 2.000 | R$0.030 | 100 |

**Margem:** mesmo no pacote mais barato (R$0.05/pg), o custo real é R$0.0016/pg. Margem de 96%.

### 7.3 Planos mensais (assinatura)

| Plano | Preço/mês | Créditos/mês | Extra |
|-------|-----------|--------------|-------|
| Free | R$0 | 10/dia | Funcionalidades básicas |
| Leitor (R$15) | R$15 | 800/mês | Fila prioritária no pipeline, qualidade alta |
| Tradutor (R$30) | R$30 | 2.000/mês | + Glossário ilimitado, modo batch, suporte |
| Scanlator (R$60) | R$60 | 5.000/mês | + API acesso, exportação PSD, modo bulk |

### 7.4 Pagamentos
- **PIX:** Woovi (API simples, ~R$0.85/transação ou 1%)
- **Cartão:** Stripe (2.99% + R$0.39)
- Créditos armazenados localmente com verificação server-side (JWT token com saldo)

### 7.5 Infraestrutura necessária (mínima)

Como 95% roda local, o servidor é mínimo:

| Serviço | Função | Custo estimado |
|---------|--------|---------------|
| API server (fly.io/Railway) | Auth, créditos, validação | R$0-50/mês |
| PostgreSQL (Supabase free) | Usuários, créditos, transações | R$0 |
| Claude API (pay-per-use) | Tradução | ~R$0.50 por 1000 páginas |
| Woovi + Stripe | Pagamentos | Taxa por transação |
| **Total** | | **R$0-100/mês para começar** |

Comparado com o modelo web que custaria R$500-1.680/mês, isso é 95% mais barato.

---

## 8. TELAS DO APP (UI/UX)

### 8.1 Tema visual
- **Dark mode** como padrão (leitores de mangá preferem escuro)
- Paleta: fundo #0F0F14, cards #1A1A24, acento roxo #7C5CFF, texto #E8E8F0
- Fontes: Geist Sans (UI) + JetBrains Mono (monospace/código)
- Cantos arredondados, transições suaves, micro-animações

### 8.2 Tela: Home
```
┌──────────────────────────────────────────┐
│  MangáTL          [Config] [Créditos: 47]│
│                                          │
│  ┌──────────────┐  ┌──────────────┐     │
│  │ + Nova       │  │ ↗ Abrir      │     │
│  │   Tradução   │  │   Projeto    │     │
│  └──────────────┘  └──────────────┘     │
│                                          │
│  Projetos Recentes                       │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐           │
│  │Solo│ │One │ │Blue│ │JJK │           │
│  │Lev.│ │Pie.│ │Lock│ │    │           │
│  │24pg│ │18pg│ │30pg│ │22pg│           │
│  │ ✓  │ │ ✓  │ │ 70%│ │ ✓  │           │
│  └────┘ └────┘ └────┘ └────┘           │
│                                          │
│  ──────── Status do Sistema ──────────  │
│  GPU: NVIDIA RTX 4060 (CUDA ✓)         │
│  Modelos: OK (1.4GB)                    │
│  API: Conectada                          │
└──────────────────────────────────────────┘
```

### 8.3 Tela: Setup (Configuração da tradução)
```
┌──────────────────────────────────────────┐
│  ← Nova Tradução                         │
│                                          │
│  Nome da Obra: [Solo Leveling      🔍]  │
│  (autocomplete: Solo Leveling - manhwa)  │
│                                          │
│  ┌─ Informações detectadas ────────────┐│
│  │ Sinopse: Sung Jin-Woo é o caçador...││
│  │ Gênero: Action, Fantasy              ││
│  │ Personagens: Sung Jin-Woo, Cha Hae..││
│  └──────────────────────────────────────┘│
│                                          │
│  Idioma: [EN → PT-BR ▼]                │
│  Qualidade: ○ Rápida  ● Normal  ○ Alta  │
│  Capítulo: [42]                          │
│                                          │
│  Glossário (editável):                   │
│  Hunter = Caçador                        │
│  Gate = Portal                           │
│  + Adicionar termo                       │
│                                          │
│  Arquivo: [Selecionar .zip / .cbz / 📁] │
│                                          │
│  [        🚀 TRADUZIR        ]          │
│                                          │
│  Estimativa: ~24 páginas, ~2 min        │
│  Créditos necessários: 24                │
└──────────────────────────────────────────┘
```

### 8.4 Tela: Processamento (Pipeline)
```
┌──────────────────────────────────────────┐
│  Solo Leveling - Cap. 42                 │
│                                          │
│  ████████████████████░░░░░ 78%          │
│  Tempo restante: ~28 segundos            │
│                                          │
│  ✓ Extração (0.3s)                      │
│  ✓ OCR - 156 textos detectados (12s)    │
│  ✓ Contexto carregado (0.8s)            │
│  ✓ Tradução concluída (8s)              │
│  ● Inpainting... página 19/24 (15s)     │
│  ○ Typesetting                           │
│                                          │
│  ┌─ Preview ao vivo ──────────────────┐ │
│  │  [Página 18 - Antes]  [Depois]     │ │
│  │  ┌────────┐  ┌────────┐           │ │
│  │  │ imagem │  │ imagem │           │ │
│  │  │ orig.  │  │ trad.  │           │ │
│  │  └────────┘  └────────┘           │ │
│  └────────────────────────────────────┘ │
│                                          │
│  [Cancelar]                              │
└──────────────────────────────────────────┘
```

### 8.5 Tela: Preview + Exportação
```
┌──────────────────────────────────────────┐
│  Solo Leveling - Cap. 42    [Exportar ↓] │
│                                          │
│  ┌─────────────────────────────────────┐│
│  │                                      ││
│  │    Visualizador de página           ││
│  │    (swipe/setas para navegar)       ││
│  │                                      ││
│  │    Toggle: [Original] [Traduzido]   ││
│  │                                      ││
│  └─────────────────────────────────────┘│
│                                          │
│  Pg: ← [18/24] →                        │
│                                          │
│  Textos desta página:                    │
│  "I alone can level up."                 │
│  → "Somente eu posso subir de nível."   │
│  [✏️ Editar]                             │
│                                          │
│  ┌─ Exportar ───────────────────────┐   │
│  │ ○ ZIP completo (images + transl.) │   │
│  │ ● Somente traduzidos (.jpg)       │   │
│  │ ○ CBZ                             │   │
│  │ [       Download .zip       ]     │   │
│  └───────────────────────────────────┘   │
└──────────────────────────────────────────┘
```

---

## 9. SEGURANÇA LEGAL — RESUMO

### 9.1 Por que estamos seguros
1. **Ferramenta, não plataforma** — não distribuímos conteúdo
2. **Processamento local** — arquivos nunca saem da máquina do usuário
3. **Nenhum armazenamento** — nosso servidor só gerencia créditos/auth
4. **Nenhum compartilhamento** — não existe funcionalidade para compartilhar
5. **Nenhuma biblioteca** — não indexamos nem catalogamos obras
6. **Terms of Service** — usuário declara ter direito de uso pessoal

### 9.2 O que trafega pela internet
- Nome da obra → AniList (para contexto) — dados públicos
- Texto extraído → Claude API (para tradução) — apenas texto, não imagens
- Auth + créditos → nosso server — dados de conta
- **Nenhuma imagem é enviada a servidor algum**

### 9.3 Modelo análogo
- Google Translate: traduz texto protegido → ferramenta
- Photoshop: edita imagens protegidas → ferramenta
- VLC: reproduz mídia protegida → ferramenta
- **MangáTL: traduz mangá → ferramenta**

---

## 10. MODELO DE DADOS (servidor mínimo)

```sql
-- Usuários
users
  id              UUID PK
  email           VARCHAR UNIQUE
  name            VARCHAR
  plan            ENUM('free','leitor','tradutor','scanlator') DEFAULT 'free'
  plan_expires_at TIMESTAMP
  credits_balance INTEGER DEFAULT 0
  api_key         VARCHAR UNIQUE
  created_at      TIMESTAMP

-- Transações
transactions
  id            UUID PK
  user_id       UUID FK → users
  type          ENUM('purchase','subscription','usage','refund','daily_free')
  amount        INTEGER (+ = crédito, - = uso)
  description   VARCHAR
  provider      ENUM('woovi','stripe','system')
  provider_id   VARCHAR
  created_at    TIMESTAMP

-- Uso diário (controle do free tier)
daily_usage
  id            UUID PK
  user_id       UUID FK → users
  date          DATE
  pages_used    INTEGER DEFAULT 0
  
-- Glossários compartilhados (futuro — por obra)
shared_glossaries
  id            UUID PK
  obra_title    VARCHAR
  anilist_id    INTEGER
  terms         JSONB
  contributor   UUID FK → users
  votes         INTEGER DEFAULT 0
  created_at    TIMESTAMP
```

---

## 11. ROADMAP DE DESENVOLVIMENTO

### MVP (4-6 semanas)
- [ ] Setup Tauri v2 + React + TypeScript
- [ ] Python sidecar com PaddleOCR básico
- [ ] Tradução via Claude Haiku (prompt simples)
- [ ] Inpainting com LaMA (local)
- [ ] Typesetting básico (Pillow, fonte única)
- [ ] Exportação .zip com project.json
- [ ] UI funcional (home, setup, progress, export)
- [ ] Free tier (10 pg/dia) sem necessidade de pagamento

### V2 — Beta (4-6 semanas)
- [ ] Busca de contexto (AniList API)
- [ ] Glossário por obra
- [ ] Font matching (múltiplas fontes por tipo de texto)
- [ ] Detecção de estilo (cor, contorno, rotação)
- [ ] Preview antes/depois em tempo real
- [ ] Sistema de créditos + Woovi PIX + Stripe
- [ ] Reimportação de projetos
- [ ] Instalador Windows (.msi)

### V3 — Release (4 semanas)
- [ ] Editor manual de textos (clicar no balão → editar)
- [ ] Modo batch (traduzir vários capítulos de uma vez)
- [ ] Glossários compartilhados (comunidade)
- [ ] Tradução contextual avançada (últimas 5 páginas)
- [ ] Detecção de SFX e tratamento especial
- [ ] Instalador macOS (.dmg) + Linux (.AppImage)
- [ ] UX polida, onboarding, tutorial

### V4 — Growth
- [ ] Mais idiomas (JP→PT, KR→PT, ES→PT)
- [ ] Plugin system (fontes customizadas, modelos de IA)
- [ ] Exportação PSD (layers editáveis)
- [ ] Comunidade: rankings, glossários votados
- [ ] Mobile companion (visualização)
- [ ] API para automação

---

## 12. DIFERENCIAIS COMPETITIVOS vs KOHARU

| Feature | Koharu | MangáTL |
|---------|--------|---------|
| Tradução contextualizada | Genérica | Com sinopse + personagens + glossário |
| Foco idiomático | JP→EN primário | EN→PT-BR especializado |
| project.json editável | ❌ | ✅ Formato aberto |
| Reimportação de projetos | ❌ | ✅ Continuar de onde parou |
| Glossário por obra | ❌ | ✅ Compartilhável |
| UX para leitor casual | Técnica | Intuitiva, dark theme manga |
| Onboarding | Manual | Tutorial guiado |
| Free tier | Totalmente grátis | 10 pg/dia grátis |
| Planos | ❌ (open source) | Freemium com créditos |
| Comunidade BR | ❌ | ✅ Focada |

---

## 13. PERGUNTAS RESOLVIDAS

| Pergunta | Decisão | Justificativa |
|----------|---------|---------------|
| Desktop ou Web? | Desktop (Tauri v2) | Processamento local, custo zero de infra, segurança legal |
| GPU ou API? | Local (GPU do usuário) + API só para tradução | 95% mais barato, 100% dos modelos pesados rodam local |
| Qual OCR? | PaddleOCR | Gratuito, roda local, bom para inglês |
| Qual LLM? | Claude Haiku | Melhor custo-benefício para tradução PT-BR |
| Qual inpainting? | LaMA | Gratuito, local, estado da arte |
| Pagamento PIX? | Woovi | API simples, ~R$0.85/transação |
| Pagamento cartão? | Stripe | Padrão global, aceita internacional |
| Modelo de cobrança? | 1 crédito = 1 página | Simples, transparente, justo |
| Free tier? | 10 páginas/dia grátis | Suficiente para experimentar, converte para pago |
| Compartilhamento? | NENHUM | Segurança legal absoluta |

---

*Documento definitivo v2.0 — Base para desenvolvimento no Claude Code*
*Atualizado em: 30/03/2026*
