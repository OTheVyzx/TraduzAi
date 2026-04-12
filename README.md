# TraduzAi — Tradutor Desktop de Mangá/Manhwa/Manhua por IA

> Ferramenta local de tradução automática. Sem distribuição. Sem servidores. Seu mangá, sua tradução, seu arquivo.

## Requisitos

### Sistema
- **Windows 10+**, macOS 12+, ou Linux (Ubuntu 22.04+)
- **RAM:** 8GB mínimo, 16GB recomendado
- **Disco:** 3GB para o app + modelos de IA
- **Internet:** necessária apenas para tradução (API Claude) e busca de contexto

### GPU (opcional, mas recomendado)
- **NVIDIA:** GPU com 4GB+ VRAM e CUDA 11.8+ (RTX 20xx ou superior)
- **Apple Silicon:** M1/M2/M3 (Metal — suportado nativamente)
- **Sem GPU:** funciona em modo CPU (mais lento, ~30s/página vs ~5s/página)

### Desenvolvimento
- **Node.js** 20+
- **Rust** 1.77+ (via rustup)
- **Python** 3.10+
- **Tauri CLI** 2.x

## Setup de Desenvolvimento

### 1. Instalar dependências do sistema

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Tauri system dependencies (Linux)
sudo apt install libwebkit2gtk-4.1-dev build-essential curl wget file \
  libxdo-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev

# Tauri system dependencies (macOS)
xcode-select --install
```

### 2. Clonar e instalar

```bash
git clone https://github.com/seu-usuario/traduzai.git
cd traduzai

# Frontend dependencies
npm install

# Python pipeline dependencies
cd pipeline
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

### 3. Configurar API key (opcional)

```bash
# Criar arquivo de configuração
mkdir -p ~/.traduzai
echo "sk-ant-sua-chave-aqui" > ~/.traduzai/api_key
```

Sem API key, o app funciona em modo mock (tradução simulada para testar o pipeline).

### 4. Rodar em desenvolvimento

```bash
# Inicia o Tauri dev server (frontend + backend)
npm run tauri dev
```

### 5. Build para produção

```bash
# Primeiro, gerar o sidecar Python
cd pipeline
pyinstaller --onefile --name traduzai-pipeline main.py
cd ..

# Build do app Tauri
npm run tauri build
```

O instalador será gerado em `src-tauri/target/release/bundle/`.

## Estrutura do Projeto

```
traduzai/
├── src/                    # Frontend React (Tauri WebView)
│   ├── pages/              # Home, Setup, Processing, Preview, Settings
│   ├── components/         # UI components
│   ├── lib/                # Stores (Zustand), Tauri bindings
│   └── styles/             # Tailwind + tema dark
├── src-tauri/              # Backend Rust (Tauri Core)
│   └── src/commands/       # project, pipeline, credits
├── pipeline/               # Python Sidecar (IA)
│   ├── ocr/                # PaddleOCR
│   ├── translator/         # Claude Haiku API
│   ├── inpainter/          # OpenCV / LaMA
│   └── typesetter/         # Pillow text rendering
├── fonts/                  # Fontes manga livres
└── docs/                   # Documentação
```

## Pipeline de Tradução

```
Importar arquivo → Extrair imagens → OCR (PaddleOCR)
→ Contexto (AniList) → Tradução (Claude Haiku)
→ Inpainting (OpenCV/LaMA) → Typesetting (Pillow)
→ Exportar .zip com project.json
```

## Formato project.json

O TraduzAi usa um formato aberto que permite reimportação e edição:

```json
{
  "versao": "1.0",
  "app": "traduzai",
  "obra": "Solo Leveling",
  "capitulo": 42,
  "paginas": [
    {
      "numero": 1,
      "textos": [
        {
          "bbox": [120, 45, 380, 120],
          "original": "I alone can level up.",
          "traduzido": "Somente eu posso subir de nível.",
          "estilo": { "fonte": "AnimeAce", "tamanho": 16, "cor": "#FFFFFF" }
        }
      ]
    }
  ]
}
```

## Monetização

- **Free:** 2 capítulos/semana (~40 páginas)
- **Créditos:** 1 página = 1 crédito (a partir de R$0.05/página)
- **Planos:** Leitor R$15/mês, Tradutor R$30/mês, Scanlator R$60/mês
- **API própria:** use sua chave Claude para tradução ilimitada sem créditos

## Aviso Legal

TraduzAi é uma ferramenta de processamento local. Não distribui, armazena ou compartilha conteúdo protegido por direitos autorais. O usuário é responsável pelos arquivos que utiliza. Funciona de forma análoga a editores de imagem e tradutores de texto.

## Licença

MIT
