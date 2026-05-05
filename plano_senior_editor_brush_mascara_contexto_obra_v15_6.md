# Plano Sênior v15.6 — Editor Brush/Máscara + Typesetting Safe Area + Máscara Regional + Contexto da Obra

> Branch: **`feat/editor-brush-mask-typesetting`**  
> Base: **`chore/repo-cleanup-disable-lab`**  
> Projeto: **TraduzAi**  
> Objetivo: corrigir texto cortado, brush/máscara/borracha, UI do editor, máscara regional real e fazer o **Contexto da Obra** funcionar de ponta a ponta com baixo risco de regressão.

---

## 1. Veredito sênior

O plano v15.5 já está tecnicamente forte e serve como base para implementação. Esta versão v15.6 consolida os últimos ajustes:

- solução mais segura para os warnings Rust de `internet_context.rs`;
- `PageActionResult` com enum/tipos fortes;
- `changed_assets` padronizado;
- auto-fit de texto **não destrutivo**;
- `clearMaskLayer` bem definido;
- Translate regional pragmático por bbox na primeira versão;
- separação entre bitmap layer geral e bitmap layer mutável;
- e, principalmente, uma fase dedicada para o **Contexto da Obra** realmente funcionar no fluxo de tradução.

O plano agora não deve tratar “contexto da obra” como recurso decorativo. Ele precisa alimentar a tradução, preservar nomes/personagens/termos, manter consistência entre páginas/capítulos e aparecer na UI de forma editável e verificável.

---

## 2. Regras absolutas

1. Auditar o código real antes de alterar.
2. Não reescrever `applyBitmapStroke` sem diagnóstico.
3. Não salvar `?v=` dentro de `image_layers.path` ou `project.json`.
4. Qualquer cache-bust deve ser runtime-only ou campo separado de version/cacheKey.
5. Não remover UI que não existe.
6. Não remover funcionalidade sem substituição equivalente.
7. Não quebrar Tauri, build, project.json, reload, export ou preview.
8. Cada fase deve ser um commit isolado.
9. Se uma fase quebrar, deve ser possível reverter sem perder as anteriores.
10. Testes devem acompanhar cada fase quando possível.
11. O auto-fit de texto deve ser não destrutivo por padrão.
12. A máscara regional deve preservar conteúdo fora da região.
13. O contexto da obra deve ser persistente, editável e usado pela tradução de forma explícita.
14. O contexto da obra não pode inventar nomes, relações ou termos sem fonte do usuário/projeto.

---

## 3. Correções obrigatórias antes da implementação

### 3.1. Não salvar `?v=` dentro de `image_layers.path`

Não fazer:

```ts
image_layers[layerKey].path = `${absolutePath}?v=${Date.now()}`
```

O correto é separar o **path real persistido** do **cache-bust visual runtime**.

Recomendado:

```ts
image_layers[layerKey].path = absolutePath
bumpBitmapLayerVersion(layerKey)
```

Uso visual:

```ts
const src = convertFileSrc(path) + `?v=${bitmapLayerVersions[layerKey] ?? 0}`
```

Regra:

> O `project.json` nunca deve receber paths com `?v=`.

---

### 3.2. Confirmar o bug do brush antes de assumir cache

Antes de alterar:

1. Desenhar com brush.
2. Verificar se o PNG muda no disco após `mouseup`.
3. Verificar se `currentPage.image_layers.brush/mask` recebe o path correto.
4. Verificar se o Konva continua renderizando imagem antiga.
5. Só então aplicar cache-bust por `version/cacheKey`.

Critério:

- Se o PNG não muda no disco, o problema não é cache.
- Se o PNG muda, mas a imagem visual não atualiza, o problema provavelmente é cache.

---

### 3.3. Clipping: detectar não basta, precisa corrigir

A integração do `text_fit_guard` não deve apenas emitir `text_clipped`.

Fluxo obrigatório:

1. Calcular `safe_text_box`.
2. Renderizar texto em camada/máscara temporária.
3. Medir `ink_bbox` real via alpha mask.
4. Se `ink_bbox` extrapolar `safe_text_box`:
   - reduzir fonte;
   - refazer wrap;
   - tentar novamente.
5. Repetir até caber ou atingir fonte mínima.
6. Se ainda não couber:
   - renderizar melhor esforço;
   - emitir `text_clipped` com `severity=critical`.

Regra:

> O QA deve reportar o problema, mas o renderer precisa primeiro tentar resolvê-lo automaticamente.

---

### 3.4. Não medir `ink_bbox` por glyph no primeiro patch

Evitar medir `ink_bbox` via glyph/FT2Font no primeiro patch.

Melhor abordagem:

> Renderizar o texto em uma máscara alpha temporária e medir o bbox real dos pixels desenhados.

Exemplo:

```python
text_alpha_mask = Image.new("L", (page_width, page_height), 0)
measured_ink_bbox = text_alpha_mask.getbbox()
```

---

### 3.5. Render de texto: uma única primitiva

A medição por `text_alpha_mask` deve reutilizar a mesma função/primitiva de renderização usada no render final.

Não criar uma segunda lógica paralela de render de texto.

O alpha mask deve receber exatamente os mesmos parâmetros do render final:

- fonte;
- tamanho;
- wrap;
- alinhamento;
- stroke;
- sombra, se ela entra no bbox visual;
- posição;
- rotação/escala, se suportado.

---

### 3.6. Tratar `getbbox() == None`

Se `measured_ink_bbox is None`:

- não chamar `validate_rendered_text_fit` para essa layer;
- emitir QA warning `empty_text_render`;
- continuar renderização sem quebrar a página.

Exemplo:

```python
measured_ink_bbox = text_alpha_mask.getbbox()

if measured_ink_bbox is None:
    qa_flags.append({
        "type": "empty_text_render",
        "severity": "warning",
        "region_id": layer.get("id"),
        "page": page_index + 1,
    })
else:
    fit_result = validate_rendered_text_fit(...)
```

---

### 3.7. `build_safe_area` precisa ter fallback

Não substituir o cálculo antigo de `safe_text_box` de forma cega.

```python
fallback_safe_text_box = current_manual_safe_text_box

try:
    safe_area_result = build_safe_area(...)
    safe_text_box = safe_area_result.get("safe_bbox") or fallback_safe_text_box
except Exception:
    safe_text_box = fallback_safe_text_box
    # emitir QA warning: safe_area_fallback_used
```

---

### 3.8. Auto-fit não destrutivo

O auto-fit do renderer deve ser não destrutivo por padrão.

Ele pode usar `font_size_final` apenas no render atual.

Não deve alterar permanentemente o estilo salvo da layer, a menos que exista ação explícita de “aplicar auto-fit”.

Regra:

> Preview/render pode adaptar fonte para caber, mas não deve sobrescrever `fontSize` salvo no projeto sem ação explícita.

---

### 3.9. Máscara regional deve usar pixels reais, não só bbox

A `bbox` da máscara serve para performance, mas a seleção real deve ser a máscara.

```txt
bbox = área ampla de processamento
pixels da máscara = seleção real
```

Por ação:

| Ação | Regra |
|---|---|
| Inpaint | usar pixels reais da máscara como mask de inpaint |
| OCR | usar bbox para filtrar candidatos, refinamento por pixels quando necessário |
| Translate | primeira versão pode usar interseção `layer.bbox` x `mask_bbox` |
| Detect | rodar por último; usar bbox expandida e converter coordenadas de volta |

---

### 3.10. Threshold da máscara

Não considerar qualquer pixel maior que zero como máscara ativa.

Usar threshold documentado:

```txt
pixel ativo = luminância/alpha >= 8
```

Centralizar em constante:

```rust
const MASK_ACTIVE_THRESHOLD: u8 = 8;
```

```python
MASK_ACTIVE_THRESHOLD = 8
```

---

### 3.11. Convenção única de bbox

Todas as bboxes regionais devem usar convenção half-open:

```txt
[x1, y1, x2, y2)
```

Regras:

- `x1/y1` são inclusivos;
- `x2/y2` são exclusivos;
- `width = x2 - x1`;
- `height = y2 - y1`.

Essa convenção deve valer em Rust, Python, frontend, testes, logs e QA report.

---

### 3.12. Contrato explícito de merge regional

Quando rodar ação regional, o sistema não pode apagar a página inteira.

Contrato:

- Fora da máscara: preservar tudo.
- Dentro da máscara: atualizar apenas o necessário.
- Layers manuais não devem ser apagadas sem regra explícita.
- Ações regionais nunca devem resetar a página inteira.
- Detect regional deve evitar duplicatas por IoU.
- IDs de layers existentes devem ser preservados quando houver correspondência.

Merge por ação:

| Ação | Merge recomendado |
|---|---|
| Detect | substituir detecções dentro da região, preservar fora |
| OCR | atualizar texto original só das layers intersectadas |
| Translate | atualizar tradução só das layers intersectadas |
| Inpaint | atualizar bitmap inpaint só na região pintada |
| Preview | renderizar página inteira usando assets atualizados |

---

### 3.13. `PageActionResult` com tipos fortes

Na Fase 4A, usar enum/tipo forte, não `String` solta.

Rust:

```rust
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PageActionMode {
    Global,
    Regional,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangedAsset {
    Brush,
    Mask,
    Inpaint,
    Rendered,
    Preview,
    ProjectJson,
}

#[derive(Serialize)]
pub struct PageActionResult {
    pub action: String,
    pub mode: PageActionMode,
    pub bbox: Option<[u32; 4]>,
    pub changed_assets: Vec<ChangedAsset>,
    pub changed_layers: Vec<String>,
    pub message: String,
}
```

TypeScript:

```ts
type PageActionMode = "global" | "regional"

type ChangedAsset =
  | "brush"
  | "mask"
  | "inpaint"
  | "rendered"
  | "preview"
  | "project_json"

type PageActionResult = {
  action: "detect" | "ocr" | "translate" | "inpaint"
  mode: PageActionMode
  bbox?: [number, number, number, number] | null
  changed_assets: ChangedAsset[]
  changed_layers: string[]
  message: string
}
```

---

### 3.14. Estado de loading por ação

Adicionar no store/UI:

```ts
activePageAction: null | "detect" | "ocr" | "translate" | "inpaint"
```

Enquanto houver ação ativa:

- desabilitar botões de ação;
- mostrar spinner no botão ativo;
- impedir duplo clique;
- impedir execução concorrente;
- limpar estado em `finally`.

---

### 3.15. Indicador visual de escopo sem custo pesado

Adicionar na toolbar:

```txt
Escopo: Página inteira
```

ou:

```txt
Escopo: Região mascarada
```

ou:

```txt
Escopo: será confirmado ao executar
```

A UI não deve escanear o PNG da máscara em todo render.

Criar metadata/cache leve:

```ts
maskStats: {
  status: "unknown" | "empty" | "nonempty"
  bbox?: [number, number, number, number]
  updatedAt?: number
  version?: number
}
```

Atualizar `maskStats` quando:

- stroke de máscara for aplicado;
- borracha apagar máscara;
- `clearMaskLayer` rodar;
- página for carregada;
- backend retornar `PageActionResult`.

A decisão final global/regional continua sendo do backend.

---

### 3.16. Cache-bust genérico

Separar tipo geral de tipo mutável:

```ts
type BitmapLayerKey = "base" | "mask" | "inpaint" | "brush" | "rendered" | "preview"
type MutableBitmapLayerKey = Exclude<BitmapLayerKey, "base">
```

`bumpBitmapLayerVersion` deve aceitar preferencialmente `MutableBitmapLayerKey`.

Cobrir:

- `brush`;
- `mask`;
- `inpaint`;
- `rendered`;
- `preview`, se existir.

A layer `base` só deve receber bump se houver fluxo real que reescreva a imagem base.

---

### 3.17. `clearMaskLayer` bem definido

`clearMaskLayer` limpa somente a mask bitmap da página atual.

Não deve:

- alterar máscaras de outras páginas;
- apagar brush;
- apagar inpaint;
- apagar text layers;
- apagar traduções;
- remover arquivo se o sistema espera path válido.

Deve:

1. limpar o bitmap da máscara no disco ou substituir por PNG vazio do mesmo tamanho da página;
2. manter o path real da máscara válido, se o sistema depender dele;
3. atualizar `bitmapLayerVersions.mask`;
4. atualizar `maskStats.status = "empty"`;
5. marcar projeto como dirty;
6. marcar preview como stale somente se a máscara afetar preview/render atual.

Após limpar máscara:

> `runMaskedAction` deve cair em modo global.

---

### 3.18. `EditorBitmapOverlay` deve validar dimensões

O overlay deve garantir que o bitmap tenha as mesmas dimensões da imagem base/página.

Critério:

```txt
overlay.width == page.width
overlay.height == page.height
```

Se houver mismatch:

- emitir warning no console/log;
- renderizar com escala controlada;
- nunca distorcer silenciosamente;
- se incompatível, não renderizar overlay e mostrar aviso técnico.

---

### 3.19. Preservar dirty state, undo/redo e project.json limpo

Toda ação que alterar layer, bitmap ou texto deve:

1. marcar projeto como dirty;
2. marcar preview como stale;
3. preservar seleção quando possível;
4. criar entrada no histórico/undo se o editor já tiver esse sistema;
5. não salvar dados temporários no `project.json`.

Durante a Fase 0:

- confirmar se existe undo/redo;
- se existir, toda mutação deve criar entry;
- se não existir, documentar explicitamente como pendência.

---

### 3.20. Contexto da Obra precisa funcionar de ponta a ponta

O contexto da obra não pode ser apenas um campo visual.

Ele deve funcionar em quatro camadas:

1. **Persistência**: salvar no projeto.
2. **UI**: usuário consegue editar/importar/consultar.
3. **Pipeline**: tradução recebe contexto real.
4. **QA/Logs**: é possível confirmar que contexto foi usado.

Regra:

> Se o usuário preencher contexto da obra, a tradução precisa usar esse contexto para nomes, termos, tom, relações e continuidade. Se não usar, deve ser considerado bug.

---

## 4. Plano v15.6 para implementação

---

## Fase 0 — Auditoria obrigatória

Antes de implementar qualquer alteração, auditar o código real.

### Verificações

1. Confirmar onde `renderer.py` calcula `safe_text_box`.
2. Confirmar se estes arquivos estão realmente sem callers:
   - `pipeline/typesetter/text_fit_guard.py`;
   - `pipeline/layout/safe_area.py`;
   - `pipeline/layout/connected_balloon_splitter.py`.
3. Confirmar como `renderer.py` importa módulos vizinhos hoje.
4. Confirmar se `applyBitmapStroke` altera corretamente:
   - arquivo PNG no disco;
   - `currentPage.image_layers.brush/mask`;
   - `selectedImageLayerKey`;
   - preview stale;
   - dirty state.
5. Confirmar se o bug do brush é cache.
6. Confirmar se existe sistema de undo/redo.
7. Confirmar onde estão os botões Detect/OCR/Traduzir/Inpaint.
8. Confirmar se `deleteSelectedLayer` já existe.
9. Confirmar se existe hook próprio para carregar imagem no stage.
10. Confirmar se preview/render/inpaint são reescritos no mesmo path.
11. Confirmar onde `qa_report.json` é gerado e como flags são propagadas.
12. Confirmar se há padrão de toast/loading/error handling na UI.
13. Confirmar se já existe algum schema/campo de contexto da obra.
14. Confirmar se a tradução atual aceita prompt/contexto extra.
15. Confirmar onde ficam metadados do projeto e capítulos no `project.json`.

### Entregável da Fase 0

Documentar:

- diagnóstico do brush;
- diagnóstico do texto cortado;
- diagnóstico de cache de bitmap;
- diagnóstico do contexto da obra atual;
- arquivos reais tocados;
- riscos encontrados;
- funções confirmadas;
- funções inexistentes;
- existência ou ausência de undo/redo;
- convenção de bbox encontrada no código atual, se houver.

---

## Fase 0.5 — Resolver warnings Rust de `internet_context.rs`

### Objetivo

Remover os 4 warnings atuais do `npm run tauri dev` sem esconder problema com `#[allow(dead_code)]` indevido.

Warnings atuais:

```txt
InternetContextSourceState is never constructed
InternetContextConfig is never constructed
source_states is never used
cache_path is never used
```

Arquivo:

```txt
src-tauri/src/internet_context.rs
```

### Auditoria obrigatória

Rodar:

```bash
cd src-tauri
rg "internet_context|InternetContext|source_states|cache_path" src
```

PowerShell sem ripgrep:

```powershell
Get-ChildItem -Recurse src -Include *.rs | Select-String "internet_context|InternetContext|source_states|cache_path"
```

### Regra de decisão

#### Caso A — Não tem caller real agora

Se `internet_context.rs` não tem UI, command ou pipeline usando agora:

- remover `mod internet_context;` do build atual;
- manter o arquivo apenas se houver intenção clara de uso futuro;
- documentar como pendência.

Esta é a opção preferida se o módulo está morto/inativo.

#### Caso B — Funcionalidade futura

Pode usar feature flag:

```rust
#[cfg(feature = "internet-context")]
mod internet_context;
```

Mas antes de editar `Cargo.toml`, ler o bloco `[features]`.

Nunca substituir:

```toml
default = [...]
```

por:

```toml
default = []
```

Adicionar somente:

```toml
internet-context = []
```

Se a feature compilar com warnings quando ativada, documentar que ela ainda é futura ou conectar commands reais sob a mesma feature.

#### Caso C — Deve existir agora

Conectar commands reais, registrar no `invoke_handler` e criar uso real na UI/pipeline.

Não criar command falso só para calar warning.

### Não fazer

- Não aplicar `#[allow(dead_code)]` como solução principal.
- Não criar command Tauri falso.
- Não quebrar `cargo run --no-default-features`.
- Não sobrescrever features existentes no `Cargo.toml`.

### Validação

```bash
cd src-tauri
cargo check --no-default-features
cargo test
```

PowerShell com warnings como erro:

```powershell
$env:RUSTFLAGS="-D warnings"
cargo check --no-default-features
Remove-Item Env:\RUSTFLAGS
```

Critério:

> `npm run tauri dev` não deve mais exibir warnings de `internet_context.rs`.

---

## Fase 1A — Wire-up do v15.2 para corrigir clipping real

### Objetivo

Corrigir o bug de texto cortado lateralmente usando `safe_area.py` e `text_fit_guard.py`, mas com tentativa real de auto-fit antes de emitir erro.

### Arquivos esperados

- `pipeline/typesetter/renderer.py`
- `pipeline/layout/safe_area.py`
- `pipeline/typesetter/text_fit_guard.py`
- `pipeline/qa/*`

### 1A.1. Integrar `build_safe_area` no renderer

Usar fallback seguro:

```python
fallback_safe_text_box = current_manual_safe_text_box

try:
    safe_area_result = build_safe_area(
        balloon_bbox=layer.get("balloon_bbox") or layer.get("layout_bbox") or layer["bbox"],
        page_width=page_width,
        page_height=page_height,
        balloon_polygon=layer.get("balloon_polygon"),
        connected_lobe_bboxes=layer.get("connected_lobe_bboxes"),
        balloon_type=layer.get("balloon_type", "white"),
    )

    safe_text_box = safe_area_result.get("safe_bbox") or fallback_safe_text_box

except Exception as exc:
    safe_text_box = fallback_safe_text_box
    qa_flags.append({
        "type": "safe_area_fallback_used",
        "severity": "warning",
        "message": str(exc),
        "region_id": layer.get("id"),
        "page": page_index + 1,
    })
```

### 1A.2. Integrar `validate_rendered_text_fit` com alpha mask

A alpha mask deve reutilizar a mesma função de render usada no render final.

Se `measured_ink_bbox is None`, emitir `empty_text_render`.

### 1A.3. Implementar loop de correção antes do QA crítico

Critérios sugeridos:

```python
font_size_min = max(10, int(original_font_size * 0.65))
max_fit_attempts = 8
```

Se ainda extrapolar:

- renderizar melhor esforço;
- emitir `text_clipped` com `severity=critical`;
- incluir `ink_bbox`, `safe_bbox`, `font_size_final`, `font_size_original` e `attempts`.

### 1A.4. Garantir emissão no `qa_report.json`

O `qa_report.json` precisa receber:

- `text_clipped`;
- `text_overflow`;
- `safe_area_fallback_used`;
- `empty_text_render`;
- `region_id`;
- `page`;
- `bbox`;
- `safe_bbox`;
- `ink_bbox`;
- `severity`.

---

## Fase 1B — Integrar `connected_balloon_splitter` com baixo risco

### Objetivo

Melhorar cálculo de área segura para balões conectados sem colocar em risco a correção principal de clipping.

### Tarefas

Em `balloon_layout.py`, após detectar contornos/balões, chamar `detect_connected_balloon`.

Se `confidence >= 0.5`, enriquecer a layer com:

```python
{
    "connected_lobe_bboxes": [...],
    "balloon_type": "...",
    "connected_balloon_confidence": confidence,
}
```

Regras:

- não quebrar layers antigas;
- preservar fallback;
- se houver dúvida, não aplicar split;
- registrar QA/debug leve quando balão conectado for detectado.

---

## Fase 2 — Brush/máscara/borracha: overlay, cursor e cache-bust seguro

### Arquivos novos

- `src/components/editor/stage/EditorBitmapOverlay.tsx`
- `src/components/editor/stage/EditorPaintCursor.tsx`

### Arquivos modificados

- `src/lib/stores/editorStore.ts`
- `src/components/editor/stage/useEditorStageController.ts`
- `src/components/editor/stage/EditorStage.tsx`
- hook de carregamento de imagem, se existir

### 2.1. Cache-bust correto e genérico

```ts
type BitmapLayerKey = "base" | "mask" | "inpaint" | "brush" | "rendered" | "preview"
type MutableBitmapLayerKey = Exclude<BitmapLayerKey, "base">

bitmapLayerVersions: Partial<Record<BitmapLayerKey, number>>
```

```ts
bumpBitmapLayerVersion(layerKey: MutableBitmapLayerKey)
```

### 2.2. Criar `EditorBitmapOverlay`

Responsabilidades:

- receber path real;
- receber `version/cacheKey`;
- converter grayscale/luminância em alpha;
- renderizar overlay RGBA colorido;
- aceitar `color` e `opacity`;
- validar dimensões.

Defaults:

```ts
brushColor = "#48B0FF"
maskColor = "#7C5CFF"
paintOpacity = 0.65
```

### 2.3. Criar `EditorPaintCursor`

- círculo Konva;
- aparece em brush, repairBrush/mask e eraser;
- `radius = brushSize / 2`;
- segue `cursorPoint`;
- não interfere na seleção de texto.

### 2.4. Ajustar `useEditorStageController`

Adicionar:

- `cursorPoint`;
- `handleStageMouseEnter`;
- `handleStageMouseLeave`;
- `window mouseup` global quando estiver pintando;
- não limpar `paintStroke` antes de `applyBitmapStroke` terminar.

### 2.5. Ajustar store

Adicionar:

```ts
brushColor
maskColor
paintOpacity
bitmapLayerVersions
activePageAction
maskStats
setBrushColor
setMaskColor
setPaintOpacity
bumpBitmapLayerVersion
clearMaskLayer
previewFaithfulSaveRender
setWorkingOriginal
runMaskedAction
```

---

## Fase 3 — UI cleanup sem regressão

### Arquivos

- `src/pages/Editor.tsx`
- `src/components/editor/LayersPanel.tsx`
- `src/components/editor/PropertyEditor.tsx`

### 3.1. Toolbar superior

Os botões Detect/OCR/Traduzir/Inpaint já estão na toolbar.

Trocar chamadas atuais por:

```ts
runMaskedAction("detect")
runMaskedAction("ocr")
runMaskedAction("translate")
runMaskedAction("inpaint")
```

### 3.2. Botão único de preview

Unificar:

- `Preview fiel`;
- `Salvar+Render`.

Novo botão:

```txt
Preview fiel (Salva + Render)
```

Comportamento:

1. `commitEdits`;
2. salvar estado atual;
3. renderizar preview fiel;
4. manter página atual;
5. atualizar preview stale;
6. atualizar `bitmapLayerVersions.rendered` ou equivalente.

### 3.3. Controles contextuais

Se `toolMode` for brush/mask:

- tamanho;
- cor;
- opacidade.

Se `toolMode` for eraser:

- tamanho;
- esconder cor.

Se `selectedTextLayer`:

- fonte;
- tamanho;
- cor;
- alinhamento, se já existir;
- stroke/sombra, se já existir.

### 3.4. Indicador visual de escopo

Adicionar na toolbar:

- `Escopo: Página inteira`;
- `Escopo: Região mascarada`;
- `Escopo: será confirmado ao executar`.

Também adicionar:

```txt
Limpar máscara
```

### 3.5. Estado de loading por ação

Enquanto `activePageAction !== null`:

- desabilitar botões;
- mostrar spinner;
- impedir duplo clique;
- limpar loading em `finally`.

### 3.6. `LayersPanel`

Trocar filename por labels fixos:

```ts
const BITMAP_LAYER_LABELS = {
  base: "Base",
  mask: "Máscara",
  inpaint: "Inpaint",
  brush: "Brush",
  rendered: "Render final",
}
```

### 3.7. `PropertyEditor`

Texto Original:

- textarea editável;
- draft local;
- blur ou Ctrl+Enter chama `setWorkingOriginal`;
- marcar dirty/stale.

Rodapé direito:

- só lixeira vermelha;
- disabled se não houver layer selecionada.

---

## Fase 4A — Infra regional sem mudar comportamento

### Arquivos

- `src/lib/tauri.ts`
- `src-tauri/src/commands/project.rs`
- `src-tauri/src/lib.rs`
- `src-tauri/src/pipeline.rs`
- `src/lib/stores/editorStore.ts`

### Criar command

```rust
run_page_action_with_optional_mask
```

### Retorno estruturado obrigatório

Usar `PageActionResult`, `PageActionMode` e `ChangedAsset` com enum, conforme seção 3.13.

### Backend deve

1. localizar mask layer da página;
2. verificar se existe;
3. verificar se tem pixels ativos usando `MASK_ACTIVE_THRESHOLD`;
4. calcular `mask_bbox` usando `[x1, y1, x2, y2)`;
5. retornar/logar `mode`, `bbox`, `action`, `changed_assets`, `changed_layers`, `message`;
6. inicialmente pode delegar para ação global atual.

---

## Fase 4B — Inpaint regional

Adicionar em `pipeline/main.py`:

```bash
--mask <path>
--bbox x1,y1,x2,y2
```

Para inpaint:

- usar bbox apenas para crop/performance;
- usar pixels reais da máscara como seleção final;
- aplicar threshold;
- preservar fora da máscara;
- mergear resultado na página inteira;
- retornar `changed_assets` padronizado.

---

## Fase 4C — Translate regional

Primeira versão pragmática:

- usar interseção `layer.bbox` x `mask_bbox`;
- não exigir leitura pixel-perfect da máscara;
- preservar traduções fora da bbox;
- não recriar layers fora da máscara;
- não apagar texto manual;
- registrar `changed_layers`.

Refinamento por pixels ativos fica como melhoria futura.

---

## Fase 4D — OCR regional

Prioridade:

1. Se já existe `text_layer` intersectando:
   - atualizar `text_original`;
   - preservar estilo, tradução e posição quando possível.
2. Se OCR detectar texto sem layer correspondente:
   - criar nova layer apenas dentro da região;
   - marcar como `auto-generated`.
3. Não apagar layers manuais.
4. Não resetar tradução já editada manualmente, a menos que a ação seja explicitamente de reprocessamento.

---

## Fase 4E — Detect regional

Detect é o mais arriscado. Implementar por último.

Regras:

1. expandir `mask_bbox` com margem, exemplo `64px`;
2. clamp no tamanho da página;
3. rodar detector no crop expandido;
4. converter coordenadas para página inteira;
5. preservar IDs por IoU quando possível;
6. criar novo ID apenas quando não houver correspondência;
7. não apagar layer manual se não for detectada novamente;
8. apagar/substituir apenas auto-generated dentro da região;
9. evitar duplicatas por IoU.

---

# Fase 6 — Contexto da Obra funcionando de ponta a ponta

> Esta fase é obrigatória. O contexto da obra precisa funcionar de verdade no fluxo de tradução, não apenas existir na UI.

---

## 6.1. Objetivo

Fazer o TraduzAi usar contexto da obra para melhorar tradução, consistência e continuidade.

O sistema deve considerar:

- título da obra;
- sinopse;
- gênero;
- tom;
- público;
- nomes de personagens;
- relações entre personagens;
- glossário de termos;
- honoríficos;
- nomes próprios;
- estilo de fala;
- resumo de capítulos anteriores;
- resumo do capítulo atual;
- notas do tradutor;
- regras de tradução;
- termos proibidos ou preferidos;
- memória de traduções anteriores.

---

## 6.2. Problema atual a resolver

Sem contexto, a tradução pode:

- trocar gênero/personagem;
- traduzir nome próprio errado;
- perder continuidade;
- traduzir o mesmo termo de formas diferentes;
- errar tom de fala;
- transformar personagem formal em informal;
- traduzir golpe, clã, cargo, cidade ou habilidade de forma inconsistente;
- ignorar relação entre personagens;
- perder sentido em balões curtos como “ele?”, “aquilo?”, “você também?”.

Regra:

> Toda tradução precisa receber contexto suficiente para interpretar balões curtos e manter consistência entre páginas.

---

## 6.3. Schema recomendado

Adicionar ao projeto um bloco persistente:

```ts
type WorkContext = {
  title?: string
  originalTitle?: string
  languageSource?: string
  languageTarget?: string
  genre?: string[]
  synopsis?: string
  tone?: string
  audience?: string

  translationRules?: string[]
  translatorNotes?: string

  characters?: WorkCharacter[]
  glossary?: WorkGlossaryEntry[]
  styleGuide?: WorkStyleGuide

  chapterContext?: ChapterContext
  previousChapterSummaries?: ChapterSummary[]
  translationMemory?: TranslationMemoryEntry[]

  updatedAt?: string
  version?: number
}

type WorkCharacter = {
  id: string
  name: string
  aliases?: string[]
  gender?: string
  role?: string
  relationshipNotes?: string
  speechStyle?: string
  doNotTranslateName?: boolean
  preferredPortugueseName?: string
}

type WorkGlossaryEntry = {
  id: string
  source: string
  target: string
  category?: "name" | "place" | "skill" | "title" | "item" | "organization" | "other"
  notes?: string
  locked?: boolean
}

type WorkStyleGuide = {
  formality?: "auto" | "informal" | "neutral" | "formal"
  honorifics?: "keep" | "adapt" | "remove"
  profanityLevel?: "soften" | "keep" | "intensify"
  soundEffects?: "keep_original" | "translate" | "both"
  names?: "keep_original" | "adapt_when_common" | "custom"
}

type ChapterContext = {
  chapterNumber?: string
  chapterTitle?: string
  currentSummary?: string
  importantEvents?: string[]
  activeCharacters?: string[]
  activeTerms?: string[]
}

type ChapterSummary = {
  chapterNumber: string
  summary: string
}

type TranslationMemoryEntry = {
  source: string
  target: string
  context?: string
  page?: number
  regionId?: string
}
```

No `project.json`, guardar em:

```json
{
  "work_context": {}
}
```

Se o projeto já tiver outro padrão, adaptar sem quebrar retrocompatibilidade.

---

## 6.4. Migração/backward compatibility

Se um projeto antigo não tiver `work_context`:

- criar valor default vazio;
- não quebrar carregamento;
- não forçar usuário a preencher;
- tradução continua funcionando sem contexto, mas com qualidade menor.

Default:

```json
"work_context": {
  "languageSource": "auto",
  "languageTarget": "pt-BR",
  "genre": [],
  "characters": [],
  "glossary": [],
  "translationRules": [],
  "previousChapterSummaries": [],
  "translationMemory": [],
  "version": 1
}
```

---

## 6.5. UI obrigatória para Contexto da Obra

Criar ou ajustar painel no editor/projeto:

```txt
Contexto da Obra
```

Seções:

1. **Informações gerais**
   - Título;
   - Título original;
   - idioma original;
   - idioma alvo;
   - gênero;
   - sinopse;
   - tom.

2. **Personagens**
   - nome;
   - aliases;
   - papel;
   - gênero, se o usuário quiser informar;
   - estilo de fala;
   - relacionamento/notas;
   - opção “não traduzir nome”.

3. **Glossário**
   - termo original;
   - tradução preferida;
   - categoria;
   - locked;
   - notas.

4. **Regras de tradução**
   - manter honoríficos;
   - adaptar honoríficos;
   - manter nomes;
   - traduzir golpes/habilidades;
   - tom formal/informal;
   - observações do tradutor.

5. **Resumo**
   - resumo do capítulo atual;
   - eventos importantes;
   - personagens ativos;
   - termos ativos;
   - resumo dos capítulos anteriores.

6. **Importar contexto**
   - colar texto livre;
   - importar `.txt`;
   - importar `.json`;
   - converter texto livre em campos estruturados, se houver LLM/provider disponível.

7. **Exportar contexto**
   - exportar `.json`;
   - exportar `.txt`.

---

## 6.6. Store/frontend

Adicionar no store:

```ts
workContext: WorkContext

setWorkContextPatch(patch: Partial<WorkContext>): void
addCharacter(character: WorkCharacter): void
updateCharacter(id: string, patch: Partial<WorkCharacter>): void
removeCharacter(id: string): void

addGlossaryEntry(entry: WorkGlossaryEntry): void
updateGlossaryEntry(id: string, patch: Partial<WorkGlossaryEntry>): void
removeGlossaryEntry(id: string): void

importWorkContextFromText(text: string): Promise<void>
importWorkContextFromJson(json: unknown): void
exportWorkContextJson(): WorkContext

buildTranslationContextForPage(pageIndex: number): TranslationContextPayload
```

Toda mutação deve:

- marcar projeto dirty;
- ser salva no `project.json`;
- preservar compatibilidade com projetos antigos;
- não travar editor se campos estiverem vazios.

---

## 6.7. Payload para tradução

Criar payload compacto para cada página/bloco.

Exemplo:

```ts
type TranslationContextPayload = {
  work: {
    title?: string
    synopsis?: string
    genre?: string[]
    tone?: string
  }
  styleGuide?: WorkStyleGuide
  characters: Array<{
    name: string
    aliases?: string[]
    role?: string
    speechStyle?: string
    relationshipNotes?: string
  }>
  glossary: Array<{
    source: string
    target: string
    category?: string
    locked?: boolean
    notes?: string
  }>
  chapter: {
    chapterNumber?: string
    currentSummary?: string
    importantEvents?: string[]
    activeCharacters?: string[]
    activeTerms?: string[]
  }
  previousSummaries?: ChapterSummary[]
  translationMemory?: TranslationMemoryEntry[]
  pageContext?: {
    pageIndex: number
    neighboringTexts?: string[]
    previousPageTexts?: string[]
    nextPageTexts?: string[]
  }
}
```

Regra:

> O payload enviado ao tradutor deve ser compacto. Não enviar contexto gigante inteiro em toda chamada se isso estourar limite/token.

---

## 6.8. Integração no tradutor

Localizar a função atual que traduz:

- bloco único;
- página;
- capítulo;
- reprocessamento por região.

Antes de chamar o provider/LLM/tradutor, montar prompt/contexto:

```txt
Você está traduzindo uma obra.
Use o contexto abaixo para manter consistência.
Não invente fatos.
Preserve nomes marcados como locked.
Use o glossário obrigatório.
Mantenha o tom dos personagens.
Traduza para pt-BR natural.

[Contexto da obra]
...

[Glossário obrigatório]
...

[Personagens]
...

[Resumo do capítulo]
...

[Balões vizinhos]
...

[Texto a traduzir]
...
```

Regras:

- Glossário `locked` é obrigatório.
- Nomes com `doNotTranslateName` não devem ser traduzidos.
- Se houver conflito entre contexto e texto atual, priorizar texto atual e emitir warning.
- Não usar contexto para inventar informação.
- Para balões curtos, usar `neighboringTexts`.

---

## 6.9. Contexto por página e balões vizinhos

Para cada tradução, incluir quando possível:

- texto anterior da mesma página;
- texto seguinte da mesma página;
- textos da página anterior;
- textos da próxima página;
- speaker/personagem se já existir;
- bbox/região se relevante.

Exemplo:

```ts
pageContext: {
  pageIndex,
  neighboringTexts: [
    "Eu já disse para você não ir.",
    "Mas ele está lá!",
    "Então vamos juntos."
  ]
}
```

Isso ajuda em falas como:

```txt
Ele?
Aquilo?
Você também?
```

---

## 6.10. Translation Memory

Após traduzir, salvar pares úteis:

```json
{
  "source": "Shadow Monarch",
  "target": "Monarca das Sombras",
  "context": "título/habilidade",
  "page": 12,
  "regionId": "text_123"
}
```

Regras:

- não salvar duplicatas óbvias;
- preferir entradas confirmadas pelo usuário;
- entradas do glossário têm prioridade sobre translation memory;
- permitir limpar memory no futuro.

---

## 6.11. Importação de contexto por texto livre

Usuário pode colar algo como:

```txt
Obra: Solo Leveling
Tom: ação, fantasia, sério
Personagens:
Sung Jin-Woo: protagonista, não traduzir nome
Cha Hae-In: caçadora rank S
Termos:
Shadow Monarch = Monarca das Sombras
Hunter = Caçador
```

O sistema deve:

1. armazenar o texto bruto;
2. se houver parser simples, extrair campos básicos;
3. se houver LLM/provider, estruturar automaticamente;
4. mostrar prévia antes de sobrescrever dados existentes.

Regra:

> Importar contexto nunca deve apagar personagens/glossário existentes sem confirmação.

---

## 6.12. Rust/Tauri commands para contexto

Criar commands se necessário:

```rust
#[tauri::command]
pub async fn get_work_context(project_path: String) -> Result<WorkContext, String>

#[tauri::command]
pub async fn update_work_context(
    project_path: String,
    context: WorkContext,
) -> Result<(), String>

#[tauri::command]
pub async fn import_work_context_text(
    project_path: String,
    text: String,
) -> Result<WorkContext, String>
```

Se o frontend já salva projeto inteiro, não duplicar commands sem necessidade. Usar padrão real do projeto.

---

## 6.13. Pipeline Python

Se a tradução roda no Python, adicionar suporte a contexto:

- carregar `work_context` do projeto;
- receber contexto via JSON;
- montar prompt/entrada do tradutor;
- logar resumo do contexto usado;
- não quebrar tradução sem contexto.

Argumentos possíveis:

```bash
--work-context <path>
```

ou usar o próprio `project.json`.

Regra:

> Preferir ler do `project.json` para evitar divergência, salvo se arquitetura atual já usa payload separado.

---

## 6.14. QA/logs do contexto

Toda tradução deve registrar, em log ou debug:

```json
{
  "translation_context_used": true,
  "glossary_terms_applied": ["Shadow Monarch"],
  "characters_in_context": ["Sung Jin-Woo", "Cha Hae-In"],
  "context_version": 1
}
```

Se não houver contexto:

```json
{
  "translation_context_used": false,
  "reason": "empty_work_context"
}
```

Se glossário locked foi ignorado:

```json
{
  "type": "glossary_locked_term_not_applied",
  "severity": "warning",
  "source": "Shadow Monarch",
  "expected": "Monarca das Sombras"
}
```

---

## 6.15. UI de confirmação de uso do contexto

Na área de tradução, mostrar status:

```txt
Contexto: ativo
```

ou:

```txt
Contexto: vazio
```

Ao passar o mouse ou abrir detalhes, mostrar:

- quantos personagens;
- quantos termos no glossário;
- se há resumo do capítulo;
- se há regras de tradução.

Exemplo:

```txt
Contexto ativo: 8 personagens, 23 termos, resumo do capítulo preenchido.
```

---

## 6.16. Testes do Contexto da Obra

### Unitários frontend

- `workContext default não quebra projeto antigo`;
- `addGlossaryEntry salva dirty`;
- `locked glossary aparece no payload`;
- `buildTranslationContextForPage inclui personagens e termos`;
- `importWorkContextFromJson não apaga dados sem confirmação`.

### Python

- tradução recebe `work_context`;
- glossário locked é incluído no prompt;
- contexto vazio não quebra tradução;
- balões vizinhos entram no payload;
- translation memory é consultada.

### E2E/manual

- preencher título/sinopse/personagem/glossário;
- traduzir página;
- confirmar que termo locked foi usado;
- mudar termo no glossário;
- retraduzir página;
- confirmar nova tradução;
- abrir projeto novamente;
- confirmar contexto persistido.

---

## 6.17. Critério de aceite do Contexto da Obra

A feature só está pronta quando:

- [ ] existe UI para editar contexto;
- [ ] contexto salva no projeto;
- [ ] contexto recarrega ao abrir projeto;
- [ ] tradução usa glossário/personagens/regras;
- [ ] glossário locked é respeitado;
- [ ] balões vizinhos ajudam na tradução;
- [ ] logs mostram que contexto foi usado;
- [ ] contexto vazio não quebra tradução;
- [ ] importação `.txt` ou colagem funciona minimamente;
- [ ] importação `.json` funciona;
- [ ] projeto antigo sem contexto continua abrindo.

---

## Fase 7 — Testes e QA final

Testes devem ser adicionados por fase quando possível. Esta fase consolida cobertura final.

### Python

Criar/ajustar:

- `test_text_fit_wire_up.py`
- `test_safe_area_fallback.py`
- `test_text_clipped_qa_report.py`
- `test_empty_text_render_getbbox_none.py`
- `test_mask_bbox_arg.py`
- `test_mask_threshold.py`
- `test_bbox_half_open_convention.py`
- `test_regional_inpaint_preserves_outside_mask.py`
- `test_regional_translate_preserves_outside_mask.py`
- `test_regional_ocr_preserves_outside_mask.py`
- `test_regional_detect_preserves_outside_mask.py`
- `test_work_context_translation_payload.py`
- `test_work_context_glossary_locked.py`
- `test_work_context_empty_does_not_break_translation.py`

### Rust

Criar/ajustar:

- `test_mask_has_nonzero_pixels`
- `test_mask_bounding_box`
- `test_mask_threshold_ignores_low_alpha_noise`
- `test_empty_mask_goes_global`
- `test_nonempty_mask_goes_regional`
- `test_bbox_half_open_convention`
- `test_page_action_result_serializes`
- `test_work_context_schema_backward_compatible`

### Vitest

Criar/ajustar:

- `editorStore.runMaskedAction global fallback`
- `editorStore.runMaskedAction regional`
- `editorStore.activePageAction prevents double click`
- `bitmapLayerVersions não polui project.json`
- `setWorkingOriginal marca dirty/stale`
- `bumpBitmapLayerVersion updates only runtime state`
- `scope indicator renders expected label`
- `workContext default state`
- `workContext glossary payload`
- `workContext import json`

### Playwright

Criar/ajustar:

- `editor-brush-persists`
- `editor-brush-mouseup-outside`
- `editor-mask-overlay-color`
- `editor-preview-faithful-save-render`
- `editor-original-text-editable`
- `editor-scope-indicator`
- `editor-action-loading-prevents-double-click`
- `editor-mask-regional-inpaint`
- `editor-work-context-edit-save-reload`
- `editor-work-context-glossary-used-in-translation`

---

## 5. Ordem de execução recomendada

1. Criar branch:

```bash
git checkout chore/repo-cleanup-disable-lab
git pull
git checkout -b feat/editor-brush-mask-typesetting
```

2. Implementar Fase 0.
3. Implementar Fase 0.5 em commit isolado.
4. Implementar Fase 1A em commit isolado.
5. Implementar Fase 1B em commit isolado.
6. Implementar Fase 2 em commit isolado.
7. Implementar Fase 3 em commit isolado.
8. Implementar Fase 4A em commit isolado.
9. Implementar Fase 4B em commit isolado.
10. Implementar Fase 4C em commit isolado.
11. Implementar Fase 4D em commit isolado.
12. Implementar Fase 4E em commit isolado.
13. Implementar Fase 6A: schema/persistência do contexto da obra.
14. Implementar Fase 6B: UI do contexto da obra.
15. Implementar Fase 6C: contexto no payload de tradução.
16. Implementar Fase 6D: logs/QA/translation memory.
17. Consolidar Fase 7 por área.

Regra:

> Cada fase deve ser reversível sem perder as anteriores.

Ordem prioritária segura:

```txt
Fase 0 → Fase 0.5 → Fase 1A → Fase 2 → Fase 6A → Fase 6C
```

Depois:

```txt
Fase 1B → Fase 3 → Fase 4A → Fase 4B → Fase 4C → Fase 4D → Fase 4E → Fase 6B → Fase 6D
```

---

## 6. Comandos finais de verificação

```bash
# Frontend
npm run build

# Rust/Tauri
cd src-tauri
cargo check --no-default-features
cargo check
cargo test

# Rust/Tauri com warnings como erro, opcional mas recomendado
# PowerShell:
# $env:RUSTFLAGS="-D warnings"; cargo check --no-default-features; Remove-Item Env:\RUSTFLAGS

# Pipeline
cd ../pipeline
python -m pytest -q
python -m pytest tests/regression/test_text_fit_wire_up.py -v

# E2E
cd ..
npx playwright test --grep "@editor-brush|@mask-regional|@work-context"
```

Se o projeto não tiver todos esses scripts/testes ainda, documentar quais existem, quais foram criados e quais ficaram pendentes.

---

## 7. Checklist manual

### Brush/máscara/borracha

- [ ] Brush desenhado continua visível após `mouseup`.
- [ ] Brush continua visível ao soltar mouse fora do canvas.
- [ ] Arquivo PNG realmente muda no disco.
- [ ] Path salvo no `project.json` não contém `?v=`.
- [ ] Overlay muda cor.
- [ ] Overlay muda opacidade.
- [ ] Cursor circular segue mouse.
- [ ] Borracha apaga brush/mask.
- [ ] Cache-bust funciona para brush/mask/inpaint/rendered/preview.

### Máscara regional

- [ ] Máscara vazia executa ação global.
- [ ] Máscara com pixels acima do threshold executa ação regional.
- [ ] Pixels abaixo do threshold não ativam máscara regional.
- [ ] Bbox segue `[x1, y1, x2, y2)`.
- [ ] Inpaint regional preserva fora da máscara.
- [ ] Translate regional preserva layers fora da máscara.
- [ ] OCR regional preserva layers fora da máscara.
- [ ] Detect regional não duplica layers fora da região.
- [ ] Detect regional não apaga layers manuais fora da máscara.
- [ ] Retorno estruturado informa `mode`, `bbox`, `changed_assets` e `changed_layers`.

### UI

- [ ] Botões Detect/OCR/Traduzir/Inpaint continuam na toolbar.
- [ ] Botões chamam `runMaskedAction`.
- [ ] Existe apenas um botão: `Preview fiel (Salva + Render)`.
- [ ] Existe indicador de escopo.
- [ ] Existe botão `Limpar máscara`.
- [ ] Durante ação ativa, botões ficam desabilitados.
- [ ] Clique duplo não dispara duas ações.
- [ ] Layers Bitmap mostra labels fixos.
- [ ] Texto Original é editável.
- [ ] Lixeira vermelha remove layer selecionada.

### Typesetting/QA

- [ ] Texto não corta lateralmente no capítulo de regressão.
- [ ] Renderer tenta auto-fit antes de emitir `text_clipped`.
- [ ] Auto-fit não altera permanentemente estilo salvo.
- [ ] `qa_report.json` registra `text_clipped` quando forçado.
- [ ] `safe_area_fallback_used` aparece como warning.
- [ ] `empty_text_render` aparece como warning.
- [ ] Render não quebra se `build_safe_area` falhar.

### Contexto da obra

- [ ] Existe UI de Contexto da Obra.
- [ ] Contexto salva no projeto.
- [ ] Contexto recarrega ao abrir projeto.
- [ ] É possível cadastrar personagem.
- [ ] É possível cadastrar glossário.
- [ ] Glossário locked é usado na tradução.
- [ ] Regras de tradução entram no payload.
- [ ] Balões vizinhos entram no payload.
- [ ] Logs mostram `translation_context_used`.
- [ ] Contexto vazio não quebra tradução.
- [ ] Importar `.txt` ou colar contexto funciona minimamente.
- [ ] Importar `.json` funciona.
- [ ] Projeto antigo sem contexto continua abrindo.

### Build/testes

- [ ] `npm run build` passa.
- [ ] `npm run tauri dev` não exibe warnings de `internet_context.rs`.
- [ ] `cargo check` passa.
- [ ] `cargo test` passa.
- [ ] `python -m pytest -q` passa.
- [ ] Playwright passa ou pendências são documentadas.

---

## 8. O que não fazer

- Não reescrever `applyBitmapStroke` sem diagnóstico.
- Não salvar `?v=` no path persistido.
- Não colocar dados temporários no `project.json`.
- Não remover botões flutuantes que não existem.
- Não remover seção "Ações Manuais" que não existe.
- Não implementar Detect regional antes do Inpaint/Translate/OCR regional.
- Não usar bbox como seleção final da máscara.
- Não considerar qualquer pixel maior que zero como máscara ativa.
- Não apagar layers fora da região mascarada.
- Não criar sistema novo de estilo se já não existir suporte.
- Não quebrar import Python em modo diferente de execução.
- Não juntar todas as fases em um único commit.
- Não retornar string solta para ação regional.
- Não esconder warnings de `internet_context.rs` com `#[allow(dead_code)]` sem justificativa.
- Não permitir clique duplo em ação pesada.
- Não apenas detectar clipping sem tentar corrigi-lo.
- Não deixar contexto da obra apenas como campo visual.
- Não ignorar glossário locked.
- Não sobrescrever contexto importado sem confirmação.
- Não alterar permanentemente estilo de texto por causa do auto-fit.

---

## 9. Entregável final esperado

Ao final, entregar:

- resumo técnico do que foi alterado;
- arquivos modificados;
- decisões tomadas;
- testes executados;
- bugs encontrados;
- pendências reais;
- comandos rodados;
- hash do commit;
- confirmação de que `project.json` não recebeu cache-bust temporário;
- confirmação de que ações regionais preservam conteúdo fora da máscara;
- confirmação da convenção de bbox `[x1, y1, x2, y2)`;
- confirmação do threshold usado para máscara;
- confirmação de que warnings de `internet_context.rs` foram removidos;
- exemplos de retorno `PageActionResult` global/regional;
- exemplo de `work_context` salvo no projeto;
- exemplo de payload de tradução com contexto da obra;
- evidência de que glossário locked foi usado.

---

## 10. Resumo executivo

Ordem completa:

```txt
Fase 0 → Fase 0.5 → Fase 1A → Fase 1B → Fase 2 → Fase 3 → Fase 4A → Fase 4B → Fase 4C → Fase 4D → Fase 4E → Fase 6 → Fase 7
```

Prioridade real:

1. Resolver warnings Rust atuais de `internet_context.rs` sem `#[allow(dead_code)]` indevido.
2. Corrigir texto cortado com wire-up do v15.2 e auto-fit real.
3. Corrigir brush/máscara com cache-bust seguro, overlay e cursor.
4. Fazer o Contexto da Obra funcionar na tradução.
5. Limpar UI sem regressão e mostrar escopo da ação.
6. Implementar máscara regional de forma incremental.
7. Cobrir com testes por fase.

Parte mais segura para começar:

```txt
Fase 0 → Fase 0.5 → Fase 1A → Fase 2 → Fase 6A → Fase 6C
```

Regra final:

> Se uma alteração não foi confirmada no código real, ela deve entrar como hipótese na Fase 0, não como implementação cega.
