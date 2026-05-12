# Prompt Completo — Refactor Profissional do Editor TraduzAi

Você é um engenheiro senior full-stack trabalhando no editor do TraduzAi.

Quero que você pegue o plano abaixo, revise internamente, use as skills disponíveis em:

```txt
C:\Users\PICHAU\.claude\skills
```

Use principalmente qualquer skill relacionada a:

- Karpathy/guidelines;
- planejamento de engenharia;
- React/TypeScript;
- Tauri/Rust;
- testes;
- UI/UX;
- debugging;
- refactor seguro;
- Python pipeline;
- design system;
- boas práticas de frontend.

## Importante

Não apenas faça um plano. Implemente o plano completo, fase por fase, validando cada fase antes de avançar.

Não trate isso como MVP pequeno. Quero a versão completa da funcionalidade, com qualidade profissional. Porém, faça de forma segura, incremental e testável.

## Regras de execução

1. Antes de começar, leia/inspecione as skills disponíveis em `C:\Users\PICHAU\.claude\skills`.
2. Use as skills relevantes para orientar arquitetura, testes, UI/UX, refactor e debugging.
3. Antes de alterar código, faça uma análise do estado atual do projeto.
4. Crie uma lista de arquivos afetados.
5. Implemente fase por fase.
6. Ao final de cada fase:
   - rode testes;
   - rode build/typecheck/lint se existirem;
   - faça verificação manual guiada;
   - adicione logs/debugs temporários quando necessário;
   - corrija erros encontrados;
   - só avance quando a fase estiver validada.
7. Não deixe TODOs críticos.
8. Não deixe código morto.
9. Não remova funções internas importantes sem fallback.
10. Se alguma fase exigir ajuste de schema, crie migration backward-compatible.
11. Se encontrar falha em Detect/OCR/Traduzir/Inpaint, corrija antes de continuar.
12. No final, rode uma bateria completa de testes e validação end-to-end.
13. Gere um relatório final com:
   - arquivos alterados;
   - fases concluídas;
   - testes executados;
   - bugs encontrados e corrigidos;
   - pendências reais, se houver;
   - como usar as novas funções.

Se bater limite de contexto, antes de parar gere um arquivo de continuidade com:

- estado atual;
- fases concluídas;
- fase atual;
- arquivos modificados;
- comandos já rodados;
- erros encontrados;
- próximos passos exatos.

Mas enquanto estiver dentro do contexto e ferramentas disponíveis, continue executando até concluir.

---

# Objetivo geral

Refatorar completamente o editor do TraduzAi para ficar profissional, intuitivo e próximo da experiência de um editor como Photoshop, mantendo o pipeline de tradução/inpaint funcional.

O editor deve permitir:

- zoom no topo;
- remoção de nomes técnicos como `001.png`/`001.jpg`;
- remoção da aba Propriedades;
- edição Original/Tradução ao lado do texto selecionado;
- auto-save a cada 3 segundos;
- preview e render automáticos;
- fontes funcionando corretamente em tempo real;
- fontes bundle e importação manual de fontes;
- Detectar/OCR/Traduzir/Inpaint funcionando com logs claros;
- Brush como pincel real;
- Máscara como laço estilo Photoshop;
- Borracha apagando Paint ou Mask corretamente;
- Bitmap como painel de camadas profissional;
- testes e debug por fase;
- validação final completa.

---

# Princípios obrigatórios

## 1. Diagnóstico antes do refactor

Antes de mexer pesado na UI, valide o estado atual dos botões:

- Detectar;
- OCR;
- Traduzir;
- Inpaint.

Nenhum desses pode falhar silenciosamente.

Adicione logs estruturados no frontend, Rust/Tauri e Python.

Exemplo de log esperado:

```txt
[EditorAction] start detect page=001
[EditorAction] progress detect 40%
[EditorAction] success detect changed_assets=[...]
[EditorAction] error ocr message="..."
```

## 2. Live Preview separado de Auto Fidelity Render

Não faça render Python a cada tecla digitada.

Use dois níveis:

### Live Preview

Instantâneo, feito no Konva:

- texto;
- fonte;
- tamanho;
- cor;
- contorno;
- sombra;
- posição;
- alinhamento.

### Auto Fidelity Render

Render fiel feito pelo Python em background:

- debounce de aproximadamente 1.5s;
- cancelamento/ignorar resultado antigo se o usuário editou de novo;
- indicador visual de status;
- não bloquear a UI.

## 3. Auto-save robusto

Não use debounce puro.

Use:

- dirty flag;
- timer a cada 3 segundos;
- flush obrigatório em troca de página;
- flush obrigatório antes de ações pipeline;
- flush obrigatório ao fechar;
- flush no Tauri `close-requested`;
- flush em `beforeunload`/`pagehide`/`visibilitychange` quando aplicável.

## 4. Coordenadas centralizadas

Criar módulo único:

```ts
src/lib/canvasGeometry.ts
```

Ele deve centralizar conversões:

```ts
screenToStagePoint()
stageToImagePoint()
imageToBitmapPoint()
screenToImagePoint()
imageToScreenPoint()
bboxToScreenRect()
```

Usar esse módulo obrigatoriamente em:

- Brush;
- Eraser;
- Lasso;
- FloatingTextEditor;
- hit-test;
- seleção;
- overlays.

## 5. Não duplicar texto no preview

Cuidado crítico:

Não pode acontecer:

```txt
render fiel com texto embutido + Konva.Text por cima = texto duplicado
```

Durante edição, o canvas deve usar:

```txt
base limpa/inpaint + Konva.Text live
```

O render fiel deve ser usado como cache/export/comparação, mas não pode gerar duplicação visual.

Defina explicitamente os modos de visualização:

```txt
Live Editing Mode:
- base limpa/inpaint/render sem texto duplicado
- Konva.Text por cima

Fidelity Mode:
- render fiel completo
- sem Konva.Text duplicado por cima
```

Se necessário, o Fidelity Mode pode ser interno/automático e o Live Mode continua sendo a visualização principal.

---

# Fase 0 — Diagnóstico obrigatório do pipeline

## Objetivo

Validar o estado atual antes do refactor.

## Tarefas

1. Localizar fluxo atual de:
   - Detectar;
   - OCR;
   - Traduzir;
   - Inpaint.

2. Adicionar logs no frontend:
   - início;
   - progresso;
   - sucesso;
   - erro;
   - assets modificados.

3. Adicionar logs no Rust/Tauri:
   - comando chamado;
   - argumentos principais;
   - página;
   - máscara usada ou não;
   - saída do pipeline;
   - erro completo.

4. Adicionar logs no Python:
   - action recebida;
   - input path;
   - output path;
   - changed assets;
   - stacktrace em erro.

5. Testar cada ação com:
   - página sem máscara;
   - página com máscara;
   - página com texto;
   - página sem texto.

6. Corrigir falhas silenciosas.

7. Criar arquivo de debug por run, por exemplo:

```txt
debug_runs/editor_refactor/phase_0_pipeline_diagnosis/
```

Com:

- logs do frontend;
- logs do Tauri;
- logs do Python;
- resultado esperado;
- resultado obtido.

## Verificações obrigatórias

Rodar:

```bash
npm run tauri dev
```

E, se existirem:

```bash
npm run typecheck
npm run lint
npm test
cd pipeline && python -m pytest -q
```

## Critério de pronto

- Detectar funciona ou mostra erro claro.
- OCR funciona ou mostra erro claro.
- Traduzir funciona ou mostra erro claro.
- Inpaint funciona ou mostra erro claro.
- Nenhum botão falha silenciosamente.
- Logs permitem diagnosticar o problema.

Não avance para a Fase 1 enquanto essa fase não estiver validada.

---

# Fase 1 — Limpeza visual da UI

## Objetivo

Remover ruído visual e reorganizar elementos principais.

## Tarefas

1. Mover controles de zoom do canto inferior direito para o topo, próximo das ações:
   - Detectar;
   - OCR;
   - Traduzir;
   - Inpaint.

2. Layout esperado:

```txt
[Detectar] [OCR] [Traduzir] [Inpaint]        [-] [100%] [+] [Fit]
```

3. Remover nomes técnicos como:

```txt
001.png
001.jpg
```

Da aba Bitmap.

4. Mostrar nomes amigáveis:

```txt
Original
Inpaint
Render
Paint
Mask
Texto
```

5. Remover da UI principal:
   - botão Salvar;
   - botão Preview Fiel;
   - botão Render;
   - ações manuais;
   - painel Propriedades antigo.

6. Manter funções internas via:
   - `Ctrl+S` para flush auto-save;
   - `Ctrl+Shift+R` para render fiel manual/debug;
   - painel debug opcional com `?debug=1`.

## Testes e debug

Após implementar:

```bash
npm run typecheck
npm run lint
npm run tauri dev
```

Verificar manualmente:

- zoom aparece no topo;
- zoom não aparece mais no canto inferior direito;
- não aparecem mais `001.png`/`001.jpg`;
- botões removidos sumiram da UI principal;
- atalhos ainda funcionam;
- nenhuma função interna importante foi deletada.

## Critério de pronto

UI limpa, sem quebrar ações do editor.

---

# Fase 2 — Fontes completas e confiáveis

## Objetivo

Corrigir fontes no canvas em tempo real e preparar suporte completo a fontes customizadas.

## Parte A — Fontes bundle

1. Mover fontes do projeto para pasta servível pelo app, por exemplo:

```txt
public/fonts/
```

2. Garantir que Tauri inclua as fontes em produção.

3. Criar:

```ts
src/lib/fonts.ts
```

Com `FONT_REGISTRY`.

4. Criar `@font-face` com:
   - family correta;
   - weight correto;
   - style correto;
   - path correto.

5. Garantir que os nomes usados no Konva sejam exatamente os nomes do CSS.

6. Pré-carregar fontes no boot do editor:

```ts
preloadEditorFonts()
await document.fonts.ready
forceRedrawTextLayers()
```

7. Corrigir especialmente:
   - Comic Neue;
   - Bangers;
   - Komika;
   - Newrotic;
   - CCDave.

## Parte B — Importação manual de fontes

Implementar opção:

```txt
Adicionar fonte...
```

O usuário escolhe arquivo:

```txt
.ttf
.otf
```

O app deve:

1. copiar a fonte para o projeto;
2. registrar no font registry do projeto;
3. disponibilizar no FontSelect;
4. fazer Konva usar a fonte;
5. fazer Python usar o arquivo copiado;
6. persistir no `project.json`.

Evite depender inicialmente de `queryLocalFonts()` como fluxo principal. Ela pode ser adicionada depois, mas o fluxo confiável deve ser importação manual.

## Parte C — Fontes do sistema

Se implementar fontes do sistema, faça com cuidado:

- não copie automaticamente fontes do Windows sem ação explícita do usuário;
- trate `.ttf`, `.otf` e `.ttc`;
- se não conseguir usar `.ttc`, mostre mensagem clara;
- evite quebrar o pipeline Python.

## Testes e debug

Criar tela/rotina de diagnóstico:

```txt
debug_runs/editor_refactor/phase_2_fonts/
```

Testar:

- Comic Neue Regular;
- Comic Neue Bold;
- Bangers;
- Komika;
- uma fonte importada manualmente;
- mudança de fonte no canvas;
- mudança de fonte no render fiel;
- build Tauri de produção.

Comandos:

```bash
npm run typecheck
npm run lint
npm run tauri dev
npm run tauri build
```

## Critério de pronto

- Comic Neue aparece corretamente no preview em tempo real.
- Fonte selecionada no editor bate com o render/export.
- Fonte importada manualmente funciona no Konva e no Python.
- Não há fallback invisível para fonte do sistema sem aviso.

---

# Fase 3 — Auto-save completo a cada 3 segundos

## Objetivo

Eliminar necessidade de salvar manualmente sem perder segurança.

## Estado necessário

Adicionar ao store:

```ts
dirty: boolean
lastSavedAt: number | null
autoSaveStatus: 'idle' | 'pending' | 'saving' | 'saved' | 'error'
saveVersion: number
saveInFlightVersion: number | null
lastSaveError: string | null
```

## Funções

Implementar:

```ts
markDirty()
runAutoSave()
flushAutoSave()
pauseAutoSave()
resumeAutoSave()
```

## Regras

1. Qualquer alteração chama `markDirty()`:
   - texto;
   - tradução;
   - estilo;
   - fonte;
   - posição;
   - brush;
   - mask;
   - eraser;
   - opacidade;
   - lock;
   - visibilidade;
   - ordem de camadas.

2. Timer a cada 3 segundos:
   - se `dirty=true`;
   - se não houver pipeline rodando;
   - se não houver save em andamento;
   - salva.

3. Flush obrigatório:
   - troca de página;
   - unmount;
   - `beforeunload`;
   - `pagehide`;
   - `visibilitychange`;
   - Tauri `close-requested`;
   - antes de Detectar/OCR/Traduzir/Inpaint;
   - antes de render fiel manual.

4. Se edição acontecer durante save:
   - não marcar como salvo incorretamente;
   - manter dirty para próximo ciclo.

## UI

Criar:

```ts
AutoSaveIndicator.tsx
```

Estados:

```txt
Salvando...
Salvo agora
Salvo há 12s
Alterações pendentes
Erro ao salvar
```

Se erro, permitir retry.

## Testes e debug

Criar logs:

```txt
debug_runs/editor_refactor/phase_3_autosave/
```

Testar:

1. editar texto e esperar 3s;
2. editar continuamente por 10s;
3. trocar de página imediatamente depois de editar;
4. fechar app depois de editar;
5. rodar OCR logo depois de editar;
6. simular erro de save;
7. verificar se não há flicker/reload desnecessário.

Comandos:

```bash
npm run typecheck
npm run lint
npm run tauri dev
```

## Critério de pronto

- Alterações persistem sem clicar em salvar.
- Não perde edição ao trocar de página.
- Não perde edição ao fechar.
- Não salva estado antigo por cima de estado novo.
- Indicador visual está correto.

---

# Fase 4 — Reorganização completa da UI

## Objetivo

Criar layout profissional.

## Topbar principal

```txt
[Original/Limpa/Camadas] [Detectar] [OCR] [Traduzir] [Inpaint]       [Zoom]
```

## Segunda barra quando texto selecionado

Criar:

```ts
TypesettingBar.tsx
```

Com:

```txt
Fonte
Tamanho
Cor
Alinhamento
Bold
Italic
Contorno
Sombra
Brilho
Espaçamento
Rotação
Opacidade do texto
```

## Toolbar esquerda

Criar:

```ts
ToolSidebar.tsx
```

Ferramentas:

```txt
Selecionar
Mover/Pan
Texto
Brush
Borracha
Máscara/Laço
Mão
Zoom
Conta-gotas, se fizer sentido
```

Atalhos:

```txt
V - Selecionar
H - Pan
T - Texto
B - Brush
E - Borracha
L - Lasso/Mask
+ / - - Zoom
Esc - cancelar/fechar
```

## Painel direito

Deixar apenas:

```txt
Camadas/Bitmap
Texto
Histórico, se Undo/Redo for implementado visualmente
```

Remover completamente a aba Propriedades antiga da experiência principal.

## Testes e debug

Verificar:

- seleção de texto mostra TypesettingBar;
- seleção de bitmap não mostra controles de texto;
- toolbar esquerda muda ferramenta corretamente;
- atalhos funcionam;
- layout não quebra em resoluções menores;
- scroll/pan continuam funcionando;
- não há sobreposição de painel com canvas.

Comandos:

```bash
npm run typecheck
npm run lint
npm run tauri dev
```

## Critério de pronto

Interface reorganizada, com comportamento previsível e sem perda de função.

---

# Fase 5 — FloatingTextEditor

## Objetivo

Ao clicar em uma camada de texto, abrir editor flutuante ao lado do texto selecionado.

## Componente

Criar:

```ts
src/components/editor/stage/FloatingTextEditor.tsx
```

## Conteúdo

```txt
Original
Tradução
Confiança OCR
Tipo de bloco, se existir
Botão Restaurar original
```

## Regras

1. Aparece ao lado do texto selecionado.
2. Acompanha zoom.
3. Acompanha pan.
4. Não sai da tela.
5. Não cobre desnecessariamente o balão.
6. ESC fecha.
7. Clique fora fecha.
8. Edição chama auto-save.
9. Usa `canvasGeometry.ts`.
10. Não bloqueia drag/zoom do canvas inteiro.

## Testes e debug

Testar:

- texto no canto superior;
- texto no canto inferior;
- texto na borda direita;
- texto na borda esquerda;
- zoom 50%;
- zoom 100%;
- zoom 200%;
- pan deslocado;
- edição rápida;
- ESC;
- click fora.

Comandos:

```bash
npm run typecheck
npm run lint
npm run tauri dev
```

## Critério de pronto

Editor flutuante sempre aparece em posição correta e edita sem quebrar canvas.

---

# Fase 6 — Preview e render automáticos

## Objetivo

Remover necessidade de botão Preview/Render.

## Estado

Adicionar ao store:

```ts
renderVersion: number
renderInFlightVersion: number | null
renderStatus: 'idle' | 'stale' | 'rendering' | 'updated' | 'error'
renderError: string | null
```

## Funções

```ts
markRenderStale()
scheduleAutoFidelityRender()
runAutoFidelityRender()
forceFidelityRender()
```

## Regras

1. Live Preview atualiza imediatamente no Konva.
2. Auto Fidelity Render dispara depois de 1.5s sem edição.
3. Se usuário editar durante render, resultado antigo é ignorado.
4. Render não bloqueia UI.
5. Render não duplica texto.
6. Render mostra status visual.
7. Ctrl+Shift+R força render manual/debug.

## UI

Criar:

```ts
RenderStatusBadge.tsx
```

Estados:

```txt
Fiel atualizado
Renderizando fiel...
Render desatualizado
Erro no render fiel
```

## Testes e debug

Criar:

```txt
debug_runs/editor_refactor/phase_6_auto_render/
```

Testar:

1. editar tradução;
2. mudar fonte;
3. mudar cor;
4. mudar tamanho;
5. editar rapidamente várias vezes;
6. mudar de página durante render;
7. forçar erro no Python;
8. verificar duplicação de texto;
9. comparar canvas com export/render final.

Comandos:

```bash
npm run typecheck
npm run lint
npm run tauri dev
cd pipeline && python -m pytest -q
```

## Critério de pronto

- Preview muda instantâneo.
- Render fiel atualiza sozinho.
- Não há texto duplicado.
- Resultado antigo não sobrescreve resultado novo.
- Erros são visíveis.

---

# Fase 7 — Brush estilo Photoshop completo

## Objetivo

Criar brush real para pintar pixels finais.

## Camada

Criar camada:

```txt
Paint
```

Ela deve ser separada de:

```txt
Original
Inpaint
Render
Mask
Texto
```

## Configurações do Brush

Implementar:

```txt
Cor
Tamanho
Opacidade
Dureza
Suavização
Swatches recentes
Preview circular
```

## Comportamento

1. Brush pinta apenas na camada Paint.
2. Brush respeita zoom/pan.
3. Brush usa coordenadas da imagem, não da tela.
4. Brush não altera Original.
5. Brush não altera Mask.
6. Brush não altera Render diretamente.
7. Brush persiste no projeto.
8. Brush aparece no export final.
9. Lock da camada Paint impede pintura.
10. Opacidade da camada Paint afeta visualização/export.

## Engine do brush

Implementar traço contínuo com suavização.

Se usar interpolação:

- evitar buracos em movimentos rápidos;
- evitar serrilhado grosseiro;
- respeitar opacidade.

## Testes e debug

Criar:

```txt
debug_runs/editor_refactor/phase_7_brush/
```

Testar:

- pintar preto;
- pintar vermelho;
- pintar com opacidade 50%;
- pintar em zoom 50%;
- pintar em zoom 200%;
- pintar depois de pan;
- salvar/reabrir;
- exportar/ver render final;
- lock da camada Paint;
- undo se já implementado.

Comandos:

```bash
npm run typecheck
npm run lint
npm run tauri dev
cd pipeline && python -m pytest -q
```

## Critério de pronto

Brush se comporta como ferramenta real de pintura e não interfere em Mask/Inpaint.

---

# Fase 8 — Máscara como Laço estilo Photoshop

## Objetivo

Transformar máscara em seleção para inpaint.

## Modos

```txt
Laço livre
Laço poligonal
```

## Operações

Implementar:

```txt
Adicionar à máscara
Subtrair da máscara
Substituir máscara
Limpar máscara
```

Atalhos desejados:

```txt
Shift = adicionar
Alt = subtrair
Esc = cancelar
Enter = fechar poligonal
```

## Visual

Máscara deve aparecer como overlay roxo translúcido.

```txt
#6C5CE7
alpha ~40%
```

## Regras

1. Máscara não aparece no export.
2. Máscara serve apenas para Inpaint.
3. Máscara usa coordenadas da imagem.
4. Máscara pode ser apagada pela borracha.
5. Máscara pode ser limpa.
6. Inpaint deve afetar somente região mascarada.
7. Máscara residual não deve afetar ações futuras sem o usuário perceber.

## Rasterização

Ao fechar o laço:

1. converter pontos para coordenadas da imagem;
2. desenhar em canvas offscreen;
3. preencher polígono;
4. gerar diff/bitmap;
5. enviar ao backend;
6. atualizar overlay.

## Testes e debug

Criar:

```txt
debug_runs/editor_refactor/phase_8_mask_lasso/
```

Testar:

- laço livre simples;
- laço livre com zoom;
- laço livre com pan;
- laço poligonal;
- Enter fecha;
- Esc cancela;
- clicar no ponto inicial fecha;
- adicionar máscara;
- subtrair máscara;
- substituir máscara;
- limpar máscara;
- rodar Inpaint com máscara;
- verificar se região correta foi alterada.

Comandos:

```bash
npm run typecheck
npm run lint
npm run tauri dev
cd pipeline && python -m pytest -q
```

## Critério de pronto

Máscara funciona como ferramenta de seleção profissional e o Inpaint respeita a região correta.

---

# Fase 9 — Borracha inteligente

## Objetivo

Borracha deve apagar Paint ou Mask sem afetar outras camadas.

## Target

A borracha deve escolher alvo assim:

1. Se camada ativa for Paint, apaga Paint.
2. Se camada ativa for Mask, apaga Mask.
3. Se última camada editada foi Paint, apaga Paint.
4. Se última camada editada foi Mask, apaga Mask.
5. Padrão: Paint.

## UI

Mostrar:

```txt
Apagando: Paint
Apagando: Mask
```

Permitir troca manual de alvo.

## Regras

A borracha nunca pode apagar:

```txt
Original
Inpaint
Render
Texto
```

## Testes e debug

Criar:

```txt
debug_runs/editor_refactor/phase_9_eraser/
```

Testar:

- pintar com Brush e apagar;
- criar Mask e apagar;
- alternar alvo manualmente;
- apagar com zoom;
- apagar com pan;
- lock da camada;
- salvar/reabrir;
- export final.

Comandos:

```bash
npm run typecheck
npm run lint
npm run tauri dev
cd pipeline && python -m pytest -q
```

## Critério de pronto

Borracha apaga exatamente o alvo correto, sem destruir dados importantes.

---

# Fase 10 — Bitmap como painel de camadas completo

## Objetivo

Transformar Bitmap em painel de camadas estilo Photoshop.

## Camadas

Exibir:

```txt
Texto
Paint
Render
Inpaint
Original
Mask
```

## Cada camada deve ter

```txt
Thumbnail
Nome amigável
Visibilidade
Lock
Opacidade
Selecionar camada
Indicador de camada ativa
```

## Reorder

Implementar reorder com proteção.

Regras:

1. Texto sempre no topo.
2. Mask é camada técnica e não exporta.
3. Original não deve cobrir Inpaint/Render de forma que confunda o usuário.
4. Se permitir reorder, validar composição e export.
5. Se uma ordem for inválida, bloquear e mostrar aviso.

## Persistência

Salvar no projeto:

```ts
visible
locked
opacity
order
```

Com migration para projetos antigos.

## Export/compositor

O export final deve respeitar:

- Paint;
- opacidade;
- visibilidade;
- render final;
- texto;
- ordem válida.

Mask deve ser ignorada no export final.

## Testes e debug

Criar:

```txt
debug_runs/editor_refactor/phase_10_layers/
```

Testar:

- ocultar Original;
- ocultar Paint;
- ocultar Texto;
- opacidade 50%;
- lock;
- reorder válido;
- reorder inválido;
- salvar/reabrir;
- exportar;
- comparar canvas vs export.

Comandos:

```bash
npm run typecheck
npm run lint
npm run tauri dev
cd pipeline && python -m pytest -q
```

## Critério de pronto

Painel de camadas está funcional, previsível e não quebra export.

---

# Fase 11 — Undo/Redo obrigatório

## Objetivo

Como o editor terá auto-save, precisa ter Undo/Redo para evitar perda por erro do usuário.

## Atalhos

```txt
Ctrl+Z = Undo
Ctrl+Y = Redo
Ctrl+Shift+Z = Redo alternativo
```

## Deve cobrir

```txt
Edição de texto
Edição de tradução
Estilo de texto
Mudança de fonte
Mudança de posição/bbox
Brush stroke
Mask stroke
Eraser stroke
Opacidade de camada
Visibilidade de camada
Lock de camada
Reorder de camada
```

## Modelo

Criar histórico com ações reversíveis.

Cada ação deve ter:

```ts
do()
undo()
redo()
description
timestamp
affectedLayerId
```

Para bitmap, salvar diff antes/depois ou snapshot otimizado por stroke.

## UI

Adicionar opção visual simples:

```txt
Undo
Redo
Histórico recente, se viável
```

## Testes e debug

Criar:

```txt
debug_runs/editor_refactor/phase_11_undo_redo/
```

Testar:

- editar texto e desfazer;
- mudar fonte e desfazer;
- pintar e desfazer;
- apagar e desfazer;
- criar máscara e desfazer;
- opacidade e desfazer;
- salvar automático após undo;
- reabrir projeto após autosave.

Comandos:

```bash
npm run typecheck
npm run lint
npm run tauri dev
cd pipeline && python -m pytest -q
```

## Critério de pronto

Auto-save não torna erros irreversíveis. O usuário consegue desfazer ações importantes.

---

# Fase 12 — Teste final end-to-end completo

## Objetivo

Validar o editor inteiro como produto.

## Criar cenário de teste

Usar um capítulo/página teste com:

- imagem original;
- balão com texto;
- fonte Comic Neue;
- fonte importada manualmente;
- inpaint;
- brush;
- mask;
- tradução;
- export.

## Roteiro manual obrigatório

1. Abrir projeto.
2. Rodar Detectar.
3. Rodar OCR.
4. Rodar Traduzir.
5. Selecionar texto.
6. Editar tradução pelo FloatingTextEditor.
7. Mudar fonte para Comic Neue.
8. Mudar fonte para uma fonte importada.
9. Mudar cor/tamanho/contorno.
10. Esperar auto-save.
11. Trocar de página e voltar.
12. Confirmar persistência.
13. Pintar com Brush.
14. Apagar parte com Eraser.
15. Criar máscara com laço livre.
16. Criar máscara com laço poligonal.
17. Subtrair parte da máscara.
18. Rodar Inpaint.
19. Ajustar camadas.
20. Testar opacidade.
21. Testar lock.
22. Testar undo/redo.
23. Esperar Auto Fidelity Render.
24. Confirmar que não há texto duplicado.
25. Exportar/renderizar resultado final.
26. Comparar canvas vs export.
27. Fechar app.
28. Reabrir app.
29. Confirmar que tudo persistiu.

## Comandos finais

Rodar todos os comandos disponíveis:

```bash
npm run typecheck
npm run lint
npm test
npm run tauri dev
npm run tauri build
cd pipeline && python -m pytest -q
```

Se algum comando não existir, registrar no relatório final.

## Relatório final

Gerar:

```txt
debug_runs/editor_refactor/final_report.md
```

Com:

```md
# Relatório Final — Refactor Editor TraduzAi

## Fases concluídas

## Arquivos alterados

## Testes executados

## Resultado dos testes

## Bugs encontrados

## Bugs corrigidos

## Pendências reais

## Como usar as novas funções

## Como debugar se algo falhar

## Riscos restantes
```

## Critério final de pronto

A implementação só está concluída quando:

- zoom está no topo;
- nomes `001.png`/`001.jpg` sumiram;
- aba Propriedades foi removida;
- auto-save funciona;
- preview é instantâneo;
- render fiel é automático;
- fontes funcionam no canvas e no export;
- Detect/OCR/Traduzir/Inpaint funcionam ou mostram erro claro;
- Brush pinta com cor/tamanho/opacidade/dureza;
- Mask funciona como laço;
- Eraser apaga Paint/Mask corretamente;
- Bitmap funciona como painel de camadas;
- Undo/Redo funciona;
- export final bate com o canvas;
- não há texto duplicado;
- não há falha silenciosa;
- build final passa.

Comece agora pela leitura das skills disponíveis, depois faça diagnóstico do código atual, depois implemente fase por fase.
