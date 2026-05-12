# Plano Revisado — Refactor Profissional do Editor TraduzAi

## Objetivo

Transformar o editor atual do TraduzAi em uma interface mais próxima de um editor profissional, usando Photoshop como referência visual e de interação, mas sem deixar o app pesado ou instável.

As mudanças principais são:

- Mover zoom do canto inferior direito para o topo.
- Remover nomes como `001.png`, `001.jpg` da aba Bitmap.
- Remover aba Propriedades.
- Colocar edição de texto Original/Tradução próxima ao texto selecionado.
- Auto-save a cada 3 segundos.
- Remover botões de ações manuais.
- Remover botões Preview Fiel e Render.
- Fazer preview/render atualizarem automaticamente.
- Corrigir carregamento de fontes, principalmente Comic Neue.
- Verificar Detectar, OCR, Traduzir e Inpaint.
- Transformar Brush em pincel estilo Photoshop.
- Transformar Máscara em ferramenta tipo laço.
- Fazer Borracha apagar o que foi pintado pelo Brush.
- Fazer Bitmap funcionar como painel de camadas.

---

# Revisão crítica do plano atual

## 1. Auto-save com debounce puro está errado

O plano anterior sugeria debounce de 3000ms. Isso pode ser um problema.

Se o usuário digitar continuamente, o debounce reinicia sempre e talvez nunca salve até ele parar. O pedido foi “salvar as alterações a cada 3 segundos”, então o correto é:

- salvar no máximo a cada 3 segundos enquanto houver alterações;
- também salvar quando trocar de página;
- salvar ao fechar/sair do editor;
- salvar antes de rodar Detectar/OCR/Traduzir/Inpaint;
- não salvar no meio de uma ação pesada.

### Correção

Usar modelo híbrido:

- `dirty = true` quando algo muda;
- `autoSaveTimer` roda a cada 3s se houver dirty;
- `flushAutoSave()` em troca de página, unmount e antes de pipeline action;
- controle por versão para evitar salvar coisa antiga.

---

## 2. Render real-time precisa ser dividido em dois níveis

Não dá para chamar o render Python completo a cada pequena alteração. Isso deixaria o editor travado.

A solução correta é:

### Preview imediato

Feito no próprio canvas/Konva.

- Texto muda instantaneamente.
- Fonte, cor, tamanho, contorno e posição aparecem na hora.
- Esse é o preview visual rápido.

### Render fiel automático

Feito em background com debounce.

- Espera o usuário parar de mexer por 1.2s a 1.8s.
- Chama o render Python.
- Atualiza o bitmap renderizado por baixo.
- Se o usuário mexer de novo antes de terminar, o resultado antigo é descartado.

### Correção

Não chamar isso simplesmente de “render real-time”. Nome melhor:

- `Live Preview`: Konva imediato.
- `Auto Fidelity Render`: render Python automático com debounce.

---

## 3. Remover botão Preview/Render é ok, mas precisa manter fallback escondido

O usuário pediu remover os botões. Certo.

Mas tecnicamente é perigoso apagar completamente a função manual logo de cara.

### Correção

Remover da UI principal, mas manter:

- atalho `Ctrl + Shift + R`;
- opção escondida em menu de debug/dev;
- função interna ainda disponível.

Assim, se o auto-render falhar, ainda dá para testar sem recriar código.

---

## 4. Problema das fontes não é só `@font-face`

O plano acerta ao apontar `@font-face`, mas falta mais coisa.

No Konva, depois que a fonte carrega, às vezes o texto já foi desenhado com fallback. Então precisa:

- carregar a fonte antes de desenhar;
- chamar `layer.batchDraw()` depois do carregamento;
- garantir que o nome usado no CSS é exatamente o mesmo usado no `fontFamily`;
- garantir que Comic Neue Regular/Bold/Italic tenham nomes corretos;
- garantir que Tauri consiga servir os arquivos em produção, não só no dev.

### Correção

Criar sistema centralizado de fontes:

```ts
FONT_REGISTRY = {
  comicNeue: {
    cssFamily: "Comic Neue",
    files: {
      regular: "/fonts/ComicNeue-Regular.ttf",
      bold: "/fonts/ComicNeue-Bold.ttf",
      italic: "/fonts/ComicNeue-Italic.ttf",
      boldItalic: "/fonts/ComicNeue-BoldItalic.ttf"
    }
  }
}
```

E no editor:

- `preloadEditorFonts()`;
- `await document.fonts.ready`;
- `forceRedrawTextLayers()`.

---

## 5. Brush, Mask e Eraser precisam virar ferramentas separadas de verdade

O plano atual mistura conceitos.

Hoje parece existir:

- brush azul;
- mask;
- eraser;
- overlay bitmap.

Mas para ficar profissional, precisa separar:

### Brush

Serve para pintar pixels finais.

- cor selecionável;
- opacidade;
- tamanho;
- dureza;
- preview circular do pincel;
- pinta numa camada `paint`.

### Mask

Serve para selecionar região para inpaint.

- não é pintura final;
- não deve aparecer no export;
- pode ser roxa/transparente;
- age como seleção/laço.

### Eraser

Deve apagar somente:

- pintura do Brush;
- ou máscara, se a máscara estiver selecionada.

Não deve apagar imagem original, render ou texto.

---

## 6. Máscara tipo laço precisa de conversão correta de coordenadas

Esse é um ponto crítico.

O usuário desenha no canvas, mas o bitmap real está em coordenadas da imagem.

Então a ferramenta laço precisa converter:

```txt
screen position -> stage position -> image position -> bitmap mask position
```

Se isso não for feito, a máscara vai aparecer em um lugar e o inpaint vai rodar em outro.

### Correção

Criar função única:

```ts
screenToImagePoint(pointer, zoom, panOffset, imageScale)
```

E usar essa função para:

- brush;
- eraser;
- mask freehand;
- mask polygonal;
- seleção de texto;
- hit testing.

---

## 7. Aba Bitmap como Layers do Photoshop precisa de escopo MVP

Boa ideia, mas o plano pode ficar grande demais.

O MVP correto é:

- visibilidade;
- lock;
- opacidade;
- ordem;
- thumbnail;
- seleção de camada ativa.

Não colocar ainda:

- blend modes;
- grupos;
- máscaras por camada;
- clipping mask;
- filtros;
- ajustes.

Isso evita transformar o refactor em um projeto gigante.

---

## 8. Verificar Detect/OCR/Traduzir/Inpaint deve vir antes de mexer pesado

O plano anterior colocou diagnóstico no fim. O ideal é mudar isso.

Antes de refatorar a UI, precisa confirmar se os botões atuais funcionam.

Se não funcionam agora, depois do refactor vai ficar difícil saber se quebrou antes ou depois.

### Correção

Fazer uma fase inicial de diagnóstico.

---

# Plano final recomendado

## Fase 0 — Diagnóstico obrigatório antes do refactor

### Objetivo

Garantir que o editor atual está funcional antes de alterar a UI.

### Tarefas

1. Testar botão Detectar.
2. Testar botão OCR.
3. Testar botão Traduzir.
4. Testar botão Inpaint.
5. Registrar logs no console.
6. Registrar logs no backend Rust.
7. Registrar logs no pipeline Python.
8. Verificar se cada ação retorna assets esperados.
9. Confirmar se máscara antiga interfere no inpaint.
10. Confirmar se fontes quebram apenas no preview Konva ou também no render Python.

### Critério de pronto

- Cada ação deve mostrar início, progresso, sucesso ou erro claro.
- Nenhum botão pode falhar silenciosamente.
- Se uma etapa falhar, a UI deve mostrar mensagem de erro amigável.

---

## Fase 1 — Limpeza visual imediata

### 1.1 Mover Zoom

Mover controles de zoom do canto inferior direito para o canto superior direito, logo abaixo/ao lado das abas de ações:

- Detectar;
- OCR;
- Traduzir;
- Inpaint.

O zoom deve ficar compacto:

```txt
[-] 100% [+] [Fit]
```

### 1.2 Remover nomes de arquivo do Bitmap

Na aba Bitmap, remover textos como:

```txt
001.png
001.jpg
```

Exibir apenas nomes amigáveis:

```txt
Original
Inpaint
Paint
Render
Mask
```

### 1.3 Remover ações manuais

Remover da interface:

- ações manuais;
- botões soltos no canto inferior direito;
- botões duplicados;
- comandos que confundem o fluxo principal.

### 1.4 Remover botões Preview Fiel e Render da UI principal

Eles não devem aparecer como botões principais.

Mas manter internamente:

- função de render;
- função de preview fiel;
- atalho de debug.

---

## Fase 2 — Corrigir fontes no preview em tempo real

### Objetivo

A fonte selecionada deve aparecer corretamente no canvas, principalmente Comic Neue.

### Tarefas

1. Mover fontes para pasta servível pelo app.
2. Criar `@font-face` para todas as fontes.
3. Criar `FONT_REGISTRY`.
4. Garantir nomes canônicos.
5. Pré-carregar fontes ao abrir o editor.
6. Esperar `document.fonts.ready`.
7. Forçar redraw das camadas Konva.
8. Testar Regular/Bold/Italic.
9. Testar Comic Neue especificamente.
10. Testar no modo dev e no build Tauri.

### Critério de pronto

Ao selecionar Comic Neue, Bangers, Komika ou outra fonte, o texto no canvas deve mudar visualmente na hora.

---

## Fase 3 — Auto-save correto a cada 3 segundos

### Objetivo

O usuário não deve precisar clicar em salvar.

### Comportamento esperado

- Qualquer alteração marca o projeto como sujo.
- A cada 3 segundos, se houver alterações, salva automaticamente.
- Ao trocar de página, salva antes.
- Ao fechar o editor, tenta salvar.
- Antes de Detectar/OCR/Traduzir/Inpaint, salva.
- Durante ações pesadas, o auto-save pausa.
- Depois da ação, volta.

### Estados visuais

Mostrar pequeno indicador:

```txt
Salvando...
Salvo agora
Salvo há 12s
Erro ao salvar
Alterações pendentes
```

### Importante

Não usar apenas debounce. Usar throttle/interval com `dirty flag`.

---

## Fase 4 — Reorganização da interface superior e lateral

### Objetivo

Deixar o editor parecido com ferramenta profissional.

### Topo

A parte superior deve conter:

```txt
[Detectar] [OCR] [Traduzir] [Inpaint]        [Zoom - 100% + Fit]
```

Quando texto estiver selecionado, mostrar uma segunda barra:

```txt
Fonte | Tamanho | Cor | Alinhamento | Bold | Italic | Contorno | Sombra
```

### Esquerda

A lateral esquerda deve virar barra de ferramentas:

```txt
Selecionar
Mover
Texto
Brush
Borracha
Máscara/Lasso
Mão/Pan
```

### Direita

A lateral direita deve ter apenas:

```txt
Bitmap / Camadas
Texto
```

Remover:

```txt
Propriedades
```

---

## Fase 5 — Editor flutuante de texto

### Objetivo

Ao selecionar um texto no canvas, abrir uma aba perto do texto selecionado com:

```txt
Original
Tradução
```

### Comportamento

- A aba aparece ao lado do texto selecionado.
- Não deve cobrir completamente o balão.
- Deve acompanhar zoom e pan.
- Deve ter limite de tela para não sair da viewport.
- Campo Original pode ser editável ou readonly, dependendo do modo.
- Campo Tradução deve ser editável.
- Alterações entram no auto-save.
- ESC fecha.
- Clicar fora fecha.

### Critério de pronto

Ao clicar em uma camada de texto, o usuário edita a tradução diretamente perto do balão, sem precisar ir até a aba Propriedades.

---

## Fase 6 — Preview e render automáticos

### Objetivo

O usuário vê o resultado sem clicar em Preview ou Render.

### Modelo correto

## 6.1 Live Preview

Atualização instantânea via Konva:

- texto;
- posição;
- tamanho;
- fonte;
- cor;
- contorno;
- sombra.

## 6.2 Auto Fidelity Render

Render Python automático em background:

- aguardar 1.5s após última alteração;
- cancelar resultado antigo se o usuário editar de novo;
- atualizar cache da página;
- não bloquear a interface;
- mostrar indicador discreto.

### Estados visuais

```txt
Preview atualizado
Renderizando fiel...
Render fiel atualizado
Render fiel atrasado
Erro no render fiel
```

### Critério de pronto

O usuário edita e vê resultado imediato. Depois de alguns segundos, o render fiel atualiza sozinho.

---

## Fase 7 — Brush estilo Photoshop

### Objetivo

Transformar o brush em uma ferramenta real de pintura.

### Comportamento

Ao clicar no Brush, abrir opções:

```txt
Cor
Tamanho
Opacidade
Dureza
```

### Regras

- Brush pinta em camada `Paint`.
- Brush não altera `Original`.
- Brush não altera `Mask`.
- Brush não altera `Render`.
- Cor padrão: preto.
- Deve ter preview circular do pincel.
- Deve respeitar zoom.
- Deve pintar suave, sem serrilhado forte.
- Deve permitir traço contínuo.

### Critério de pronto

O usuário escolhe uma cor e pinta como em um editor de imagem básico.

---

## Fase 8 — Máscara como laço do Photoshop

### Objetivo

Transformar máscara em ferramenta de seleção para inpaint.

### Modos

```txt
Laço livre
Laço poligonal
```

### Laço livre

- Clica e segura.
- Desenha a área.
- Solta o mouse.
- A seleção fecha automaticamente.

### Laço poligonal

- Cada clique adiciona um ponto.
- Enter fecha.
- ESC cancela.
- Clicar no primeiro ponto fecha.

### Regras

- Máscara não aparece no export final.
- Máscara serve apenas para inpaint.
- Máscara deve ser exibida como overlay roxo transparente.
- Máscara precisa ser rasterizada para bitmap antes de enviar ao backend.
- Coordenadas precisam bater com a imagem real.

### Critério de pronto

O usuário seleciona uma área com laço, roda Inpaint, e apenas aquela região é afetada.

---

## Fase 9 — Borracha inteligente

### Objetivo

A borracha deve apagar o que o usuário pintou com Brush ou Máscara.

### Comportamento

Se a camada ativa for:

```txt
Paint -> apaga pintura
Mask -> apaga máscara
```

Se nenhuma estiver selecionada:

```txt
apaga última camada editada
```

Se não houver última camada:

```txt
apaga Paint por padrão
```

### Importante

A borracha não deve apagar:

- imagem original;
- inpaint;
- texto;
- render final.

### Critério de pronto

Pintou com Brush, a borracha apaga a pintura. Criou máscara, a borracha apaga a máscara.

---

## Fase 10 — Bitmap como painel de camadas

### Objetivo

Transformar Bitmap em algo parecido com Layers do Photoshop.

### Camadas iniciais

```txt
Texto
Paint
Render
Inpaint
Original
Mask
```

### Cada camada deve ter

- thumbnail;
- nome amigável;
- olho de visibilidade;
- lock;
- opacidade;
- seleção ativa.

### Reorder

Permitir arrastar camadas para mudar ordem, mas com restrições.

Sugestão inicial:

- Texto sempre acima.
- Paint acima do Render.
- Mask separada ou marcada como camada técnica.
- Original geralmente no fundo.

### Melhor abordagem

Não permitir ordem totalmente livre no primeiro MVP. Usar ordem semi-controlada para evitar quebrar export.

Exemplo:

```txt
Texto
Paint
Render
Inpaint
Original
Mask técnica
```

Depois, numa versão futura, liberar reorder total.

### Critério de pronto

O usuário entende visualmente quais camadas existem e consegue ocultar, bloquear e ajustar opacidade.

---

# Ordem correta de implementação

```txt
0. Diagnóstico Detect/OCR/Traduzir/Inpaint
1. Limpeza visual básica
2. Correção das fontes
3. Auto-save
4. Reorganização da UI
5. Editor flutuante de texto
6. Preview/render automático
7. Brush estilo Photoshop
8. Máscara tipo laço
9. Borracha inteligente
10. Bitmap como camadas
```

Essa ordem é mais segura porque primeiro estabiliza o que já existe, depois melhora a experiência, e só depois entra nas ferramentas mais complexas.

---

# Prompt pronto para mandar ao Claude/Codex

```md
Revise e implemente este plano como engenheiro senior de frontend/backend para o editor do TraduzAi.

Contexto:
O objetivo é refatorar o editor para ficar mais profissional, intuitivo e parecido com Photoshop, mas sem quebrar o pipeline atual de Detectar, OCR, Traduzir e Inpaint.

Antes de implementar qualquer mudança grande, faça uma análise do código atual e valide os pontos abaixo.

## Fase 0 — Diagnóstico obrigatório

Antes de refatorar a UI, verifique se os botões atuais estão funcionando:

- Detectar
- OCR
- Traduzir
- Inpaint

Adicione logs claros no frontend, Rust/Tauri e Python. Nenhuma ação pode falhar silenciosamente. Cada ação deve mostrar início, progresso, sucesso ou erro.

Também verifique:
- se máscara antiga/residual está interferindo no inpaint;
- se selectedImageLayerKey está correto;
- se fontes locais estão quebrando apenas no Konva ou também no render Python;
- se o pipeline retorna changed_assets corretos.

## Fase 1 — Limpeza visual básica

1. Mova os controles de zoom do canto inferior direito para o canto superior direito, logo abaixo ou ao lado da barra de ações Detectar/OCR/Traduzir/Inpaint.
2. Remova nomes como 001.png/001.jpg da aba Bitmap.
3. Remova ações manuais do canto inferior direito.
4. Remova os botões Preview Fiel e Render da interface principal.
5. Mantenha as funções internas de preview/render disponíveis via atalho ou modo debug, mas não como botões principais.

## Fase 2 — Corrigir fontes

Corrija o carregamento das fontes locais no preview em tempo real.

Requisitos:
- criar @font-face para todas as fontes;
- mover fontes para pasta servível pelo Tauri/WebView;
- criar um FONT_REGISTRY central;
- garantir que o nome do fontFamily usado pelo Konva seja exatamente igual ao CSS;
- usar document.fonts.load/document.fonts.ready;
- forçar redraw das camadas Konva depois que a fonte carregar;
- testar Comic Neue especificamente.

Critério:
A fonte Comic Neue e as demais fontes devem aparecer corretamente no canvas em tempo real.

## Fase 3 — Auto-save a cada 3 segundos

Implementar auto-save real.

Não use debounce puro. Use dirty flag + timer/throttle.

Comportamento:
- qualquer alteração marca dirty=true;
- a cada 3 segundos, se dirty=true, salvar;
- flush obrigatório ao trocar de página;
- flush obrigatório ao sair/desmontar editor;
- flush obrigatório antes de Detectar/OCR/Traduzir/Inpaint;
- pausar auto-save durante ações pesadas;
- retomar depois;
- mostrar indicador visual: Salvando, Salvo agora, Salvo há Xs, Erro ao salvar.

## Fase 4 — Reorganização da UI

Topo:
- Detectar
- OCR
- Traduzir
- Inpaint
- Zoom

Quando texto estiver selecionado:
- mostrar barra horizontal de typesetting com Fonte, Tamanho, Cor, Alinhamento, Bold, Italic, Contorno e Sombra.

Esquerda:
- Selecionar
- Mover
- Texto
- Brush
- Borracha
- Máscara/Laço
- Pan

Direita:
- Bitmap/Camadas
- Texto

Remover aba Propriedades.

## Fase 5 — Editor flutuante de texto

Ao selecionar uma camada de texto, abrir um painel flutuante próximo ao texto selecionado com:

- Original
- Tradução

Requisitos:
- deve abrir ao lado do texto selecionado;
- deve acompanhar zoom e pan;
- deve fazer clamp para não sair da tela;
- ESC fecha;
- clicar fora fecha;
- edição entra no auto-save;
- não deve bloquear o canvas inteiro com pointer-events.

## Fase 6 — Preview/render automático

Separar dois conceitos:

1. Live Preview:
- imediato;
- feito no Konva;
- atualiza texto, fonte, cor, tamanho, contorno, sombra e posição.

2. Auto Fidelity Render:
- feito pelo render Python;
- dispara em background após 1.5s sem edição;
- cancela/ignora resultado antigo se a página ou versão mudou;
- não bloqueia a UI;
- atualiza cache renderizado.

Remover botões manuais da UI principal.

## Fase 7 — Brush estilo Photoshop

Transformar o Brush em ferramenta de pintura real.

Requisitos:
- criar camada Paint;
- Brush pinta apenas na camada Paint;
- opção de cor ao selecionar Brush;
- tamanho;
- opacidade;
- dureza;
- preview circular do pincel;
- traço suave;
- respeitar zoom/pan;
- não alterar Original, Mask ou Render diretamente.

## Fase 8 — Máscara como ferramenta laço

Transformar a máscara em ferramenta de seleção para inpaint.

Modos:
- Laço livre
- Laço poligonal

Laço livre:
- mouse down inicia;
- arrastar desenha;
- mouse up fecha seleção.

Laço poligonal:
- clique adiciona ponto;
- Enter fecha;
- ESC cancela;
- clicar no ponto inicial fecha.

A máscara deve:
- aparecer como overlay roxo transparente;
- não aparecer no export final;
- ser rasterizada para bitmap;
- usar conversão correta screen -> stage -> image -> bitmap.

## Fase 9 — Borracha inteligente

A borracha deve apagar:

- Paint, se Paint estiver ativo;
- Mask, se Mask estiver ativa;
- a última camada editada, se nada estiver ativo;
- Paint por padrão.

Nunca deve apagar:
- Original;
- Inpaint;
- Render;
- Texto.

## Fase 10 — Bitmap como painel de camadas

Transformar Bitmap em painel tipo Layers.

MVP:
- thumbnail;
- nome amigável;
- visibilidade;
- lock;
- opacidade;
- camada ativa.

Ordem recomendada inicialmente:
- Texto
- Paint
- Render
- Inpaint
- Original
- Mask técnica

Não liberar reorder totalmente no primeiro MVP se isso quebrar o pipeline. Se implementar reorder, proteja camadas técnicas.

## Critérios finais

Ao terminar:

1. Zoom aparece no topo.
2. Não aparecem mais 001.png/001.jpg no Bitmap.
3. Aba Propriedades foi removida.
4. Edição Original/Tradução abre ao lado do texto selecionado.
5. Auto-save funciona a cada 3 segundos.
6. Preview e render atualizam sem botão.
7. Fontes aparecem corretamente no canvas, especialmente Comic Neue.
8. Detectar/OCR/Traduzir/Inpaint funcionam com logs claros.
9. Brush pinta com cor selecionável.
10. Máscara funciona como laço.
11. Borracha apaga Paint ou Mask corretamente.
12. Bitmap parece e funciona como painel básico de camadas.
13. Nenhuma ação pesada bloqueia a UI.
14. O app não perde alterações do usuário.
```

---

# Avaliação final

O plano anterior estava bom como rascunho, mas esta versão é mais segura para implementação.

O ponto mais importante é não tratar “render em tempo real” como render Python constante. O correto é:

```txt
Konva instantâneo + render fiel automático em background
```

Isso dá sensação de tempo real sem travar o TraduzAi.
