# No-Reflow Connected Balloons Design

**Contexto**

O renderer atual recalcula quebras de linha com base na largura do balao. Isso altera a composicao original do OCR e introduz reflow artificial em falas e baloes conectados.

**Objetivo**

Parar de quebrar texto automaticamente no renderer. O renderer deve:

- Preservar apenas linhas explicitas vindas do OCR ou de agrupamentos previos.
- Ajustar somente alinhamento, tamanho de fonte e posicao.
- Continuar suportando baloes conectados, mas usando heuristicas geometricas para decidir quando separar o texto em dois blocos.

**Abordagem**

1. Trocar o reflow por um modo `preserve_lines`, em que o texto vira uma unica linha ou respeita apenas `\n`.
2. Inferir alinhamento automatico (`left`, `center`, `right`) pela posicao do bbox original dentro do balao ou subregiao.
3. Para baloes conectados com dois lobos, reforcar as heuristicas:
- bloco abaixo e a direita sugere balao conectado diagonal;
- distancia horizontal/vertical relevante entre grupos sugere separacao em dois blocos;
- diferenca de altura entre grupos reforca split em dois blocos.
4. Manter o ajuste de fonte e o reposicionamento dentro do `position_bbox`, sem voltar a quebrar por largura.

**Impacto esperado**

- Textos longos vao reduzir mais a fonte em vez de criar linhas novas.
- Grupos com `\n` explicito continuam em multiplas linhas.
- Baloes conectados continuam podendo renderizar em dois lobos, mas sem reflow semantico agressivo.
