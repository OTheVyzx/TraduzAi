# Connected Balloon Composer Design

**Date:** 2026-04-16

**Problem**

Baloes duplos conectados ainda ficam abaixo do nivel de scan profissional por tres motivos centrais:

- o pipeline decide subregioes boas o bastante para "cabem dois textos", mas nao para uma composicao bonita;
- quando o OCR ja separa um texto por lobo, cada metade e renderizada como balao comum, perdendo a logica especial de lobo conectado;
- o renderer atual ainda privilegia "maior fonte que cabe" em vez de "melhor composicao visual total".

O resultado aparece nos casos reais enviados pelo usuario: fonte pequena demais, espacamento ruim, linhas fracas, lobo subutilizado e quebra sem ritmo de leitura.

**Goal**

Introduzir um compositor dedicado para baloes conectados que trate o grupo inteiro como uma unica decisao tipografica, escolhendo a melhor composicao por score visual e mantendo fallback seguro para os casos onde a deteccao nao for confiavel.

**Scope desta fase**

- focar apenas em baloes duplos conectados;
- melhorar layout, composicao e renderizacao dos lobos;
- preservar fallback atual para nao regredir baloes simples;
- nao mexer ainda no nucleo de traducao ou inpaint, exceto quando algum helper precisar expor melhor contexto geometrico.

**Arquitetura proposta**

1. Layout produz plano conectado mais rico

- `pipeline/layout/balloon_layout.py` continua detectando `balloon_subregions`, mas passa a anexar metadados do grupo conectado:
- orientacao dominante (`left-right`, `top-bottom`, `diagonal`);
- confianca da deteccao;
- subregioes ordenadas para leitura;
- opcionalmente, bboxes-semente dos lobos detectados.

2. Typesetter promove o grupo conectado a entidade propria

- `pipeline/typesetter/renderer.py` deixa de quebrar o grupo em renderizacoes independentes quando a contagem `textos == subregions`;
- em vez disso, `build_render_blocks()` passa a gerar um bloco conectado unico contendo:
- `connected_children` quando o OCR ja separou os textos por lobo;
- `balloon_subregions` quando o texto precisa ser repartido semanticamente;
- estilo consolidado e metadados do grupo.

3. Compositor conectado gera candidatos

- para grupo `1:1`, o compositor recebe os children ja separados e testa apenas alternativas de sizing, occupancy e order;
- para grupo `N:M`, o compositor gera varios candidatos de split textual:
- por sentenca;
- por clausula;
- por balanceamento por palavras;
- por pesos de area do lobo;
- por pesos de leitura.

4. Dry-run tipografico com score visual

Cada candidato passa por resolve de layout sem desenhar na imagem final. O score combina:

- ocupacao de largura e altura por lobo;
- penalidade para fonte pequena;
- penalidade para linhas de uma palavra;
- penalidade para altura/quantidade de linhas desequilibrada;
- penalidade para grande diferenca de fonte entre lobos;
- bonus para quebras semanticas naturais;
- penalidade para texto colado na costura central;
- bonus para composicao parecida com referencia humana: densa, legivel e sem "ar morto".

5. Render final com fallback

- o melhor candidato acima do piso minimo e renderizado;
- se nenhum candidato passar do piso, o renderer cai para o comportamento atual de connected balloons;
- se nem isso for seguro, cai para balao simples.

**Mudancas principais de codigo**

- `pipeline/layout/balloon_layout.py`
- enriquecer subregioes com orientacao/ordem;
- expor plano conectado leve no texto enriquecido.

- `pipeline/typesetter/renderer.py`
- parar de quebrar grupos `1:1` em blocos independentes;
- criar caminho de composicao unica para connected balloons;
- permitir sizing quase uniforme, nao cegamente identico;
- selecionar composicao pelo score total.

- `pipeline/tests/test_layout_analysis.py`
- cobrir orientacao e ordenacao de subregioes.

- `pipeline/tests/test_typesetting_layout.py`
- cobrir agrupamento `1:1` como bloco conectado;
- cobrir escolha de composicao mais densa;
- cobrir penalty de linhas orfas e font-size excessivamente pequeno.

**Riscos**

- aumentar demais a agressividade do score e começar a esmagar texto em casos limite;
- introduzir regressao em baloes simples se a promocao de grupos conectados vazar para o fluxo comum;
- deixar o renderer mais pesado, o que e aceitavel nesta fase, mas precisa de fallback claro.

**Mitigacoes**

- manter thresholds e fallbacks explicitos;
- adicionar testes de regressao para agrupamento `1:1`, `N:M` e fallback;
- limitar a nova logica apenas a grupos com pelo menos 2 subregioes e confianca minima.

**Success Criteria**

- o caso enviado pelo usuario deixa de renderizar dois textos pequenos como baloes comuns;
- connected balloons passam a ocupar melhor cada lobo;
- fontes dos dois lados ficam proximas, mas podem variar ligeiramente quando isso melhora a leitura;
- textos longos deixam de quebrar em composicoes "magras" e artificiais;
- testes automatizados cobrindo esses cenarios passam.
