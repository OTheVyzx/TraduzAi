# OCR Prewarm And Progress Design

## Objetivo

Reduzir o cold start do OCR no primeiro processamento e melhorar a percepcao de progresso na UI durante a etapa de OCR, especialmente em capitulos com uma unica pagina.

## Decisoes

1. Adicionar um prewarm do stack visual no boot da app.
   O prewarm vai carregar detector, PaddleOCR e font detector em background, usando o Python do pipeline via um comando novo do Tauri.

2. Adicionar progresso granular dentro do OCR.
   A pipeline passara a emitir subetapas como carregamento de modelos, deteccao, reconhecimento e pos-processamento/fonte, em vez de atualizar so ao fim da pagina.

3. Manter a UI simples.
   O frontend vai reaproveitar `pipeline-progress` e um estado leve de warmup, sem criar uma tela nova de diagnostico.

## Arquitetura

### Backend/Tauri

- Novo comando `warmup_visual_stack`.
- O comando sobe um processo Python curto e idempotente.
- O comando emite eventos de warmup para a UI, mas nao bloqueia o boot.
- Em `AppInit`, o warmup sera disparado uma vez so, protegido contra `StrictMode`.

### Python

- Novo modulo de warmup do stack visual.
- `run_detect_ocr` e `run_ocr` passam a aceitar callback opcional de progresso.
- O OCR emitira subetapas:
  - preparar imagem
  - carregar detector
  - carregar OCR
  - detectar baloes
  - reconhecer texto
  - analisar estilo/fonte
  - revisar layout

### Frontend

- `App.tsx` dispara o prewarm em background no boot.
- `Processing.tsx` passa a mostrar a mensagem granular recebida da pipeline.
- O percentual do OCR deixa de ficar travado em `0%` em jobs de uma pagina.

## Tratamento de erro

- Falha no prewarm nao pode impedir a app de abrir.
- Se o warmup falhar, a app segue e o processamento normal continua funcionando.
- O OCR continua funcional mesmo sem callback de progresso.

## Testes

- Teste Python para garantir que `run_detect_ocr` envia callbacks granulares na ordem esperada.
- Teste Rust para garantir que o comando de warmup monta o processo Python esperado ou ao menos que o helper de spawn usa o Python do pipeline.
- Validacao manual com a `002__002.jpg` para medir tempo frio antes/depois.
