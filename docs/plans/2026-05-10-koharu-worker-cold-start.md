# Plano: reduzir cold start do vision-worker Koharu

## Contexto

O benchmark local mostrou que o `traduzai-vision.exe` acerta melhor alguns casos de SFX, mas sofre com cold start: a primeira pagina levou cerca de 81s enquanto as seguintes ficaram perto de 3s. O binario atual e CLI de uma requisicao: carrega runtime/modelos, processa, imprime JSON e encerra. O `--warmup` tambem encerra depois de preparar os modelos, entao nao mantem GPU/RAM aquecidas.

## Objetivo

Manter o worker vivo durante a sessao do app e enviar varias paginas/ROIs para ele sem recarregar detector, LLM backend e PaddleOCR-VL a cada chamada.

## Fase 1: medir sem mudar contrato publico

- Adicionar telemetria por chamada: `prepare_ms`, `detect_ms`, `ocr_ms`, `total_ms`, `cold_start`.
- Expor no pipeline a diferenca entre primeira chamada, chamadas seguintes e batch.
- Comparar HTTP Koharu vs worker em paginas com SFX, balao conectado e texto CJK comum.

## Fase 2: worker batch

- Criar modo `--batch-request-file` aceitando uma lista de requests.
- Carregar runtime/modelos uma vez e processar N paginas/ROIs em sequencia.
- Retornar lista de respostas preservando ordem e erro por item.
- Conectar primeiro apenas ao precompute CJK ROI, atras de flag.

## Fase 3: daemon persistente

- Adicionar modo `--serve-stdio` ou HTTP local.
- Inicializar o worker no start do Tauri dev/app, com cancelamento e health check.
- Reusar o processo em `run_koharu_cjk_pages`.
- Encerrar no shutdown do app.

## Decisao recomendada

Implementar primeiro o batch do worker. Ele reduz cold start sem exigir ciclo de vida novo no Tauri. Depois, se o ganho justificar, evoluir para daemon persistente.

## Status 2026-05-10

- Fase 2 implementada no codigo: `traduzai-vision` aceita `--batch-request-file`, carrega runtime/detector/OCR uma vez e retorna respostas por item.
- O pipeline CJK em strip agora tenta `vision_worker_path` em batch primeiro e cai para Koharu HTTP batch se o binario antigo ou o ambiente falhar.
- Validado no Python com `test_vision_stack_runtime.py` completo e teste focado de precompute ROI.
- Validacao Rust desbloqueada com MSVC Developer Prompt, CUDA 13.2, `LLAMA_CPP_TAG=b8935`, `LIBCLANG_PATH` do `.toolvenv` e `CARGO_TARGET_DIR=N:\t\vw`.
- O worker precisou seguir o padrao dos binarios Koharu no Windows: thread com stack de 64 MB. Sem isso, o runtime CUDA/CuDNN estourava stack durante o prepare.
- `stdout` do worker agora fica reservado para uma unica linha JSON; logs nativos ficam em `stderr`. Tambem ha `process::exit` apos flush do JSON para evitar panic de drop do `cudarc/cuDNN` no encerramento.
- Benchmark real do capitulo `exemplos\exemploko\환생천마\113화.cbz`: total caiu de `377.8s` para `135.9s`; `koharu_cjk_precompute` caiu de `274.6s` para `69.2s`, sem fallback no log.
- Saidas de referencia: fallback antigo `debug\koharu_worker_batch_benchmark_20260510_180243`; worker batch valido `debug\koharu_worker_batch_benchmark_20260510_195829`.
