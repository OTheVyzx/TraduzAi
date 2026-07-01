# Vast.ai Serverless no TraduzAI

Este modo e opcional. O fluxo atual com instancia Vast normal continua funcionando.

## Como funciona

1. O site cria o job e salva o arquivo de entrada.
2. O backend enfileira o job.
3. Com `VAST_PROVIDER=serverless`, o backend garante que existe um Endpoint + Workergroup no Vast Serverless.
4. O backend pede uma rota ao Vast e chama `/run` no worker serverless.
5. O worker serverless reivindica o `job_id` especifico no backend, baixa o arquivo, roda o pipeline e envia artifacts/logs.

## Variaveis minimas do backend

```powershell
$env:VAST_AUTOSTART="1"
$env:VAST_PROVIDER="serverless"
$env:VAST_API_KEY="sua-chave-vast"
$env:VAST_WORKER_API_URL="https://seu-tunnel.trycloudflare.com"
$env:TRADUZAI_WORKER_TOKEN="mesmo-token-do-worker"

$env:VAST_SERVERLESS_ENDPOINT_NAME="traduzai-serverless"
$env:VAST_SERVERLESS_WORKERGROUP_NAME="traduzai-worker"
$env:VAST_SERVERLESS_TEMPLATE_HASH="hash-do-template-serverless"

$env:VAST_OFFER_MAX_DPH="0.160"
$env:VAST_OFFER_MIN_GPU_RAM_GB="12"
$env:VAST_OFFER_MIN_RELIABILITY="0.98"
$env:VAST_OFFER_MIN_DLPERF="5.0"
$env:VAST_OFFER_MIN_CUDA="12.6"
$env:VAST_OFFER_GPU_NAMES="RTX 3060,RTX 3070,RTX 3080,RTX 3090,RTX 4060,RTX 4070,RTX 4080,RTX 4090,RTX 5060,RTX 5070,RTX 5080,RTX 5090"

npm run saas:server
```

Se ja existir um endpoint, defina `VAST_SERVERLESS_ENDPOINT_ID` para evitar lookup por nome.

## Template serverless

O template deve usar a imagem Docker do TraduzAI com dependencias e modelos ja instalados. Para buildar uma imagem compativel:

```bash
docker build \
  -f scripts/vast/Dockerfile \
  --build-arg TRADUZAI_REPO_BRANCH=Troca_de_motores \
  --build-arg TRADUZAI_INSTALL_SERVERLESS=1 \
  -t seu-registry/traduzai-worker:serverless .
```

No template serverless, o comando de start deve chamar:

```bash
bash /workspace/TraduzAI/scripts/vast/start-serverless-worker.sh
```

E o env do template deve conter:

```bash
TRADUZAI_API_URL=https://seu-tunnel.trycloudflare.com
TRADUZAI_WORKER_TOKEN=mesmo-token-do-backend
TRADUZAI_REPO_BRANCH=Troca_de_motores
TRADUZAI_FAST_PAGE_SERVER=1
TRADUZAI_WORKER_WARMUP_ON_START=1
TRADUZAI_REQUIRE_GPU=1
```

## Observacoes

- Serverless so fica rapido se a imagem ja vier com PaddleOCR, Torch CUDA e modelos baixados.
- Com `min_load=0`, o primeiro job ainda pode ter cold start.
- Para reduzir espera, use `VAST_SERVERLESS_MIN_LOAD=1`, mas isso passa a manter capacidade pronta e aumenta custo.
- O worker reivindica o `job_id` recebido, entao jobs de usuarios diferentes nao devem ser trocados quando varios estiverem na fila.
