# Vast.ai worker pronto para TraduzAI

Este fluxo reduz o tempo sem manter GPU ligada o dia todo: a instancia nasce com dependencias e modelos prontos, roda o worker, processa o capitulo e depois voce para ou destroi a instancia.

## Objetivo

Evitar repetir estes custos em todo job:

- instalar dependencias Python
- baixar modelos
- descobrir dependencias faltando durante o pipeline

O job ainda paga o boot da instancia e o aquecimento inicial, mas nao paga `pip install` e download de modelos.

## Bootstrap unico em uma instancia Vast

Use um template PyTorch/CUDA com Jupyter ou SSH. Na primeira instancia, clone o repo e rode o bootstrap:

```bash
cd /workspace
git clone https://github.com/OTheVyzx/TraduzAi.git TraduzAI
cd TraduzAI
bash scripts/vast/bootstrap.sh
```

Depois desse bootstrap, salve a instancia como template/snapshot no Vast. Esse template e o que deve ser usado nos proximos jobs.

Para repo privado, defina `TRADUZAI_REPO_URL` com a URL que a instancia consegue acessar antes de rodar o bootstrap.

## Arquivo de ambiente da instancia

Crie `/workspace/traduzai-worker.env` na instancia/template:

```bash
TRADUZAI_API_URL=https://SUA-API.trycloudflare.com
TRADUZAI_WORKER_TOKEN=troque-por-um-token-forte
TRADUZAI_FAST_PAGE_SERVER=1
TRADUZAI_WORKER_WARMUP_ON_START=1
TRADUZAI_WARMUP_PROFILE=quality
TRADUZAI_WARMUP_LANG=en
```

Use o mesmo valor de `TRADUZAI_WORKER_TOKEN` no backend local. Se o token for diferente, a instancia sobe mas nao consegue pegar jobs.

Se o fast-page falhar em uma GPU/template especifico, troque para:

```bash
TRADUZAI_FAST_PAGE_SERVER=0
```

## Start do worker

```bash
cd /workspace/TraduzAI
bash scripts/vast/start-worker.sh
```

Para rodar uma unica tentativa e sair:

```bash
TRADUZAI_WORKER_ONCE=1 bash scripts/vast/start-worker.sh
```

Use `--once` somente quando o job ja estiver na fila. Se nao houver job, o worker encerra sem processar nada.

## Warmup manual

Para validar o ambiente antes de criar job:

```bash
cd /workspace/TraduzAI
bash scripts/vast/warmup.sh
```

Esse comando testa o worker e o servidor fast-page. O `start-worker.sh` tambem faz warmup real no mesmo processo quando `TRADUZAI_WORKER_WARMUP_ON_START=1`.

Para validar a stack de GPU do template, use:

```bash
cd /workspace/TraduzAI
source .venv/bin/activate
python scripts/vast/verify-gpu-stack.py
```

No Vast, o bootstrap instala PaddleOCR GPU primeiro e depois instala PyTorch CUDA sem resolver dependencias novamente. Isso evita o conflito conhecido entre os pins de `nvidia-nccl-cu12` do Paddle e do PyTorch. Para esse template, prefira `verify-gpu-stack.py` em vez de `pip check`.

## Automacao pelo backend

No backend local, configure:

```env
VAST_AUTOSTART=1
VAST_API_KEY=sua-chave-da-vast
VAST_INSTANCE_ID=38646242
VAST_IDLE_STOP_MINUTES=10
TRADUZAI_WORKER_TOKEN=troque-por-um-token-forte
```

Com `VAST_INSTANCE_ID`, o servidor tenta religar a instancia pausada quando um job real entra na fila. Se preferir criar uma nova instancia quando nao houver uma fixa, configure tambem:

```env
VAST_OFFER_ID=12345
VAST_IMAGE=vastai/pytorch:cuda-12.1.1-auto
VAST_WORKER_API_URL=https://SUA-API.trycloudflare.com
VAST_REPO_BRANCH=Troca_de_motores
VAST_DISK_GB=80
```

Para deixar o backend escolher a oferta automaticamente, nao defina `VAST_OFFER_ID` e use filtros:

```env
VAST_OFFER_AUTO=1
VAST_IMAGE=vastai/pytorch:cuda-12.1.1-auto
VAST_RUNTYPE=jupyter_direct
VAST_WORKER_API_URL=https://SUA-API.trycloudflare.com
VAST_REPO_BRANCH=Troca_de_motores
VAST_DISK_GB=80
VAST_OFFER_MAX_DPH=0.20
VAST_OFFER_MIN_GPU_RAM_GB=16
VAST_OFFER_MIN_RELIABILITY=0.98
VAST_OFFER_MIN_DLPERF=5.0
VAST_OFFER_MIN_DIRECT_PORTS=1
VAST_OFFER_MIN_CUDA=12.1
VAST_OFFER_GPU_NAMES=Tesla P100,RTX 3090,RTX 4090
```

O orquestrador busca ofertas verificadas, disponiveis, NVIDIA, 1 GPU, dentro do preco e com VRAM suficiente. Depois escolhe a mais barata; em empate, prefere maior confiabilidade e maior DLPerf.

Por padrao, o backend cria a instancia diretamente com `VAST_IMAGE=vastai/pytorch:cuda-12.1.1-auto` e `VAST_RUNTYPE=jupyter_direct`, sem depender de um template customizado. Isso evita falhas da Vast como `docker_build() error writing dockerfile` quando o template salvo esta invalido. Se quiser forcar template, deixe `VAST_IMAGE` vazio e configure `VAST_TEMPLATE_HASH`.

Quando o backend cria uma instancia nova via `VAST_OFFER_ID` ou `VAST_OFFER_AUTO`, ele envia um `onstart` para a Vast que:

1. cria `/workspace/traduzai-worker.env` com `TRADUZAI_API_URL`, `TRADUZAI_WORKER_TOKEN`, warmup e GPU;
2. clona ou atualiza `/workspace/TraduzAI`;
3. roda `scripts/vast/bootstrap.sh`;
4. inicia `scripts/vast/start-worker.sh`.

Ou seja: em instancias criadas pelo orquestrador, voce nao precisa criar `traduzai-worker.env` manualmente. Para instancias pausadas ja existentes em `VAST_INSTANCE_ID`, o arquivo ainda precisa existir na propria instancia/template, porque o start de uma instancia existente apenas religa a maquina.

A API key da Vast fica somente no servidor. Nunca coloque `VAST_API_KEY` no frontend.

## Fluxo barato por capitulo

1. Mantenha sua API local e o Cloudflare Tunnel ativos.
2. Crie o job no site.
3. O backend chama a API da Vast e inicia a instancia pausada.
4. O worker roda `bash scripts/vast/start-worker.sh` e pega o job da fila.
5. Quando a fila ficar vazia, o backend pode parar a instancia se `VAST_IDLE_STOP_MINUTES` estiver ativo.

Com template pronto, o tempo perdido fica concentrado em boot + warmup, nao em instalacao. Para custo baixo, compare sempre por custo por capitulo, nao apenas por preco/hora.
