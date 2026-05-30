# Vast.ai worker pronto para TraduzAI

Este fluxo reduz o tempo sem manter GPU ligada o dia todo: a instancia nasce com dependencias e modelos prontos, roda o worker, processa o capitulo e depois voce destrói a instancia.

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

## Arquivo de ambiente

Crie `/workspace/traduzai-worker.env` na instancia/template:

```bash
TRADUZAI_API_URL=https://SUA-API.trycloudflare.com
TRADUZAI_WORKER_TOKEN=troque-por-um-token-forte
TRADUZAI_FAST_PAGE_SERVER=1
TRADUZAI_WORKER_WARMUP_ON_START=1
TRADUZAI_WARMUP_PROFILE=quality
TRADUZAI_WARMUP_LANG=en
```

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

## Fluxo barato por capitulo

1. Suba uma instancia a partir do template pronto.
2. Mantenha sua API local e o Cloudflare Tunnel ativos.
3. Crie o job no site.
4. Rode `bash scripts/vast/start-worker.sh`.
5. Quando o job terminar e os artifacts aparecerem, destrua a instancia no Vast.

Com template pronto, o tempo perdido fica concentrado em boot + warmup, nao em instalacao. Para custo baixo, compare sempre por custo por capitulo, nao apenas por preco/hora.
