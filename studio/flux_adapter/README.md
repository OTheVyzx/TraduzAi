# Adaptador FLUX local do Studio

Este worker implementa o contrato JSONL `1.0` usado pelo TraduzAI Studio. Ele recebe crop, máscara e prompt pelo `stdin`, executa `FluxFillPipeline` localmente e devolve 2 a 4 PNGs pelo `stdout`.

O processo permanece aberto entre jobs para manter o modelo carregado em memória. O Studio serializa as gerações; ao cancelar, encerra de verdade o worker e inicia outro somente no próximo job. Assim, cancelamento libera GPU/RAM e gerações normais não recarregam o checkpoint a cada uso.

Nenhuma imagem é enviada para a internet. O download de modelo fica bloqueado por padrão; use um caminho local ou um modelo já presente no cache do Hugging Face.

## Dependencias

No ambiente Python que será usado pelo Studio:

```powershell
python -m pip install torch diffusers transformers accelerate pillow safetensors
```

## Configuracao no PowerShell

```powershell
$env:TRADUZAI_STUDIO_FLUX_COMMAND="C:\caminho\python.exe"
$env:TRADUZAI_STUDIO_FLUX_ARGS_JSON='["N:\\TraduzAI\\studio\\flux_adapter\\worker.py"]'
$env:TRADUZAI_STUDIO_FLUX_MODEL="D:\modelos\flux-fill-local"
npm --prefix studio run tauri:dev
```

O modelo precisa ser compatível com `diffusers.FluxFillPipeline`. Para GPUs com pouca VRAM, `TRADUZAI_STUDIO_FLUX_CPU_OFFLOAD=1` é o padrão. Um checkpoint quantizado/local pode ser selecionado por `TRADUZAI_STUDIO_FLUX_MODEL`, desde que seja carregável pela mesma pipeline.

O painel mostra o adaptador como `configurado` antes do primeiro job; dependências e disponibilidade real do modelo são validadas ao gerar, sem prometer um estado “pronto” prematuramente.

Downloads automáticos permanecem desativados. Somente se o usuário aceitar a licença do modelo e quiser baixar pelo Hugging Face:

```powershell
$env:TRADUZAI_STUDIO_FLUX_ALLOW_MODEL_DOWNLOAD="1"
```
