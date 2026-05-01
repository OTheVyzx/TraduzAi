# Export

## Modos

- `Clean`: export final sem flags criticas ou altas ativas.
- `With warnings`: export com avisos preservados.
- `Debug`: export tecnico para investigacao.
- `Review package`: pacote para revisao externa.

## Bloqueio de export limpo

O modo `Clean` fica bloqueado quando ainda existem flags criticas ou altas. Corrija a pagina ou ignore a flag com motivo antes de exportar limpo.

## CBZ e imagens

Use export de imagens para leitura comum. Use CBZ quando quiser um pacote unico. Preserve o projeto editavel quando ainda houver revisao pendente.

## Relatorios

Exports com avisos devem incluir relatorio de QA para explicar o que foi aceito, corrigido ou ignorado.
