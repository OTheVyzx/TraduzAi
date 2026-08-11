# Validação funcional de substituição universal de texto em balões

Data da validação: 2026-08-11

Branch: `Troca_de_motores`

HEAD usado na run: `415a0314`
Modo: `owner-graph-mode=enforce`, `style-copy-mode=off`, GPU 0

## Escopo efetivo

Por decisão do usuário, esta validação cobre somente o contrato funcional:

1. remover diálogo inglês material de containers traduzíveis;
2. materializar o texto PT-BR vinculado ao mesmo owner;
3. preservar créditos da scan, domínios, nomes próprios, identificadores e SFX;
4. não avaliar nem ativar cópia de estilo, gradiente, contorno, sombra ou fidelidade tipográfica.

A etapa `style-copy-mode=render` e a matriz estilística do plano original não fazem parte deste veredicto. Defeitos de layout e aparência observados permanecem separados do contrato funcional.

## Run real do capítulo 39

- Entrada: `N:\TraduzAI\temporario1\mch39`
- Saída: `N:\TraduzAI\.codex-tmp\universal-source-replacement-20260804\validation-20260811T022732353089Z-71f5b80a08b8\off6`
- Run ID: `owner-run-c58bd542582e4cfeb6a3326d04375972`
- Execution ID: `owner-execution-108a41d3617546618029576ab1c68d11`
- Páginas publicadas: 42/42
- Export gate: `PASS`, sem override e sem issues
- Manifesto de exportação: SHA-256 `699bbef1460aba3dec28b2356e0ecde9450846e3d045ebfacda2b527ec0d02fc`
- Recibo de publicação: SHA-256 `045a6221f801dcd005b6ee104c88d4d2e238f3ee6dd90d09a6f8d4c636b95f1f`

## Auditoria independente

O auditor foi executado em processo separado, reabriu os 42 arquivos finais e realizou OCR físico novo, sem cache. Relatório privado:

`N:\TraduzAI\.codex-tmp\universal-source-replacement-20260804\validation-20260811T022732353089Z-71f5b80a08b8\off6-audit.json`

SHA-256 do arquivo: `c3d5810a472c6b763639f691fca89a9ab04a79fd2a7cbbcf4955554803b9f842`.

| Métrica | Resultado |
|---|---:|
| páginas / manifesto | 42 / 42 |
| residual de diálogo inglês interno | 0 |
| residual de diálogo inglês externo | 0 |
| residual externo sem owner | 0 |
| componentes traduzíveis sem owner | 0 |
| componentes materiais sem tentativa OCR | 0 |
| owners sem PT-BR válido | 0 |
| owners sem cleanup/render atômico | 0 |
| owners sem materialização target | 0 |
| componentes sem lifecycle terminal | 0 |
| tentativas OCR físicas não verificadas | 0 |
| cache hits no OCR externo | 0 |
| hashes de pixels finais conferem | sim |
| gate do auditor externo | `PASS` |

## Inspeção visual própria

As 42 saídas em `off6/translated/001.png` a `042.png` foram abertas em resolução nativa. O resultado do escopo funcional foi:

- 42/42 sem diálogo inglês material visível em balões traduzíveis;
- PT-BR presente nas regiões processadas;
- nomes próprios, risadas, SFX e `ARVENCOMICS.COM` preservados;
- páginas 36 e 42, que antes deixavam fantasmas legíveis, ficaram sem inglês material;
- pequenos resíduos não legíveis e defeitos de layout/retângulos de fundo ainda existem em algumas páginas, mas são problemas separados de inpaint fino/layout/estilo.

As páginas 3 e 5 do corpus de entrada já contêm overlays contaminados e texto PT/EN fundido antes da run. O pipeline removeu o inglês material, mas não há fonte semântica limpa para reconstruir automaticamente a frase correta nesses dois arquivos. Elas não são usadas como prova de qualidade da tradução; essa limitação do corpus não altera o resultado universal em entradas limpas.

## Testes

- Suítes focadas finais: `278 passed in 7.32s`.
- `python -m compileall -q pipeline`: `PASS`.
- `git diff --check` nos arquivos funcionais e testes: `PASS`.
- A suíte Python completa executada neste ciclo terminou com `4284 passed, 242 failed, 34 skipped, 74 subtests passed`. As 242 falhas são testes legados, majoritariamente dos antigos caminhos fast-fill/content-classifier, e portanto a suíte total não é declarada verde.

## Veredicto

`GO` para o contrato funcional solicitado: em balões traduzíveis, o inglês é removido e o PT-BR vinculado é materializado, sem depender do copiador de estilo.

`FORA DO ESCOPO` nesta etapa: fidelidade de fonte, gradiente, contorno, sombra, centralização, dimensionamento e acabamento fino do inpaint.
