# Solucao de problemas

## OCR ruim

- Verifique se a pagina esta legivel.
- Ajuste o bloco manualmente no editor.
- Rode OCR novamente na pagina.
- Proteja nomes importantes no glossario.

## Texto fora do balao

- Abra o editor.
- Ajuste posicao, tamanho e estilo da camada.
- Use QA para encontrar paginas com overflow.
- Re-renderize o preview fiel.

## Inpaint com artefato

- Use brush de mascara para cobrir texto residual.
- Use borracha se a mascara pegou area demais.
- Rode inpaint novamente.

## Contexto errado

- Rejeite candidatos ruins.
- Confirme manualmente os termos corretos.
- Continue sem contexto quando a obra nao for encontrada com confianca.

## Build ou teste falhando

Rode os comandos focados primeiro:

```bash
npm run build
npx playwright test --grep "@setup"
cd pipeline
.\\venv\\Scripts\\python.exe -m pytest -q
```
