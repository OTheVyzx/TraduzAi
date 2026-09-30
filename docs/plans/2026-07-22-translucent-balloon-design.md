# Perfil automático para balões translúcidos

## Objetivo

Adicionar o perfil persistente `translucent_balloon` para balões claros cuja
arte, gradiente ou figura permanece visível atrás do texto. O perfil evita
preenchimentos sólidos e preserva a textura da página durante o inpaint.

## Decisão

O classificador existente de fundo translúcido deixa de ser apenas uma recusa
local de fast fill. Quando houver evidência suficiente, ele grava
`layout_profile` e `block_profile` como `translucent_balloon`. Esse perfil
prevalece sobre a inferência de `dark_panel`.

## Fluxo

1. A segmentação detecta variação/gradiente ou arte visível dentro do balão.
2. O bloco recebe `translucent_balloon` e métricas de evidência.
3. A máscara fica limitada aos glifos e a uma expansão pequena, recortada pelo
   interior do balão.
4. Os caminhos de preenchimento sólido, incluindo `dark_panel_fill`, são
   proibidos.
5. O inpaint local/texturizado é obrigatório. Se não houver segurança para
   executá-lo, o bloco vai para revisão; ele nunca recebe um retângulo plano.
6. O QA confirma que o perfil não usou fill sólido e que a máscara não excede
   o interior do balão.

## Compatibilidade e segurança

- `white_balloon` só muda quando a evidência de translucidez for positiva.
- `dark_panel` continua disponível para painéis realmente opacos; não pode
  substituir um perfil translúcido já confirmado.
- SFX e textos sobre arte permanecem fora deste fluxo.
- A ausência ou baixa confiança de evidência favorece revisão, não uma
  classificação agressiva.

## Critérios de aceite

- O caso `page_001_band_005` de Mythic ch39 não usa `dark_panel_fill`.
- A arte/gradiente sob o texto continua visível, sem patch retangular.
- O perfil fica registrado nos artefatos de debug.
- Testes cobrem classificação, bloqueio de fill sólido e fallback seguro.
