export const features = [
  {
    title: "Contexto da obra",
    body: "Busca fontes como AniList, MyAnimeList, MangaUpdates, Wikipedia, Wikidata e Fandom sem enviar paginas do usuario.",
  },
  {
    title: "Glossario inteligente",
    body: "Preserva nomes, tecnicas, cargos, lugares e termos de lore com prioridade para termos revisados.",
  },
  {
    title: "Inpaint automatico",
    body: "Remove texto original dos baloes e reconstrui o fundo para preparar o typesetting.",
  },
  {
    title: "Typesetting editavel",
    body: "Recria o texto traduzido com ajuste de posicao, tamanho, estilo e revisao manual.",
  },
  {
    title: "QA automatico",
    body: "Detecta ingles restante, erro de glossario, texto grande demais e paginas suspeitas.",
  },
  {
    title: "Processamento local",
    body: "As imagens ficam no computador. Recursos online podem ser desativados.",
  },
];

export const steps = [
  "Importe seu capitulo",
  "Busque o contexto da obra",
  "Revise o glossario",
  "Traduza automaticamente",
  "Corrija alertas no editor",
  "Exporte imagens, CBZ ou projeto editavel",
];

export const plans = [
  {
    name: "Free",
    price: "Beta local",
    items: ["Testes locais", "Modo mock", "Poucas paginas", "Export basico"],
  },
  {
    name: "Pro",
    price: "Em desenvolvimento",
    items: ["Traducao com API propria", "Contexto online", "QA completo", "Glossario e memoria", "Export avancado"],
  },
  {
    name: "Studio",
    price: "Futuro",
    items: ["Lotes", "Presets", "Relatorios", "Pacotes de revisao", "Recursos avancados"],
  },
];

export const faq = [
  {
    question: "O TraduzAI fornece mangas?",
    answer: "Nao. O app edita arquivos do proprio usuario e nao hospeda nem distribui obras.",
  },
  {
    question: "Minhas paginas sao enviadas para a internet?",
    answer: "Nao para busca de contexto. A internet e usada apenas para contexto textual e traducao quando o usuario ativa esses recursos.",
  },
  {
    question: "Posso revisar antes de exportar?",
    answer: "Sim. O fluxo inclui glossario, memoria, editor visual, QA e modos de export.",
  },
];
