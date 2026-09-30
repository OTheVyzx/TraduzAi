# TraduzAI Source Runtime

Runtime local do TraduzAI para extensões Mihon/Tachiyomi. O runtime será distribuído com uma JVM privada e não dependerá de um servidor Suwayomi externo.

## Fronteiras

- O backend Tauri/Rust controla lifecycle, sandbox, staging e acesso aos projetos.
- O broker JVM controla catálogo, biblioteca de leitura e agendamentos.
- Cada pacote de extensão executa em um worker JVM separado.
- Nenhum repositório ou extensão é configurado por padrão.
- Código derivado do Suwayomi/AndroidCompat permanece isolado sob MPL-2.0.

O estado de leitura pertence ao runtime. A biblioteca de projetos de tradução continua separada e recebe páginas somente por importação explícita e transacional.

Consulte `docs/protocol-v1.md`, `docs/schema-v1.sql` e `docs/THREAT_MODEL.md` antes de implementar novos comandos.

## Bootstrap reproduzível no Windows

`scripts/bootstrap-windows.ps1` baixa e verifica JDK/Gradle fixados, obtém o commit Suwayomi registrado em `UPSTREAM.lock.json` e produz a distribuição de compatibilidade. O script recusa um checkout divergente ou com alterações locais.

`scripts/prepare-runtime.ps1` compila broker e worker, gera o JRE privado com `jlink`, copia notices e lockfile e cria um SBOM CycloneDX do classpath empacotado.

## Estado desta entrega

Implementado e testado: protocolo JSONL v1 com UTF-8 explícito, JVM 21 privada, broker, worker por operação, carregamento da API Mihon e AndroidCompat no snapshot Suwayomi fixado, fixture JAR, persistência de `memo`/cookies/preferências, repositório HTTPS com fingerprint e chave de assinatura, catálogos JSON legados e Protobuf v2 compactados com gzip, validação criptográfica de JAR assinado, quarentena, busca/detalhes/capítulos/páginas, populares/recentes e Job Object no Windows. Uma extensão oficial Keiyoushi foi carregada e consultada em teste de fumaça não determinístico.

Ainda não é uma release compatível com todo o ecossistema Mihon: a matriz API 1.3–1.6 não foi congelada, conversão APK/dex2jar, AppContainer/token restrito/ACL/WFP, leitor visual de páginas, downloads, scheduler, backup, ponte transacional para tradução, assinatura Authenticode e validação em VM limpa permanecem gates abertos. Fontes WebView/login seguem explicitamente fora da primeira fase e pacotes sem caminho JAR falham fechados.
