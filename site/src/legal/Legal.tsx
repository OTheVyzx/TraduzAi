import { ReactNode, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";

const LAST_UPDATED = "24 de maio de 2026";
const VERSION = "1.0";
const CONTACT_EMAIL = "[EMAIL_CONTATO]";
const SERVICE_NAME = "TraduzAi";

type DocMeta = {
  slug: string;
  path: string;
  title: string;
  short: string;
  description: string;
};

export const LEGAL_DOCS: DocMeta[] = [
  {
    slug: "termos",
    path: "/legal/termos",
    title: "Termos de Uso",
    short: "Termos",
    description: "Regras de uso do serviço, criação de conta, créditos e responsabilidades.",
  },
  {
    slug: "privacidade",
    path: "/legal/privacidade",
    title: "Política de Privacidade",
    short: "Privacidade",
    description: "Dados pessoais coletados, finalidades, bases legais (LGPD) e direitos do titular.",
  },
  {
    slug: "cookies",
    path: "/legal/cookies",
    title: "Política de Cookies",
    short: "Cookies",
    description: "Quais cookies utilizamos, com qual finalidade e como gerenciar seu consentimento.",
  },
  {
    slug: "direitos-autorais",
    path: "/legal/direitos-autorais",
    title: "Política de Direitos Autorais e Notificação de Remoção",
    short: "Direitos autorais",
    description: "Postura sobre conteúdo protegido e procedimento de notificação extrajudicial.",
  },
  {
    slug: "uso-aceitavel",
    path: "/legal/uso-aceitavel",
    title: "Política de Uso Aceitável",
    short: "Uso aceitável",
    description: "Condutas proibidas e responsabilidades do usuário na utilização do serviço.",
  },
  {
    slug: "isencao",
    path: "/legal/isencao",
    title: "Isenção de Garantias e Limitação de Responsabilidade",
    short: "Isenção",
    description: "Limitações de responsabilidade e ausência de garantias sobre o resultado da IA.",
  },
];

function useScrollTopOnRoute() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0 });
  }, [pathname]);
}

function LegalShell({ title, children }: { title: string; children: ReactNode }) {
  useScrollTopOnRoute();
  return (
    <main className="legal-shell">
      <nav className="legal-shell-topbar" aria-label="Navegação">
        <Link className="legal-shell-brand" to="/">
          <img src="/assets/traduzai-logo.svg" alt={SERVICE_NAME} />
        </Link>
        <div className="legal-shell-links">
          <Link to="/legal">Índice legal</Link>
          <Link to="/login">Entrar</Link>
        </div>
      </nav>
      <article className="legal-document">
        <header className="legal-document-head">
          <p className="eyebrow">Documento legal</p>
          <h1>{title}</h1>
          <p className="legal-meta">
            Versão {VERSION} · Última atualização em {LAST_UPDATED}
          </p>
        </header>
        <div className="legal-document-body">{children}</div>
        <footer className="legal-document-foot">
          <p>
            Dúvidas sobre este documento? Entre em contato pelo e-mail{" "}
            <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>.
          </p>
          <Link className="legal-back-link" to="/legal">
            ← Voltar ao índice legal
          </Link>
        </footer>
      </article>
    </main>
  );
}

export function LegalIndex() {
  useScrollTopOnRoute();
  return (
    <main className="legal-shell">
      <nav className="legal-shell-topbar" aria-label="Navegação">
        <Link className="legal-shell-brand" to="/">
          <img src="/assets/traduzai-logo.svg" alt={SERVICE_NAME} />
        </Link>
        <div className="legal-shell-links">
          <Link to="/login">Entrar</Link>
          <Link to="/signup">Criar conta</Link>
        </div>
      </nav>
      <article className="legal-document legal-index">
        <header className="legal-document-head">
          <p className="eyebrow">Central legal</p>
          <h1>Documentos do {SERVICE_NAME}</h1>
          <p>
            Esta página reúne os documentos que regem o uso do {SERVICE_NAME}. Recomendamos a leitura
            antes de criar uma conta ou enviar arquivos. Todos os documentos estão na versão {VERSION},
            atualizados em {LAST_UPDATED}.
          </p>
        </header>
        <ul className="legal-index-list">
          {LEGAL_DOCS.map((doc) => (
            <li key={doc.slug}>
              <Link to={doc.path}>
                <strong>{doc.title}</strong>
                <span>{doc.description}</span>
              </Link>
            </li>
          ))}
        </ul>
        <footer className="legal-document-foot">
          <p>
            Para solicitações relacionadas à LGPD, notificações de remoção (direitos autorais) ou
            outros assuntos jurídicos, utilize o e-mail{" "}
            <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>.
          </p>
        </footer>
      </article>
    </main>
  );
}

export function LegalTermsPage() {
  return (
    <LegalShell title="Termos de Uso">
      <p>
        Estes Termos de Uso ("Termos") regulam o acesso e a utilização do serviço {SERVICE_NAME}
        ("Serviço"), uma ferramenta de software destinada à edição assistida por inteligência artificial
        de páginas de quadrinhos, mangás, manhwas, manhuas e materiais visuais análogos. Ao criar uma
        conta, acessar o site ou utilizar o Serviço, você ("Usuário") declara que leu, compreendeu e
        concorda integralmente com estes Termos e com os demais documentos da Central Legal.
      </p>

      <h2>1. Natureza do Serviço</h2>
      <p>
        O {SERVICE_NAME} é uma ferramenta de edição que aplica processos automatizados de detecção de
        texto (OCR), tradução automática, remoção de texto da imagem (inpainting) e composição
        tipográfica (typesetting) sobre arquivos de imagem fornecidos pelo próprio Usuário. O Serviço
        não hospeda, indexa, distribui, recomenda, intermedia ou de qualquer forma fornece obras
        protegidas por direitos autorais.
      </p>
      <p>
        O Serviço é fornecido como ferramenta neutra. A escolha dos arquivos a serem processados, a
        verificação da titularidade ou autorização de uso e a destinação dos resultados são de
        responsabilidade exclusiva do Usuário.
      </p>

      <h2>2. Elegibilidade</h2>
      <p>
        O uso do Serviço é permitido a pessoas com idade igual ou superior a 16 (dezesseis) anos.
        Adolescentes entre 16 e 17 anos devem utilizar o Serviço com ciência e sob a supervisão de
        seus responsáveis legais, conforme aplicável. O cadastro e uso por crianças menores de 16
        anos não é permitido.
      </p>
      <p>
        A contratação de planos pagos, quando disponíveis, exige idade mínima de 18 (dezoito) anos
        completos e plena capacidade civil para contratar, nos termos da legislação brasileira. Em
        caso de cadastro ou contratação irregular, a conta poderá ser encerrada e os valores
        eventualmente pagos restituídos na forma da lei.
      </p>

      <h2>3. Cadastro e conta</h2>
      <p>
        Para utilizar funcionalidades autenticadas, o Usuário deverá criar uma conta fornecendo dados
        verdadeiros, atualizados e completos. O Usuário é responsável pela guarda das credenciais e por
        todas as atividades realizadas em sua conta. Em caso de uso não autorizado, o Usuário deve
        notificar o {SERVICE_NAME} imediatamente pelo e-mail{" "}
        <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>.
      </p>

      <h2>4. Modelo de uso, créditos e plano gratuito</h2>
      <p>
        O Serviço opera por sistema de créditos, onde 1 (um) crédito equivale ao processamento de 1
        (uma) página. O plano gratuito ("Plano Free") concede 40 (quarenta) páginas por semana,
        renovadas automaticamente toda segunda-feira, à exclusivo critério do {SERVICE_NAME}. O Plano
        Free pode ser modificado, suspenso ou descontinuado a qualquer tempo, sem aviso prévio e sem
        gerar qualquer direito a indenização.
      </p>
      <p>
        Créditos pagos, quando disponíveis comercialmente, terão sua precificação, regras de uso,
        validade e política de reembolso descritas em documento próprio no momento da contratação,
        que passará a integrar estes Termos.
      </p>

      <h2>5. Licença de uso</h2>
      <p>
        O {SERVICE_NAME} concede ao Usuário licença pessoal, não exclusiva, intransferível, não
        sublicenciável e revogável para acessar e utilizar o Serviço estritamente conforme estes
        Termos. Esta licença não confere ao Usuário qualquer direito sobre a marca, código-fonte,
        modelos, fontes tipográficas, layouts de interface, documentação ou demais elementos de
        propriedade intelectual do Serviço, que permanecem de titularidade exclusiva do{" "}
        {SERVICE_NAME} ou de seus respectivos licenciantes.
      </p>

      <h2>6. Conteúdo do Usuário</h2>
      <p>
        O Usuário mantém integralmente a titularidade dos arquivos e textos que enviar ao Serviço
        ("Conteúdo do Usuário"). Ao utilizar o Serviço, o Usuário concede ao {SERVICE_NAME} licença
        limitada, não exclusiva e gratuita para processar, armazenar temporariamente, transmitir e
        exibir o Conteúdo do Usuário exclusivamente na medida necessária para a operação do Serviço,
        suporte técnico e cumprimento de obrigações legais.
      </p>
      <p>
        O Usuário declara e garante que: (a) é o titular dos direitos sobre o Conteúdo do Usuário ou
        possui todas as autorizações necessárias para utilizá-lo no Serviço; (b) o Conteúdo do
        Usuário não viola direitos de terceiros, incluindo, sem limitação, direitos autorais, marcas,
        privacidade, honra ou imagem; e (c) o Conteúdo do Usuário não infringe a legislação aplicável.
      </p>

      <h2>7. Condutas proibidas</h2>
      <p>
        É vedado ao Usuário utilizar o Serviço para fins ilícitos, violar direitos de terceiros,
        contornar medidas de proteção tecnológica (DRM), processar material com conteúdo proibido por
        lei (incluindo, sem limitação, material de abuso sexual infantil), ou de qualquer forma
        comprometer a segurança ou a disponibilidade do Serviço. A Política de Uso Aceitável detalha
        as condutas proibidas e integra estes Termos.
      </p>

      <h2>8. Suspensão e encerramento</h2>
      <p>
        O {SERVICE_NAME} poderá suspender ou encerrar o acesso do Usuário, sem aviso prévio e a seu
        exclusivo critério, em caso de descumprimento destes Termos, de qualquer outro documento da
        Central Legal, de determinação legal ou administrativa, ou de notificação fundamentada de
        terceiros. O Usuário poderá encerrar sua conta a qualquer momento.
      </p>

      <h2>9. Isenção de garantias</h2>
      <p>
        O Serviço é fornecido "no estado em que se encontra" e "conforme disponível", sem qualquer
        tipo de garantia, expressa ou implícita, sobre disponibilidade, exatidão dos resultados,
        adequação a finalidade específica, qualidade da tradução automática, integridade dos arquivos
        ou ausência de erros. Detalhes adicionais constam do documento de Isenção de Garantias.
      </p>

      <h2>10. Limitação de responsabilidade</h2>
      <p>
        Na máxima extensão permitida pela legislação brasileira, o {SERVICE_NAME} não responderá por
        danos indiretos, lucros cessantes, perda de dados, perda de oportunidade, danos morais ou
        quaisquer prejuízos decorrentes do uso ou da impossibilidade de uso do Serviço. A
        responsabilidade total e agregada do {SERVICE_NAME}, em qualquer hipótese, fica limitada ao
        valor efetivamente pago pelo Usuário ao {SERVICE_NAME} nos 3 (três) meses anteriores ao fato
        gerador, ou a R$ 100,00 (cem reais), o que for menor.
      </p>

      <h2>11. Alterações destes Termos</h2>
      <p>
        Estes Termos podem ser alterados a qualquer tempo. Alterações materiais serão comunicadas com
        antecedência razoável por meio do site ou do e-mail cadastrado. O uso continuado do Serviço
        após a vigência da nova versão importa em aceitação tácita.
      </p>

      <h2>12. Lei aplicável e foro</h2>
      <p>
        Estes Termos são regidos pelas leis da República Federativa do Brasil. Fica eleito o foro da
        Comarca de São Paulo/SP como competente para dirimir quaisquer controvérsias decorrentes
        destes Termos, com renúncia expressa a qualquer outro, por mais privilegiado que seja, salvo
        nas hipóteses em que a legislação consumerista determinar de outra forma.
      </p>

      <h2>13. Disposições finais</h2>
      <p>
        A eventual tolerância quanto ao descumprimento de qualquer cláusula destes Termos não
        constituirá novação ou renúncia ao direito de exigi-la posteriormente. Caso qualquer
        disposição destes Termos seja considerada inválida ou inexequível, as demais permanecerão em
        pleno vigor.
      </p>
    </LegalShell>
  );
}

export function LegalPrivacyPage() {
  return (
    <LegalShell title="Política de Privacidade">
      <p>
        Esta Política de Privacidade descreve como o {SERVICE_NAME} ("nós") coleta, utiliza,
        compartilha e protege dados pessoais dos Usuários ("você"), em observância à Lei nº 13.709/2018
        (Lei Geral de Proteção de Dados — LGPD), ao Marco Civil da Internet (Lei nº 12.965/2014) e
        demais normas aplicáveis.
      </p>

      <h2>1. Controlador e contato</h2>
      <p>
        O controlador dos dados tratados no âmbito do Serviço é a operação responsável pela marca{" "}
        {SERVICE_NAME}. O contato para qualquer assunto relacionado a dados pessoais, incluindo
        exercício de direitos do titular e atribuições de Encarregado de Dados (DPO), é o e-mail{" "}
        <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>.
      </p>

      <h2>2. Dados que coletamos</h2>
      <h3>2.1. Dados de cadastro</h3>
      <ul>
        <li>Endereço de e-mail;</li>
        <li>Senha (armazenada de forma irreversível por algoritmo de hash);</li>
        <li>Nome de exibição (quando fornecido por provedor de autenticação de terceiros).</li>
      </ul>
      <h3>2.2. Dados de autenticação por terceiros</h3>
      <p>
        Quando você opta por entrar com Google, recebemos do Google as informações mínimas necessárias
        para identificar sua conta: endereço de e-mail, identificador exclusivo da conta Google e, se
        autorizado por você no consentimento do Google, nome e imagem de perfil. Não acessamos sua
        senha do Google nem qualquer outro dado de sua conta Google.
      </p>
      <h3>2.3. Dados de uso e técnicos</h3>
      <ul>
        <li>Endereço IP, agente de usuário (navegador) e identificadores de sessão;</li>
        <li>Registros de acesso a aplicações de internet (mantidos por 6 meses, conforme art. 15 do Marco Civil da Internet);</li>
        <li>Metadados de uso (jobs criados, capítulos processados, status de execução, número de páginas, mensagens de erro).</li>
      </ul>
      <h3>2.4. Conteúdo enviado pelo Usuário</h3>
      <p>
        Os arquivos de imagem enviados, os textos extraídos por OCR e as traduções resultantes são
        tratados estritamente para execução do Serviço solicitado por você. Esses dados são
        armazenados em volume técnico vinculado ao seu job e podem ser removidos por você a qualquer
        momento pela interface de exclusão de projeto.
      </p>
      <h3>2.5. Cookies e dados de navegação</h3>
      <p>
        Utilizamos cookies essenciais para manter sua sessão autenticada e, mediante seu consentimento,
        cookies analíticos para entender o uso do site. Detalhes na Política de Cookies.
      </p>

      <h2>3. Finalidades e bases legais</h2>
      <ul>
        <li>
          <strong>Execução de contrato (art. 7º, V, LGPD):</strong> cadastro, autenticação, operação
          do Serviço, processamento de páginas, gestão de créditos e suporte.
        </li>
        <li>
          <strong>Cumprimento de obrigação legal (art. 7º, II, LGPD):</strong> guarda de registros de
          acesso por 6 meses conforme Marco Civil da Internet; atendimento a ordens judiciais ou de
          autoridade competente.
        </li>
        <li>
          <strong>Legítimo interesse (art. 7º, IX, LGPD):</strong> prevenção a fraudes, segurança da
          informação, análises agregadas de uso e melhoria contínua do Serviço, sempre observados os
          direitos e liberdades do titular.
        </li>
        <li>
          <strong>Consentimento (art. 7º, I, LGPD):</strong> cookies analíticos e quaisquer
          comunicações de marketing, quando aplicáveis. O consentimento pode ser revogado a qualquer
          tempo.
        </li>
      </ul>

      <h2>4. Compartilhamento e operadores</h2>
      <p>
        Para operar o Serviço, podemos compartilhar dados pessoais com operadores e prestadores de
        serviço, sob obrigações contratuais de confidencialidade e proteção de dados. Os principais
        operadores são:
      </p>
      <ul>
        <li>
          <strong>Google LLC</strong> — autenticação federada (Login com Google) e, se ativado,
          análise de uso (Google Analytics);
        </li>
        <li>
          <strong>Provedor de hospedagem em nuvem</strong> — infraestrutura para execução da
          aplicação e armazenamento de dados;
        </li>
        <li>
          <strong>Provedores de tradução automática</strong> — quando o Usuário opta por enviar texto
          extraído a serviços externos (ex.: Google Tradutor) para tradução. O Serviço envia apenas o
          texto extraído, nunca a imagem original.
        </li>
      </ul>
      <p>
        Não vendemos dados pessoais. Compartilhamentos com autoridades públicas somente ocorrerão
        mediante requisição legal válida.
      </p>

      <h2>5. Transferência internacional de dados</h2>
      <p>
        Alguns operadores estão localizados fora do Brasil (ex.: Estados Unidos, União Europeia).
        Essas transferências observam as hipóteses do art. 33 da LGPD, incluindo a execução de
        contrato com o titular e a adoção de cláusulas e salvaguardas adequadas.
      </p>

      <h2>6. Retenção</h2>
      <ul>
        <li><strong>Cadastro:</strong> enquanto a conta estiver ativa, mais 12 meses após o encerramento, para fins de cumprimento de obrigações legais e exercício regular de direitos.</li>
        <li><strong>Registros de acesso:</strong> 6 meses (Marco Civil da Internet).</li>
        <li><strong>Conteúdo de jobs (imagens, textos, resultados):</strong> enquanto o projeto existir; remoção imediata após exclusão pelo Usuário.</li>
        <li><strong>Dados financeiros (quando aplicável):</strong> pelo prazo legal exigido pela legislação tributária.</li>
      </ul>

      <h2>7. Direitos do titular (LGPD)</h2>
      <p>Você pode, a qualquer tempo e gratuitamente, solicitar:</p>
      <ul>
        <li>Confirmação da existência de tratamento;</li>
        <li>Acesso aos dados;</li>
        <li>Correção de dados incompletos, inexatos ou desatualizados;</li>
        <li>Anonimização, bloqueio ou eliminação de dados desnecessários, excessivos ou tratados em desconformidade;</li>
        <li>Portabilidade dos dados a outro fornecedor de serviço ou produto;</li>
        <li>Eliminação dos dados tratados com base em consentimento;</li>
        <li>Informação sobre as entidades públicas e privadas com as quais compartilhamos dados;</li>
        <li>Informação sobre a possibilidade de não fornecer consentimento e as consequências;</li>
        <li>Revogação do consentimento.</li>
      </ul>
      <p>
        As solicitações devem ser enviadas para{" "}
        <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a> e serão respondidas em prazo razoável
        (em regra, até 15 dias), podendo exigir comprovação de identidade.
      </p>

      <h2>8. Segurança da informação</h2>
      <p>
        Adotamos medidas técnicas e administrativas razoáveis para proteger dados pessoais contra
        acesso não autorizado, perda, alteração ou divulgação indevida, incluindo controle de acesso,
        criptografia em trânsito (HTTPS), armazenamento seguro de senhas e segregação de ambientes.
        Nenhum sistema é absolutamente imune a incidentes; em caso de incidente de segurança com
        risco relevante, comunicaremos a Autoridade Nacional de Proteção de Dados (ANPD) e os
        titulares afetados, nos termos do art. 48 da LGPD.
      </p>

      <h2>9. Crianças e adolescentes</h2>
      <p>
        O Serviço é destinado a Usuários com idade igual ou superior a 16 (dezesseis) anos. Não
        tratamos intencionalmente dados pessoais de crianças (menores de 12 anos) e não realizamos
        coleta direcionada a esse público. O tratamento de dados pessoais de adolescentes (entre 12 e
        18 anos), quando ocorrer, observará o disposto no art. 14 da LGPD, sempre em seu melhor
        interesse, com as devidas salvaguardas e, quando aplicável, com a participação dos
        responsáveis legais.
      </p>
      <p>
        Caso tomemos conhecimento de tratamento involuntário de dados de criança menor de 12 anos sem
        consentimento específico e em destaque de pelo menos um dos pais ou responsável legal, os
        dados serão eliminados.
      </p>

      <h2>10. Reclamações</h2>
      <p>
        Você tem o direito de apresentar reclamação à Autoridade Nacional de Proteção de Dados (ANPD)
        a respeito do tratamento de seus dados pessoais pelo {SERVICE_NAME}.
      </p>

      <h2>11. Alterações desta Política</h2>
      <p>
        Esta Política pode ser atualizada periodicamente. Alterações materiais serão informadas pelo
        site e, quando apropriado, por e-mail. A data da última atualização consta no topo deste
        documento.
      </p>
    </LegalShell>
  );
}

export function LegalCookiesPage() {
  return (
    <LegalShell title="Política de Cookies">
      <p>
        Esta Política descreve os cookies e tecnologias similares utilizados pelo {SERVICE_NAME} e
        como você pode gerenciá-los.
      </p>

      <h2>1. O que são cookies</h2>
      <p>
        Cookies são pequenos arquivos de texto armazenados em seu dispositivo quando você visita um
        site. Eles permitem que o site se lembre de informações entre páginas, mantenha sessões
        autenticadas e colete estatísticas de uso.
      </p>

      <h2>2. Cookies que utilizamos</h2>
      <h3>2.1. Essenciais (sem necessidade de consentimento)</h3>
      <p>
        Indispensáveis para o funcionamento do Serviço. Sem eles, recursos básicos como manter você
        autenticado não funcionam.
      </p>
      <ul>
        <li><strong>Sessão de autenticação:</strong> identifica seu navegador como pertencente a uma conta autenticada. Cookie httpOnly, com duração de sessão.</li>
        <li><strong>Preferência de consentimento:</strong> registra suas escolhas neste banner de cookies. Duração de 12 meses.</li>
      </ul>

      <h3>2.2. Analíticos (consentimento obrigatório)</h3>
      <p>
        Utilizados, mediante seu consentimento expresso, para coletar dados estatísticos sobre o uso
        do site, ajudando-nos a entender quais páginas são mais acessadas e como melhorar a
        experiência.
      </p>
      <ul>
        <li>
          <strong>Google Analytics (GA4):</strong> cookies do tipo <code>_ga</code> e
          <code>_ga_*</code>, fornecidos pelo Google LLC, com duração padrão de até 24 meses.
          Coletam, entre outros, identificador anônimo de visitante, páginas acessadas, duração da
          sessão e referenciador. A coleta é configurada com anonimização de IP quando tecnicamente
          aplicável.
        </li>
      </ul>

      <h2>3. Como gerenciar</h2>
      <p>
        Você pode aceitar ou rejeitar cookies analíticos a qualquer momento por meio do banner
        exibido na primeira visita e pelo link "Cookies" no rodapé do site. Você também pode bloquear
        ou apagar cookies diretamente nas configurações do seu navegador. Note que a recusa de
        cookies essenciais pode impedir o funcionamento de partes do Serviço.
      </p>

      <h2>4. Bases legais (LGPD)</h2>
      <ul>
        <li><strong>Essenciais:</strong> legítimo interesse e execução de contrato (art. 7º, V e IX, LGPD).</li>
        <li><strong>Analíticos:</strong> consentimento livre, informado e inequívoco (art. 7º, I, LGPD), revogável a qualquer tempo.</li>
      </ul>

      <h2>5. Transferência internacional</h2>
      <p>
        Cookies de terceiros, como os do Google Analytics, podem implicar transferência de dados para
        os Estados Unidos. Mais informações constam da Política de Privacidade.
      </p>
    </LegalShell>
  );
}

export function LegalCopyrightPage() {
  return (
    <LegalShell title="Política de Direitos Autorais e Notificação de Remoção">
      <p>
        O {SERVICE_NAME} respeita os direitos de propriedade intelectual e adota postura ativa de
        cooperação com titulares de direitos. Esta Política descreve nosso posicionamento e o
        procedimento para notificações extrajudiciais, em linha com os artigos 19 e 21 do Marco Civil
        da Internet (Lei nº 12.965/2014) e com a Lei de Direitos Autorais (Lei nº 9.610/1998).
      </p>

      <h2>1. Posicionamento do {SERVICE_NAME}</h2>
      <p>
        O {SERVICE_NAME} é uma ferramenta de software de edição que opera sobre arquivos fornecidos
        pelo próprio Usuário. O Serviço não hospeda obras públicas, não distribui conteúdo a
        terceiros, não opera catálogo, não indexa fontes de obras e não recomenda links para obtenção
        de material protegido.
      </p>
      <p>
        Os arquivos enviados ao Serviço permanecem associados exclusivamente à conta do Usuário que
        os enviou, são utilizados apenas para a execução do processamento solicitado e podem ser
        excluídos pelo Usuário a qualquer momento. O {SERVICE_NAME} não fornece links públicos para
        os resultados.
      </p>

      <h2>2. Declaração do Usuário</h2>
      <p>
        Ao utilizar o Serviço, o Usuário declara, sob as penas da lei, que é titular dos direitos
        sobre o material enviado ou possui todas as autorizações necessárias para utilizá-lo. A
        verificação de titularidade ou licença é de responsabilidade exclusiva do Usuário.
      </p>

      <h2>3. Procedimento de notificação extrajudicial</h2>
      <p>
        Titulares de direitos autorais que entendam haver violação de seus direitos por meio do
        Serviço podem encaminhar notificação extrajudicial fundamentada para o e-mail{" "}
        <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>, contendo, no mínimo:
      </p>
      <ol>
        <li>Identificação completa do notificante (nome ou razão social, CPF/CNPJ e endereço);</li>
        <li>Identificação do titular dos direitos alegadamente violados, se distinto do notificante, com comprovação de representação;</li>
        <li>Identificação clara e específica da obra protegida;</li>
        <li>Indicação precisa do material apontado como infrator (URLs, identificadores de job, capturas de tela ou outros elementos que permitam localização);</li>
        <li>Declaração, sob as penas da lei, de boa-fé na alegação e de titularidade ou autorização para representar o titular;</li>
        <li>Dados de contato para comunicações sobre a notificação.</li>
      </ol>

      <h2>4. Procedimento interno e prazos</h2>
      <p>
        Recebida notificação válida, o {SERVICE_NAME} acusará o recebimento em até 5 (cinco) dias
        úteis e adotará as medidas que entender cabíveis, podendo incluir a remoção do material
        específico associado a contas de Usuário, observados o princípio do devido processo, o
        contraditório e os limites do art. 19 do Marco Civil da Internet, que condiciona a
        responsabilização do provedor de aplicação ao descumprimento de ordem judicial específica.
      </p>
      <p>
        Notificações infundadas, abusivas ou enviadas com má-fé podem sujeitar o notificante às
        sanções civis e penais cabíveis, inclusive nos termos do art. 22 do Marco Civil da Internet.
      </p>

      <h2>5. Reincidência</h2>
      <p>
        Contas de Usuário objeto de notificações reiteradas e procedentes poderão ser suspensas ou
        encerradas, a critério do {SERVICE_NAME}, sem prejuízo de outras medidas legais cabíveis.
      </p>

      <h2>6. Contranotificação</h2>
      <p>
        Usuários cujo material tenha sido removido em razão de notificação podem apresentar
        contranotificação fundamentada pelo mesmo canal, indicando os elementos que demonstram a
        legitimidade do uso. O {SERVICE_NAME} avaliará a contranotificação e poderá reverter a
        medida, caso entenda justificado.
      </p>
    </LegalShell>
  );
}

export function LegalAcceptableUsePage() {
  return (
    <LegalShell title="Política de Uso Aceitável">
      <p>
        Esta Política estabelece condutas vedadas no uso do {SERVICE_NAME} e integra os Termos de
        Uso. O descumprimento pode resultar em suspensão imediata, encerramento da conta e
        comunicação às autoridades competentes.
      </p>

      <h2>1. Condutas absolutamente proibidas</h2>
      <ul>
        <li>Processar, transmitir ou tentar gerar material de abuso ou exploração sexual de crianças e adolescentes (CSAM);</li>
        <li>Processar material que incite ou promova terrorismo, genocídio, tortura ou crimes contra a humanidade;</li>
        <li>Utilizar o Serviço para preparar, distribuir ou facilitar a prática de crimes;</li>
        <li>Violar dolosamente direitos autorais, marcas, patentes, segredo industrial ou direitos da personalidade de terceiros;</li>
        <li>Contornar ou tentar contornar medidas tecnológicas de proteção (DRM), inclusive descriptografar ou processar arquivos cuja proteção tenha sido removida sem autorização;</li>
        <li>Submeter ao Serviço material com objetivo de difamação, calúnia, injúria ou perseguição (stalking) contra pessoa identificada ou identificável.</li>
      </ul>

      <h2>2. Condutas operacionais vedadas</h2>
      <ul>
        <li>Realizar engenharia reversa, descompilação ou tentar extrair código-fonte, modelos ou pesos não disponibilizados publicamente;</li>
        <li>Utilizar bots, scrapers, scripts ou qualquer meio automatizado não autorizado para acessar o Serviço, exceto pelas APIs oficialmente expostas;</li>
        <li>Realizar testes de carga, varredura de vulnerabilidades ou tentativas de invasão sem autorização prévia, expressa e escrita;</li>
        <li>Compartilhar credenciais de acesso, revender contas ou ceder o uso a terceiros sem autorização;</li>
        <li>Burlar limites de créditos, cotas, antifraude ou quaisquer controles do Serviço;</li>
        <li>Utilizar o Serviço de modo que prejudique sua disponibilidade ou segurança para outros Usuários.</li>
      </ul>

      <h2>3. Pesquisa de segurança</h2>
      <p>
        Pesquisadores de segurança que identifiquem vulnerabilidades podem reportá-las de forma
        responsável pelo e-mail <a href={`mailto:${CONTACT_EMAIL}`}>{CONTACT_EMAIL}</a>. O {SERVICE_NAME}
        agradece a colaboração e não tomará medidas legais contra pesquisadores que atuarem de boa-fé,
        sem causar dano, sem acessar dados de terceiros e dentro do escopo de seus próprios testes.
      </p>

      <h2>4. Consequências</h2>
      <p>
        A violação desta Política poderá resultar, isolada ou cumulativamente, em: advertência;
        remoção do material; suspensão ou encerramento da conta sem reembolso; bloqueio de IP; e
        adoção de medidas judiciais cabíveis, incluindo comunicação a autoridades competentes em
        casos previstos em lei.
      </p>
    </LegalShell>
  );
}

export function LegalDisclaimerPage() {
  return (
    <LegalShell title="Isenção de Garantias e Limitação de Responsabilidade">
      <p>
        O presente documento detalha a isenção de garantias e os limites de responsabilidade do{" "}
        {SERVICE_NAME}, e deve ser lido em conjunto com os Termos de Uso.
      </p>

      <h2>1. Natureza experimental dos modelos de IA</h2>
      <p>
        O Serviço utiliza tecnologias de inteligência artificial para reconhecimento de texto,
        tradução automática e processamento de imagem. Esses modelos podem produzir resultados
        incorretos, incompletos, ambíguos, ofensivos ou inadequados ao contexto, inclusive em razão
        de limitações intrínsecas da tecnologia. Os resultados devem ser sempre revisados pelo
        Usuário antes de qualquer utilização final.
      </p>

      <h2>2. Ausência de garantias</h2>
      <p>
        Na máxima extensão permitida pela legislação, o {SERVICE_NAME} fornece o Serviço sem qualquer
        garantia, expressa, implícita, legal ou de outra natureza, incluindo, sem limitação,
        garantias de:
      </p>
      <ul>
        <li>Disponibilidade contínua, ininterrupta ou livre de falhas;</li>
        <li>Adequação a finalidade específica do Usuário;</li>
        <li>Exatidão, integridade ou qualidade artística dos resultados de OCR, tradução e typesetting;</li>
        <li>Preservação fiel de elementos gráficos da obra original;</li>
        <li>Compatibilidade com fluxos editoriais de terceiros;</li>
        <li>Inexistência de bugs, vulnerabilidades ou comportamentos inesperados.</li>
      </ul>

      <h2>3. Backups e perda de dados</h2>
      <p>
        Cabe ao Usuário manter cópias de segurança dos arquivos enviados e dos resultados obtidos. O{" "}
        {SERVICE_NAME} não se responsabiliza por perda de dados decorrente de exclusão pelo próprio
        Usuário, falhas de armazenamento, indisponibilidade do Serviço ou eventos de força maior.
      </p>

      <h2>4. Conteúdo de terceiros</h2>
      <p>
        Quando o Serviço integrar conteúdos ou APIs de terceiros (ex.: enriquecimento de contexto de
        obras a partir de bases públicas), o {SERVICE_NAME} não responde pela exatidão, atualidade,
        legalidade ou disponibilidade desses conteúdos.
      </p>

      <h2>5. Limitação de responsabilidade</h2>
      <p>
        Na máxima extensão permitida pela legislação, o {SERVICE_NAME} não responderá por danos
        indiretos, incidentais, consequenciais, lucros cessantes, perda de receita, perda de
        oportunidade, dano moral ou dano à imagem. A responsabilidade total agregada do{" "}
        {SERVICE_NAME}, em todas as hipóteses, fica limitada ao montante efetivamente pago pelo
        Usuário ao {SERVICE_NAME} nos 3 (três) meses anteriores ao fato gerador, ou ao valor de R$
        100,00 (cem reais), o que for menor.
      </p>

      <h2>6. Direitos do consumidor</h2>
      <p>
        Nada nesta Política ou nos Termos de Uso pretende limitar direitos irrenunciáveis do
        consumidor previstos no Código de Defesa do Consumidor (Lei nº 8.078/1990) ou em outras
        normas cogentes.
      </p>
    </LegalShell>
  );
}
