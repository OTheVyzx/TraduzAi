import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

const STORAGE_KEY = "traduzai_cookie_consent_v1";

type ConsentValue = "accepted" | "rejected";

export function getCookieConsent(): ConsentValue | null {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    if (value === "accepted" || value === "rejected") return value;
    return null;
  } catch {
    return null;
  }
}

export function setCookieConsent(value: ConsentValue) {
  try {
    window.localStorage.setItem(STORAGE_KEY, value);
  } catch {
    /* ignore */
  }
  window.dispatchEvent(new CustomEvent("traduzai:cookie-consent", { detail: value }));
}

export function CookieBanner() {
  const [consent, setConsent] = useState<ConsentValue | null>(() => getCookieConsent());

  useEffect(() => {
    if (consent !== null) return;
    const handler = (event: Event) => {
      const detail = (event as CustomEvent<ConsentValue>).detail;
      if (detail === "accepted" || detail === "rejected") setConsent(detail);
    };
    window.addEventListener("traduzai:cookie-consent", handler);
    return () => window.removeEventListener("traduzai:cookie-consent", handler);
  }, [consent]);

  if (consent !== null) return null;

  return (
    <div className="cookie-banner" role="dialog" aria-live="polite" aria-label="Aviso de cookies">
      <div className="cookie-banner-copy">
        <strong>Cookies</strong>
        <p>
          Usamos cookies essenciais para manter você autenticado e, com seu consentimento, cookies
          analíticos para entender como o site é utilizado. Você pode mudar de ideia depois pelo link
          "Cookies" no rodapé.
        </p>
        <Link to="/legal/cookies">Ler Política de Cookies</Link>
      </div>
      <div className="cookie-banner-actions">
        <button
          type="button"
          className="cookie-btn cookie-btn-ghost"
          onClick={() => {
            setCookieConsent("rejected");
            setConsent("rejected");
          }}
        >
          Recusar analíticos
        </button>
        <button
          type="button"
          className="cookie-btn cookie-btn-primary"
          onClick={() => {
            setCookieConsent("accepted");
            setConsent("accepted");
          }}
        >
          Aceitar todos
        </button>
      </div>
    </div>
  );
}
