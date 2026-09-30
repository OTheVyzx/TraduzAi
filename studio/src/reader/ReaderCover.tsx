import { BookOpen } from "lucide-react";
import { safeRemoteImageUrl } from "./navigateModel";

export function ReaderCover({ src, alt = "", className = "" }: { src?: string | null; alt?: string; className?: string }) {
  const safe = safeRemoteImageUrl(src ?? undefined);
  return <span className={`reader-media-cover ${className}`.trim()}>{safe
    ? <img src={safe} alt={alt} loading="lazy" />
    : <span className="reader-media-cover-placeholder" aria-label={alt || "Capa indisponível"}><BookOpen size={30} /></span>}
  </span>;
}
