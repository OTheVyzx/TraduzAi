import { HTMLAttributes, ReactNode } from "react";

type Tone = "neutral" | "brand" | "success" | "warning" | "error" | "info";

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: Tone;
  size?: "sm" | "md";
  icon?: ReactNode;
  children?: ReactNode;
}

const tones: Record<Tone, string> = {
  neutral: "bg-white/5 text-text-secondary border border-border",
  brand: "bg-brand/10 text-brand-300 border border-brand/25",
  success: "bg-status-success/10 text-status-success border border-status-success/25",
  warning: "bg-status-warning/10 text-status-warning border border-status-warning/25",
  error: "bg-status-error/10 text-status-error border border-status-error/25",
  info: "bg-status-info/10 text-status-info border border-status-info/25",
};

const sizes = {
  sm: "px-1.5 py-0.5 text-2xs gap-1",
  md: "px-2 py-0.5 text-xs gap-1.5",
};

export function Badge({
  tone = "neutral",
  size = "md",
  icon,
  className = "",
  children,
  ...rest
}: BadgeProps) {
  return (
    <span
      className={[
        "inline-flex items-center rounded-pill font-medium tracking-normal",
        tones[tone],
        sizes[size],
        className,
      ].join(" ")}
      {...rest}
    >
      {icon}
      {children}
    </span>
  );
}
