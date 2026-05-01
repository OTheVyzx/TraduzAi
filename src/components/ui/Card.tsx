import { HTMLAttributes, forwardRef, ReactNode } from "react";

type Variant = "default" | "elevated" | "interactive" | "highlight";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  variant?: Variant;
  padding?: "none" | "sm" | "md" | "lg";
  children?: ReactNode;
}

const base = "rounded-lg transition-all duration-180 ease-out-expo";

const variants: Record<Variant, string> = {
  default: "bg-bg-secondary border border-border",
  elevated: "bg-bg-secondary border border-border shadow-md",
  interactive:
    "bg-bg-secondary border border-border hover:border-border-strong " +
    "hover:bg-bg-tertiary cursor-pointer",
  highlight:
    "bg-gradient-to-br from-brand/10 to-brand/5 border border-brand/20",
};

const paddings = {
  none: "",
  sm: "p-3",
  md: "p-4",
  lg: "p-6",
};

export const Card = forwardRef<HTMLDivElement, CardProps>(function Card(
  { variant = "default", padding = "md", className = "", children, ...rest },
  ref
) {
  return (
    <div
      ref={ref}
      className={[base, variants[variant], paddings[padding], className].join(" ")}
      {...rest}
    >
      {children}
    </div>
  );
});
