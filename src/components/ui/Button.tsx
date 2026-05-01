import { ButtonHTMLAttributes, forwardRef, ReactNode } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger" | "success";
type Size = "sm" | "md" | "lg";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  loading?: boolean;
  fullWidth?: boolean;
}

const base =
  "inline-flex items-center justify-center gap-2 font-medium rounded-md " +
  "transition-all duration-180 ease-out-expo select-none whitespace-nowrap " +
  "disabled:opacity-40 disabled:cursor-not-allowed disabled:pointer-events-none";

const variants: Record<Variant, string> = {
  primary:
    "bg-brand text-white shadow-sm hover:bg-brand-600 active:bg-brand-700 " +
    "hover:shadow-glow-brand",
  secondary:
    "bg-bg-secondary text-text-primary border border-border hover:border-border-strong " +
    "hover:bg-bg-tertiary",
  ghost:
    "bg-transparent text-text-secondary hover:text-text-primary hover:bg-white/5",
  danger:
    "bg-status-error/10 text-status-error border border-status-error/25 " +
    "hover:bg-status-error/15 hover:border-status-error/40",
  success:
    "bg-status-success/10 text-status-success border border-status-success/25 " +
    "hover:bg-status-success/15",
};

const sizes: Record<Size, string> = {
  sm: "h-8 px-3 text-xs",
  md: "h-9 px-4 text-sm",
  lg: "h-11 px-5 text-base",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant = "primary",
    size = "md",
    leftIcon,
    rightIcon,
    loading,
    fullWidth,
    className = "",
    disabled,
    children,
    ...rest
  },
  ref
) {
  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={[
        base,
        variants[variant],
        sizes[size],
        fullWidth ? "w-full" : "",
        className,
      ].join(" ")}
      {...rest}
    >
      {loading ? (
        <span className="inline-block h-3.5 w-3.5 rounded-full border-2 border-current border-t-transparent animate-spin" />
      ) : (
        leftIcon
      )}
      {children}
      {!loading && rightIcon}
    </button>
  );
});
