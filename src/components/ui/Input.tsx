import { InputHTMLAttributes, ReactNode, forwardRef } from "react";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  invalid?: boolean;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { leftIcon, rightIcon, invalid, className = "", ...rest },
  ref
) {
  const border = invalid
    ? "border-status-error/50 focus:border-status-error"
    : "border-border hover:border-border-strong focus:border-brand/50";

  return (
    <div className="relative w-full">
      {leftIcon && (
        <span className="absolute left-3 top-1/2 -translate-y-1/2 text-text-secondary/60 pointer-events-none">
          {leftIcon}
        </span>
      )}
      <input
        ref={ref}
        className={[
          "w-full h-10 bg-bg-secondary text-text-primary rounded-md",
          "border transition-colors duration-180",
          "placeholder:text-text-muted focus:outline-none",
          border,
          leftIcon ? "pl-10" : "pl-3.5",
          rightIcon ? "pr-10" : "pr-3.5",
          className,
        ].join(" ")}
        {...rest}
      />
      {rightIcon && (
        <span className="absolute right-3 top-1/2 -translate-y-1/2 text-text-secondary/60">
          {rightIcon}
        </span>
      )}
    </div>
  );
});
