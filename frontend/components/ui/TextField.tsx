"use client";

import {
  forwardRef,
  useId,
  useState,
  type InputHTMLAttributes,
  type ReactNode,
} from "react";
import { cn } from "@/lib/cn";

interface Props extends Omit<InputHTMLAttributes<HTMLInputElement>, "size"> {
  label: string;
  helperText?: string;
  errorText?: string;
  trailing?: ReactNode;
}

/** Material 3 "filled" text field: filled container, animated floating label,
 *  bottom indicator that thickens + turns primary on focus, helper/error line. */
export const TextField = forwardRef<HTMLInputElement, Props>(function TextField(
  { label, helperText, errorText, trailing, className, value, onChange, id, ...props },
  ref,
) {
  const reactId = useId();
  const inputId = id ?? reactId;
  const helpId = `${inputId}-help`;
  const [focused, setFocused] = useState(false);
  const [hasValue, setHasValue] = useState(Boolean(value));
  const invalid = Boolean(errorText);
  const floated = focused || hasValue || Boolean(value);

  return (
    <div className={cn("w-full", className)}>
      <div
        className={cn(
          "relative flex items-stretch rounded-t-md bg-md-surface-container-highest transition-colors",
          "border-b",
          invalid
            ? "border-md-error"
            : focused
              ? "border-md-primary"
              : "border-md-on-surface-variant",
          focused && !invalid && "border-b-2",
          invalid && focused && "border-b-2",
        )}
      >
        <div className="relative flex-1">
          <label
            htmlFor={inputId}
            className={cn(
              "pointer-events-none absolute left-3 origin-left transition-all duration-150 ease-[cubic-bezier(0.2,0,0,1)]",
              floated
                ? "top-1.5 text-xs"
                : "top-1/2 -translate-y-1/2 text-[0.95rem]",
              invalid
                ? "text-md-error"
                : focused
                  ? "text-md-primary"
                  : "text-md-on-surface-variant",
            )}
          >
            {label}
          </label>
          <input
            ref={ref}
            id={inputId}
            value={value}
            aria-invalid={invalid || undefined}
            aria-describedby={helperText || errorText ? helpId : undefined}
            onChange={(e) => {
              setHasValue(e.target.value.length > 0);
              onChange?.(e);
            }}
            onFocus={(e) => {
              setFocused(true);
              props.onFocus?.(e);
            }}
            onBlur={(e) => {
              setFocused(false);
              props.onBlur?.(e);
            }}
            className={cn(
              "peer w-full bg-transparent px-3 pb-1.5 pt-5 text-[0.95rem] text-md-on-surface outline-none",
              "placeholder:text-transparent",
            )}
            {...props}
          />
        </div>
        {trailing && (
          <div className="flex items-center pr-1 text-md-on-surface-variant">{trailing}</div>
        )}
      </div>
      {(helperText || errorText) && (
        <p
          id={helpId}
          className={cn(
            "mt-1 px-3 text-xs",
            invalid ? "text-md-error" : "text-md-on-surface-variant",
          )}
        >
          {errorText ?? helperText}
        </p>
      )}
    </div>
  );
});
