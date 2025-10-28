import * as React from "react";

type Props = React.InputHTMLAttributes<HTMLInputElement>;

export function Input({ className, ...props }: Props) {
  return (
    <input
      className={
        "w-full rounded-lg border border-border bg-input-background px-3 py-2 text-sm outline-none focus-visible:ring-ring/50 focus-visible:ring-[3px] " +
        (className || "")
      }
      {...props}
    />
  );
}


