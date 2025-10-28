import * as React from "react";

type Props = React.InputHTMLAttributes<HTMLInputElement>;

export function Checkbox({ className, ...props }: Props) {
  return (
    <input
      type="checkbox"
      className={(
        "size-4 rounded border border-border text-primary focus-visible:ring-ring/50 focus-visible:ring-[3px] outline-none " +
        (className || "")
      )}
      {...props}
    />
  );
}


