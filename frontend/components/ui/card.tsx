import * as React from "react";

export function Card(props: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      data-slot="card"
      className={
        "rounded-2xl border border-border bg-card text-card-foreground shadow-sm " +
        (props.className || "")
      }
      {...props}
    />
  );
}


