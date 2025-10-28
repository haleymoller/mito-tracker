"use client";
import * as React from "react";

type Variant = "default" | "destructive" | "outline" | "secondary" | "ghost" | "link";
type Size = "sm" | "default" | "lg" | "icon";

type Props = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  size?: Size;
};

const variantClasses: Record<Variant, string> = {
  default: "bg-primary text-primary-foreground hover:bg-primary/90",
  destructive: "bg-destructive text-white hover:bg-destructive/90",
  outline: "border bg-background text-foreground hover:bg-accent",
  secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
  ghost: "hover:bg-accent hover:text-accent-foreground",
  link: "text-primary underline-offset-4 hover:underline",
};

const sizeClasses: Record<Size, string> = {
  sm: "h-8 rounded-md px-3",
  default: "h-9 px-4 py-2",
  lg: "h-10 rounded-md px-6",
  icon: "size-9 rounded-md",
};

export function Button({ className, variant = "default", size = "default", ...props }: Props) {
  const classes = [
    "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all disabled:pointer-events-none disabled:opacity-50 outline-none focus-visible:ring-ring/50 focus-visible:ring-[3px]",
    variantClasses[variant],
    sizeClasses[size],
    className || "",
  ].join(" ");
  return <button className={classes} {...props} />;
}

export const buttonVariants = { variant: variantClasses, size: sizeClasses };


