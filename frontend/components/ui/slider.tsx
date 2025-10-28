import * as React from "react";

type Props = {
  value: number[];
  onValueChange: (v: number[]) => void;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
};

export function Slider({ value, onValueChange, min=0, max=1, step=0.1, className }: Props) {
  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value[0]}
      onChange={(e)=>onValueChange([parseFloat(e.target.value)])}
      className={"w-full " + (className || "")}
    />
  );
}


