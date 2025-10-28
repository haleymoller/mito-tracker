import * as React from "react";

export function Table({ className, ...props }: React.HTMLAttributes<HTMLTableElement>) {
  return <table className={"w-full text-sm " + (className || "")} {...props} />;
}
export function TableHeader(props: React.HTMLAttributes<HTMLTableSectionElement>) {
  return <thead {...props} />;
}
export function TableBody(props: React.HTMLAttributes<HTMLTableSectionElement>) {
  return <tbody {...props} />;
}
export function TableRow(props: React.HTMLAttributes<HTMLTableRowElement>) {
  return <tr className="border-b" {...props} />;
}
export function TableHead(props: React.ThHTMLAttributes<HTMLTableCellElement>) {
  return <th className="py-2 pr-4 text-left" {...props} />;
}
export function TableCell(props: React.TdHTMLAttributes<HTMLTableCellElement>) {
  return <td className="py-2 pr-4" {...props} />;
}


