import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";

type Metric = {
  id: number;
  area_px?: number;
  perimeter_px?: number;
  circularity?: number | string;
  // Backend optional scale-aware fields (µm and µm²)
  area_um2?: number | null;
  perimeter_um?: number | null;
  length_um?: number | null;
  width_um?: number | null;
  // Legacy nm fields (if any mocks use them)
  area_nm?: number | null;
  perimeter_nm?: number | null;
  length_nm?: number | null;
  width_nm?: number | null;
};

export function MetricsTable({ metrics }: { metrics: Metric[] }) {
  if (metrics.length === 0) {
    return (
      <div className="text-center py-12 text-slate-400">
        No metrics yet.
      </div>
    );
  }

  const fmt = (v: unknown, digits = 3) =>
    typeof v === "number" && isFinite(v)
      ? v.toLocaleString(undefined, { maximumFractionDigits: digits })
      : "-";

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>ID</TableHead>
            <TableHead>Area (px²)</TableHead>
            <TableHead>Perimeter (px)</TableHead>
            <TableHead>Circularity</TableHead>
            <TableHead>Area (µm²)</TableHead>
            <TableHead>Perimeter (µm)</TableHead>
            <TableHead>Length (µm)</TableHead>
            <TableHead>Width (µm)</TableHead>
            <TableHead>Number of mitos</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {metrics.map((metric) => (
            <TableRow key={metric.id}>
              <TableCell>{metric.id}</TableCell>
              <TableCell>{fmt(metric.area_px)}</TableCell>
              <TableCell>{fmt(metric.perimeter_px)}</TableCell>
              <TableCell>
                {typeof metric.circularity === "number"
                  ? metric.circularity.toFixed(3)
                  : metric.circularity ?? "-"}
              </TableCell>
              <TableCell>{fmt(metric.area_um2)}</TableCell>
              <TableCell>{fmt(metric.perimeter_um)}</TableCell>
              <TableCell>{fmt(metric.length_um)}</TableCell>
              <TableCell>{fmt(metric.width_um)}</TableCell>
              <TableCell>{metrics.length}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}


