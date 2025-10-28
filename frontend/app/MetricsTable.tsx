import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../components/ui/table';

interface Metric {
    id: number;
    area_px: number;
    perimeter_px: number;
    circularity: string;
    area_nm: number;
    perimeter_nm: number;
    length_nm: number;
    width_nm: number;
}

interface MetricsTableProps {
    metrics: Metric[];
}

export function MetricsTable({ metrics }: MetricsTableProps) {
    if (metrics.length === 0) {
        return (
            <div className="text-center py-12 text-slate-400">
                No metrics yet.
            </div>
        );
    }

    return (
        <div className="overflow-x-auto">
            <Table>
                <TableHeader>
                    <TableRow>
                        <TableHead>ID</TableHead>
                        <TableHead>Area (px²)</TableHead>
                        <TableHead>Perimeter (px)</TableHead>
                        <TableHead>Circularity</TableHead>
                        <TableHead>Area (nm²)</TableHead>
                        <TableHead>Perimeter (nm)</TableHead>
                        <TableHead>Length (nm)</TableHead>
                        <TableHead>Width (nm)</TableHead>
                    </TableRow>
                </TableHeader>
                <TableBody>
                    {metrics.map((metric) => (
                        <TableRow key={metric.id}>
                            <TableCell>{metric.id}</TableCell>
                            <TableCell>{metric.area_px.toLocaleString()}</TableCell>
                            <TableCell>{metric.perimeter_px.toLocaleString()}</TableCell>
                            <TableCell>{metric.circularity}</TableCell>
                            <TableCell>{metric.area_nm.toLocaleString()}</TableCell>
                            <TableCell>{metric.perimeter_nm.toLocaleString()}</TableCell>
                            <TableCell>{metric.length_nm.toLocaleString()}</TableCell>
                            <TableCell>{metric.width_nm.toLocaleString()}</TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </div>
    );
}
