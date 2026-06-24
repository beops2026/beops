import { Table } from "@/components/ui/table";

export function HourlyTable({ data }) {
  const hourlyData = data.filter(row => row.time.endsWith(':00'));
  
  return (
    <div className="overflow-auto max-h-[350px] bg-zinc-900/50 rounded-lg">
      <Table>
        <thead className="sticky top-0 bg-zinc-900">
          <tr>
            <th className="text-left p-2 text-zinc-400">Hour</th>
            <th className="text-right p-2 text-zinc-400">Load</th>
          </tr>
        </thead>
        <tbody>
          {hourlyData.map((row, index) => (
            <tr 
              key={index}
              className="border-t border-zinc-800 hover:bg-zinc-800/50 transition-colors"
            >
              <td className="p-2 text-left">{row.time}</td>
              <td className="p-2 text-right">{row.load.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </Table>
    </div>
  );
}