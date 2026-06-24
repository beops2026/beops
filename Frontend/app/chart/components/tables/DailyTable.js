import { Table } from "@/components/ui/table";

export function DailyTable({ data }) {
  return (
    <div className="overflow-auto max-h-[350px] bg-zinc-900/50 rounded-lg">
      <Table>
        <thead className="sticky top-0 bg-zinc-900">
          <tr>
            <th className="text-left p-2 text-white">Time</th>
            <th className="text-right p-2 text-white">Load (MW)</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, index) => (
            <tr 
              key={index}
              className="border-t border-zinc-800 hover:bg-zinc-800/50 transition-colors text-white"
            >
              <td className="p-2 text-left">{row.time}</td>
              <td className="p-2 text-right">{parseFloat(row.load).toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
              })}</td>
            </tr>
          ))}
        </tbody>
      </Table>
    </div>
  );
} 