import { Table } from "@/components/ui/table";

export function WeeklyTable({ data }) {
  return (
    <div className="overflow-auto max-h-[350px] bg-zinc-900/50 rounded-lg">
      <Table>
        <thead className="sticky top-0 bg-zinc-900">
          <tr>
            <th className="text-left p-2 text-zinc-400">Date</th>
            <th className="text-right p-2 text-zinc-400">Avg Load</th>
          </tr>
        </thead>
        <tbody>
          {data?.map((row, index) => (
            <tr 
              key={index}
              className="border-t border-zinc-800 hover:bg-zinc-800/50 transition-colors"
            >
              <td className="p-2 text-left font-medium">{row.date}</td>
              <td className="p-2 text-right">{row.load.toLocaleString()}</td>
            </tr>
          )) || (
            <tr>
              <td colSpan={2} className="p-2 text-center text-zinc-500">
                No data available
              </td>
            </tr>
          )}
        </tbody>
      </Table>
    </div>
  );
}