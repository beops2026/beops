import { useMemo } from 'react';
import { ArrowUpIcon, ArrowDownIcon } from "lucide-react";

export function LoadStats({ data, previousData }) {
  console.log('Current Data:', data);
  console.log('Previous Data:', previousData);

  const stats = useMemo(() => {
    if (!data || data.length === 0) return {
      current: { value: 0, change: 0 },
      peak: { value: 0, change: 0 },
      average: { value: 0, change: 0 }
    };

    // Current period values
    const loads = data.map(item => item.load);
    const current = loads[loads.length - 1];
    const peak = Math.max(...loads);
    const average = Math.round(loads.reduce((a, b) => a + b, 0) / loads.length);

    // Previous period values (if available)
    let currentChange = 0;
    let peakChange = 0;

    if (previousData && previousData.length > 0) {
      const prevLoads = previousData.map(item => item.load);
      const prevCurrent = prevLoads[prevLoads.length - 1];
      const prevPeak = Math.max(...prevLoads);

      // Calculate percentage changes
      currentChange = prevCurrent ? ((current - prevCurrent) / prevCurrent) * 100 : 0;
      peakChange = prevPeak ? ((peak - prevPeak) / prevPeak) * 100 : 0;

      console.log('Calculating changes:', {
        current,
        prevCurrent,
        currentChange,
        peak,
        prevPeak,
        peakChange
      });
    }

    return {
      current: { 
        value: current, 
        change: Number(currentChange.toFixed(1)) 
      },
      peak: { 
        value: peak, 
        change: Number(peakChange.toFixed(1)) 
      },
      average: { 
        value: average, 
        change: null 
      }
    };
  }, [data, previousData]);

  const StatCard = ({ title, value, change, unit = "MW" }) => (
    <div className="bg-zinc-900/50 p-4 rounded-lg">
      <div className="text-zinc-400 text-sm font-medium mb-1">{title}</div>
      <div className="flex items-baseline gap-2">
        <div className="text-2xl font-bold text-white">
          {value.toLocaleString()} <span className="text-sm text-zinc-400">{unit}</span>
        </div>
        {change !== null && (
          <div className={`flex items-center text-sm ${change >= 0 ? 'text-green-500' : 'text-red-500'}`}>
            {change >= 0 ? (
              <ArrowUpIcon className="w-4 h-4 mr-1" />
            ) : (
              <ArrowDownIcon className="w-4 h-4 mr-1" />
            )}
            {Math.abs(change)}%
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="grid grid-cols-4 gap-4 mb-4">
      <StatCard 
        title="Current Load" 
        value={stats.current.value} 
        change={stats.current.change}
      />
      <StatCard 
        title="Peak Load" 
        value={stats.peak.value} 
        change={stats.peak.change}
      />
      <StatCard 
        title="Average Load" 
        value={stats.average.value} 
        change={null}
      />
    </div>
  );
} 