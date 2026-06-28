import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from "recharts";
import { motion } from "framer-motion";

export function LoadChart({ data, viewType, itemVariants }) {
  // Filter and format data based on view type
  const chartData = viewType === 'hourly' 
    ? data.filter(row => row.time.endsWith(':00'))
    : data;

  // Calculate average and peak loads
  const averageLoad = chartData.length > 0
    ? chartData.reduce((sum, item) => sum + parseFloat(item.load), 0) / chartData.length
    : 0;

  const peakLoad = chartData.length > 0
    ? Math.max(...chartData.map(item => parseFloat(item.load)))
    : 0;

  return (
    <motion.div 
      variants={itemVariants}
      className="w-full"
      whileHover={{ scale: 1.01 }}
    >
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#555" />
          <XAxis 
            dataKey="time"
            stroke="#fff"
            angle={viewType === '5min' ? -45 : 0}
            textAnchor="end"
            height={60}
            interval={viewType === '5min' ? 11 : 'preserveStartEnd'}
            tick={{ fill: '#fff' }}
          />
          <YAxis
            stroke="#fff"
            label={{
              value: 'Load (MW)',
              angle: -90,
              position: 'insideLeft',
              fill: '#fff'
            }}
            tick={{ fill: '#fff' }}
            domain={['dataMin - 100', 'dataMax + 100']}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#333', 
              border: 'none',
              borderRadius: '8px',
              boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
              color: '#fff'
            }}
            labelFormatter={(label) => `Time: ${label}`}
            formatter={(value) => [`${parseFloat(value).toFixed(2)} MW`]}
          />
          <Legend />
          
          {/* Average Load Reference Line */}
          <ReferenceLine 
            y={averageLoad} 
            stroke="#ffc658" 
            strokeDasharray="3 3"
            label={{ 
              value: `Avg: ${averageLoad.toFixed(2)} MW`,
              fill: '#ffc658',
              position: 'right'
            }}
          />

          {/* Peak Load Reference Line */}
          <ReferenceLine 
            y={peakLoad} 
            stroke="#ff4d4f" 
            strokeDasharray="3 3"
            label={{ 
              value: `Peak: ${peakLoad.toFixed(2)} MW`,
              fill: '#ff4d4f',
              position: 'right'
            }}
          />
          
          <Line
            type="monotone"
            dataKey="load"
            stroke="#8884d8"
            name="Predicted Load (MW)"
            strokeWidth={2}
            dot={viewType !== '5min'}
            activeDot={{ r: 6, strokeWidth: 0 }}
            isAnimationActive={true}
            animationDuration={1000}
            animationEasing="ease-in-out"
          />
          
          {/* Actual Load Line from Delhi SLDC */}
          <Line
            type="monotone"
            dataKey="actualLoad"
            stroke="#90EE90"
            name="Actual Load (MW)"
            strokeWidth={2}
            dot={viewType !== '5min'}
            activeDot={{ r: 6, strokeWidth: 0 }}
            isAnimationActive={true}
            animationDuration={1000}
            animationEasing="ease-in-out"
            strokeDasharray="5 5"
          />
        </LineChart>
      </ResponsiveContainer>
    </motion.div>
  );
} 