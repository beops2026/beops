import React, { useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from "recharts";

export default function LongTermForecastChart() {
  const longTermData = useMemo(() => {
    // Base load for calculations (in MW)
    const baseLoad = 3500;
    
    // Generate forecast for next 12 months
    const months = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ];

    // Seasonal patterns for Delhi's power consumption
    const seasonalFactors = {
      'Jan': 1.2,  // Winter peak
      'Feb': 1.15,
      'Mar': 1.0,
      'Apr': 0.95,
      'May': 1.1,  // Pre-summer
      'Jun': 1.25, // Summer start
      'Jul': 1.3,  // Peak summer
      'Aug': 1.3,  // Peak summer
      'Sep': 1.2,  // Late summer
      'Oct': 1.0,
      'Nov': 0.9,
      'Dec': 1.1   // Winter start
    };

    // Growth factors
    const annualGrowth = 1.05; // 5% annual growth
    const industrialFactor = 1.02; // 2% industrial growth

    return months.map((month, index) => {
      // Calculate base forecast with seasonal and growth factors
      const seasonalLoad = baseLoad * seasonalFactors[month];
      const growthFactor = Math.pow(annualGrowth, index/12); // Proportional growth
      const industrialGrowth = Math.pow(industrialFactor, index/12);
      
      // Add some variation for peak and off-peak hours
      const peakVariation = 1 + (Math.sin(index * Math.PI / 6) * 0.1);
      
      const forecast = Math.round(
        seasonalLoad * growthFactor * industrialGrowth * peakVariation
      );

      // Calculate confidence bounds (±5% of forecast)
      const upperBound = Math.round(forecast * 1.05);
      const lowerBound = Math.round(forecast * 0.95);

      return {
        month,
        forecast,
        upperBound,
        lowerBound,
        actual: index < 3 ? Math.round(forecast * (0.95 + Math.random() * 0.1)) : null // Show actual only for past months
      };
    });
  }, []);

  return (
    <div className="w-full h-full">
      <h2 className="text-center text-2xl font-bold mb-4 text-white">Long-term Load Forecast</h2>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={longTermData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#555" />
          <XAxis 
            dataKey="month" 
            stroke="#fff"
            tick={{ fill: '#fff' }}
          />
          <YAxis 
            stroke="#fff"
            tick={{ fill: '#fff' }}
            tickFormatter={value => `${(value/1000).toFixed(1)}K`}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#333', 
              border: 'none',
              borderRadius: '8px',
              boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)'
            }}
            formatter={(value) => [`${value.toLocaleString()} MW`]}
          />
          <Legend />
          <Area 
            type="monotone" 
            dataKey="upperBound"
            stroke="transparent"
            fill="#8884d8"
            fillOpacity={0.1}
            name="Confidence Range"
          />
          <Area 
            type="monotone" 
            dataKey="lowerBound"
            stroke="transparent"
            fill="#8884d8"
            fillOpacity={0.1}
            name="Confidence Range"
          />
          <Area 
            type="monotone" 
            dataKey="forecast" 
            stroke="#8884d8" 
            fill="#8884d8"
            fillOpacity={0.3}
            name="Forecast Load"
          />
          <Area 
            type="monotone" 
            dataKey="actual" 
            stroke="#82ca9d" 
            fill="#82ca9d"
            fillOpacity={0.3}
            name="Actual Load"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}