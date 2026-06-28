import {
    BarChart3,
    Activity,
    LineChart,
    Brain,
    Cloud,
    BarChart2,
    Network,
    Sun,
    Thermometer,
    Building2,
    Calendar,
    Clock,
    Wind
  } from "lucide-react";
  
  export const products = [
    {
      title: "5-Minute Load Prediction",
      description: "Real-time load forecasting with 5-minute intervals for immediate operational decisions.",
      features: [
        "Real-time predictions",
        "5-minute granularity",
        "Immediate response"
      ],
      icon: Clock,
      tags: ["Real-time", "5-Min", "Short-term"]
    },
    {
      title: "Hourly Load Analysis",
      description: "Hour-by-hour load prediction with focus on peak hours and variations.",
      features: [
        "Hourly forecasting",
        "Peak hour analysis",
        "24-hour patterns"
      ],
      icon: Activity,
      tags: ["Hourly", "Peaks", "24-Hour"]
    },
    {
      title: "Weekly Load Forecast",
      description: "Week-long load predictions accounting for weekday patterns and weekend variations.",
      features: [
        "Weekly patterns",
        "Weekend analysis",
        "7-day forecasting"
      ],
      icon: BarChart3,
      tags: ["Weekly", "Patterns", "7-Day"]
    },
    {
      title: "Monthly Load Trends",
      description: "Monthly load forecasting with seasonal pattern recognition and trend analysis.",
      features: [
        "Monthly forecasting",
        "Seasonal patterns",
        "Long-term trends"
      ],
      icon: LineChart,
      tags: ["Monthly", "Seasonal", "Long-term"]
    },
    {
      title: "Weather Impact Analysis",
      description: "Factor weather effects including temperature, humidity, wind speed, and rainfall.",
      features: [
        "Temperature correlation",
        "Humidity impact",
        "Weather patterns"
      ],
      icon: Cloud,
      tags: ["Weather", "Impact", "Analysis"]
    },
    {
      title: "Solar Generation Impact",
      description: "Analyze solar generation patterns and their impact on load predictions.",
      features: [
        "Solar output tracking",
        "Generation patterns",
        "Impact assessment"
      ],
      icon: Sun,
      tags: ["Solar", "Generation", "Impact"]
    },
    {
      title: "Holiday Pattern Analysis",
      description: "Special day load pattern analysis for holidays and events.",
      features: [
        "Holiday impacts",
        "Special events",
        "Pattern adjustment"
      ],
      icon: Calendar,
      tags: ["Holidays", "Events", "Special Days"]
    },
    {
      title: "Seasonal Variation",
      description: "Track load variations between winter (2000 MW) and summer (8300 MW) peaks.",
      features: [
        "Seasonal changes",
        "Peak tracking",
        "Extreme variations"
      ],
      icon: Thermometer,
      tags: ["Seasonal", "Peaks", "Variation"]
    },
    {
      title: "AI Model Integration",
      description: "Comprehensive AI model combining multiple time horizons and factors.",
      features: [
        "Multi-timeframe analysis",
        "Factor integration",
        "Adaptive learning"
      ],
      icon: Brain,
      tags: ["AI", "Integration", "Multi-scale"]
    }
  ];