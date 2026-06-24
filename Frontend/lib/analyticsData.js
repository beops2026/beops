/**
 * Realistic dummy data for the BEOPS Analytics Dashboard.
 * All values are based on plausible grid substation operational ranges.
 */

// ── 1. Prediction Accuracy Trend (30-day) ─────────────────────────────────────
export const predictionAccuracyData = Array.from({ length: 30 }, (_, i) => {
  const day = new Date(2026, 4, 1 + i); // May 2026
  const base = 96.2 + Math.sin(i * 0.4) * 1.1 + (Math.random() - 0.5) * 0.8;
  return {
    date: day.toLocaleDateString("en-GB", { day: "2-digit", month: "short" }),
    accuracy: parseFloat(Math.min(99.5, Math.max(93, base)).toFixed(2)),
    mape: parseFloat((100 - Math.min(99.5, Math.max(93, base))).toFixed(2)),
  };
});

// ── 2. Feature Importance ─────────────────────────────────────────────────────
export const featureImportanceData = [
  { feature: "Temperature (°C)", importance: 38.4, category: "Weather" },
  { feature: "Hour of Day", importance: 22.1, category: "Time" },
  { feature: "Day of Week", importance: 11.8, category: "Time" },
  { feature: "Humidity (%)", importance: 9.3, category: "Weather" },
  { feature: "Is Holiday", importance: 7.6, category: "Calendar" },
  { feature: "Wind Speed", importance: 5.2, category: "Weather" },
  { feature: "Cloud Cover", importance: 3.8, category: "Weather" },
  { feature: "Dew Point", importance: 1.8, category: "Weather" },
];

// ── 3. Environmental Factors vs Load (time-series) ────────────────────────────
export const environmentalImpactData = Array.from({ length: 24 }, (_, h) => {
  const tempCurve = 28 + 8 * Math.sin((h - 6) * Math.PI / 12);
  const loadBase = 2800 + 1400 * Math.sin((h - 5) * Math.PI / 12);
  const humidity = 65 - 15 * Math.sin((h - 10) * Math.PI / 14);
  return {
    hour: `${String(h).padStart(2, "0")}:00`,
    temperature: parseFloat(tempCurve.toFixed(1)),
    load: Math.round(loadBase + (Math.random() - 0.5) * 200),
    humidity: parseFloat(Math.min(95, Math.max(35, humidity + (Math.random() - 0.5) * 5)).toFixed(1)),
  };
});

// ── 4. Sensor Contribution (proportions) ─────────────────────────────────────
export const sensorContributionData = [
  { name: "BRPL Zone", value: 32.4, color: "#6366f1" },
  { name: "BYPL Zone", value: 24.8, color: "#22d3ee" },
  { name: "NDPL Zone", value: 21.3, color: "#a78bfa" },
  { name: "NDMC Zone", value: 13.7, color: "#f59e0b" },
  { name: "MES Zone", value: 7.8, color: "#34d399" },
];

// ── 5. Model Confidence Over Time ─────────────────────────────────────────────
export const modelConfidenceData = Array.from({ length: 24 }, (_, i) => {
  const base = 94 - i * 0.6 + Math.sin(i * 0.5) * 2;
  const upper = base + 2.5 + Math.random();
  const lower = base - 2 - Math.random();
  return {
    hour: `${String(i).padStart(2, "0")}:00`,
    confidence: parseFloat(Math.max(80, base).toFixed(1)),
    upperBound: parseFloat(Math.min(99, upper).toFixed(1)),
    lowerBound: parseFloat(Math.max(78, lower).toFixed(1)),
  };
});

// ── 6. Temp × Humidity → Load Correlation (scatter) ──────────────────────────
export const correlationData = Array.from({ length: 80 }, () => {
  const temp = 15 + Math.random() * 25;
  const humidity = 30 + Math.random() * 60;
  // Higher temp → higher load, humidity has moderate negative effect at high temp
  const load = 1800 + temp * 90 - humidity * 12 + (Math.random() - 0.5) * 500;
  return {
    temperature: parseFloat(temp.toFixed(1)),
    humidity: parseFloat(humidity.toFixed(1)),
    load: Math.round(Math.max(1200, load)),
  };
});

// ── 7. Resource Utilisation Trends (weekly) ───────────────────────────────────
export const resourceUtilisationData = [
  { day: "Mon", transformer: 78, feeder: 65, bus: 52 },
  { day: "Tue", transformer: 82, feeder: 71, bus: 58 },
  { day: "Wed", transformer: 85, feeder: 74, bus: 61 },
  { day: "Thu", transformer: 80, feeder: 69, bus: 55 },
  { day: "Fri", transformer: 88, feeder: 79, bus: 67 },
  { day: "Sat", transformer: 72, feeder: 58, bus: 44 },
  { day: "Sun", transformer: 65, feeder: 51, bus: 38 },
];

// ── 8. System Health Score Over Time ─────────────────────────────────────────
export const systemHealthData = Array.from({ length: 30 }, (_, i) => {
  const base = 88 + Math.sin(i * 0.3) * 5 + (Math.random() - 0.5) * 3;
  const day = new Date(2026, 4, 1 + i);
  return {
    date: day.toLocaleDateString("en-GB", { day: "2-digit", month: "short" }),
    score: parseFloat(Math.min(99, Math.max(70, base)).toFixed(1)),
    threshold: 80,
  };
});

// ── 9. Risk Factor Analysis (multi-metric for radar) ─────────────────────────
export const riskFactorData = [
  { metric: "Peak Overload", current: 72, threshold: 85 },
  { metric: "Transformer Stress", current: 58, threshold: 80 },
  { metric: "Feeder Loading", current: 65, threshold: 75 },
  { metric: "Voltage Variance", current: 42, threshold: 70 },
  { metric: "Freq. Deviation", current: 31, threshold: 60 },
  { metric: "Reactive Power", current: 55, threshold: 75 },
];

// Radar-formatted version
export const riskRadarData = [
  {
    subject: "Peak Overload",
    current: 72,
    threshold: 85,
    fullMark: 100,
  },
  {
    subject: "Transformer",
    current: 58,
    threshold: 80,
    fullMark: 100,
  },
  {
    subject: "Feeder Load",
    current: 65,
    threshold: 75,
    fullMark: 100,
  },
  {
    subject: "Voltage",
    current: 42,
    threshold: 70,
    fullMark: 100,
  },
  {
    subject: "Frequency",
    current: 31,
    threshold: 60,
    fullMark: 100,
  },
  {
    subject: "Reactive Pwr",
    current: 55,
    threshold: 75,
    fullMark: 100,
  },
];

// ── 10. Historical Prediction vs Actual (30-day) ──────────────────────────────
export const historicalPredictionData = Array.from({ length: 30 }, (_, i) => {
  const actual = 68000 + Math.sin(i * 0.5) * 12000 + (Math.random() - 0.5) * 5000;
  const predicted = actual + (Math.random() - 0.5) * 3000;
  const day = new Date(2026, 3, 1 + i); // April 2026
  return {
    date: day.toLocaleDateString("en-GB", { day: "2-digit", month: "short" }),
    actual: Math.round(actual),
    predicted: Math.round(predicted),
    error: parseFloat(Math.abs((predicted - actual) / actual * 100).toFixed(2)),
  };
});
