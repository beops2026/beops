"use client";

import { useState } from "react";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceArea,
  Cell,
  ResponsiveContainer,
  PieChart,
  Pie,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ScatterChart,
  Scatter,
  Brush,
} from "recharts";
import { motion, AnimatePresence } from "framer-motion";
import {
  TrendingUp,
  BarChart3,
  LineChart as LineIcon,
  AreaChart as AreaIcon,
  PieChart as PieIcon,
  Radar as RadarIcon,
  Crosshair,
  Layers,
  Activity,
} from "lucide-react";
import {
  formatEpochLabel,
  formatEpochFull,
  getValueUnit,
} from "@/lib/forecast";
import { ChartSkeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

// ─── Per-type theme ────────────────────────────────────────────────────────────
const THEMES = {
  hourly: { color: "#6366f1", gradientId: "grad-hourly", label: "MW" },
  weekly: { color: "#22d3ee", gradientId: "grad-weekly", label: "MW" },
  monthly: { color: "#a78bfa", gradientId: "grad-monthly", label: "MWh" },
  quarterly: { color: "#f59e0b", gradientId: "grad-quarterly", label: "MWh" },
};

// ─── Allowed chart types per forecast type ─────────────────────────────────────
// Provide a wide range of options (Line, Bar, Area, Composed, Stepped, Scatter, Pie, Radar)
// customized to each forecast type's semantics:
const ALLOWED_CHART_TYPES = {
  hourly: ["area", "line", "bar", "composed", "stepped", "scatter"],
  weekly: ["bar", "line", "area", "composed", "pie", "radar"],
  monthly: ["line", "area", "bar", "composed", "stepped"],
  quarterly: ["bar", "pie", "line", "composed"],
};

const CHART_TYPE_META = {
  area: {
    icon: AreaIcon,
    label: "Area",
    tooltip: "Area chart — volume & trend",
  },
  line: {
    icon: LineIcon,
    label: "Line",
    tooltip: "Line chart — precise trend",
  },
  bar: {
    icon: BarChart3,
    label: "Bar",
    tooltip: "Bar chart — comparative value",
  },
  composed: {
    icon: Layers,
    label: "Composed",
    tooltip: "Composed chart — bar & line hybrid",
  },
  stepped: {
    icon: Activity,
    label: "Stepped",
    tooltip: "Stepped line chart — discrete stages",
  },
  scatter: {
    icon: Crosshair,
    label: "Scatter",
    tooltip: "Scatter plot — data density",
  },
  pie: { icon: PieIcon, label: "Pie", tooltip: "Pie chart — total shares" },
  radar: {
    icon: RadarIcon,
    label: "Radar",
    tooltip: "Radar chart — cycle profile",
  },
};

// Explanations for each chart type and forecast type combination
const CHART_EXPLANATIONS = {
  hourly: {
    area: "Area view highlights continuous intraday load profile and volume.",
    line: "Line view tracks precise MW load changes hour-by-hour.",
    bar: "Bar view shows individual hourly load comparisons side-by-side.",
    composed:
      "Composed view overlays intraday load volume with a smooth trendline.",
    stepped: "Stepped view displays discrete load changes at hourly intervals.",
    scatter:
      "Scatter view shows the distribution density of MW load observations.",
  },
  weekly: {
    bar: "Bar view compares relative aggregate energy across all 7 days.",
    line: "Line view shows the daily trend over the weekly cycle.",
    area: "Area view details cumulative daily energy consumption over the week.",
    composed: "Composed view combines daily total bars with a cycle trendline.",
    pie: "Pie view highlights each day's percentage share of the weekly total.",
    radar:
      "Radar view illustrates the load distribution pattern over the 7-day cycle.",
  },
  monthly: {
    line: "Line view outlines the continuous load trend over the 30-day period.",
    area: "Area view shows volume distribution of energy load over the month.",
    bar: "Bar view plots daily load totals for comparison (best on wide screens).",
    composed: "Composed view combines daily bars with a 30-day moving trend.",
    stepped: "Stepped view tracks daily load shifts as discrete steps.",
  },
  quarterly: {
    bar: "Bar view compares discrete monthly aggregate totals for the quarter.",
    pie: "Pie view visualizes the percentage breakdown of consumption by month.",
    line: "Line view traces monthly trend aggregates across the quarter.",
    composed: "Composed view combines monthly aggregate bars with a trendline.",
  },
};

// ─── Gradient Defs ────────────────────────────────────────────────────────────
function GradientDefs() {
  return (
    <defs>
      {Object.values(THEMES).map(({ gradientId, color }) => (
        <linearGradient
          key={gradientId}
          id={gradientId}
          x1="0"
          y1="0"
          x2="0"
          y2="1"
        >
          <stop offset="5%" stopColor={color} stopOpacity={0.4} />
          <stop offset="95%" stopColor={color} stopOpacity={0.02} />
        </linearGradient>
      ))}
    </defs>
  );
}

// ─── Custom Tooltip ───────────────────────────────────────────────────────────
function ForecastTooltip({ active, payload, forecastType }) {
  if (!active || !payload?.length) return null;
  const { time, value } = payload[0].payload;
  const unit = getValueUnit(forecastType);
  return (
    <div className="rounded-xl bg-[#0d0d14]/95 border border-white/10 px-4 py-3 shadow-2xl backdrop-blur-md text-xs">
      <p className="text-zinc-400 mb-1.5">
        {formatEpochFull(time, forecastType)}
      </p>
      <p className="text-lg font-bold text-white">
        {value?.toLocaleString(undefined, { maximumFractionDigits: 2 })}
        <span className="text-sm font-normal text-zinc-400 ml-1">{unit}</span>
      </p>
    </div>
  );
}

// ─── Common axis props ────────────────────────────────────────────────────────
const axisStyle = { fill: "#6b7280", fontSize: 11 };
const axisProps = { tick: axisStyle, tickLine: false, axisLine: false };

function yFormatter(v) {
  const abs = Math.abs(v);
  if (abs >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) {
    const k = v / 1_000;
    // Keep one decimal below 100k so neighbouring ticks stay distinct
    // (e.g. 1.8k / 2.3k / 2.9k instead of all collapsing to "2k").
    return `${abs >= 100_000 ? Math.round(k) : k.toFixed(1)}k`;
  }
  return `${Math.round(v)}`;
}

// ─── Chart Renderers ──────────────────────────────────────────────────────────
function renderChart({ chartType, data, forecastType }) {
  const { color, gradientId } = THEMES[forecastType] || THEMES.hourly;
  const xKey = "time";
  const tickFmt = (t) => formatEpochLabel(t, forecastType);
  const interval =
    forecastType === "hourly"
      ? 2
      : forecastType === "monthly"
        ? Math.max(1, Math.floor(data.length / 12))
        : "preserveStartEnd";

  const avgValue = data.reduce((s, d) => s + d.value, 0) / data.length;
  const maxValue = Math.max(...data.map((d) => d.value));

  const common = {
    data,
    margin: { top: 10, right: 16, left: 0, bottom: 0 },
  };

  // Peak load risk zone (top 10% threshold)
  const riskThreshold = maxValue * 0.9;
  const isTimeSeries = ["area", "line", "bar", "composed", "stepped"].includes(
    chartType,
  );

  const referenceAreaAndLine = isTimeSeries ? (
    <>
      <ReferenceArea
        y1={riskThreshold}
        y2={maxValue * 1.05}
        fill="rgba(244, 63, 94, 0.03)"
        stroke="none"
      />
      <ReferenceLine
        y={riskThreshold}
        stroke="#f43f5e"
        strokeDasharray="3 3"
        opacity={0.3}
        label={{
          value: "Peak Risk Zone",
          fill: "#f43f5e",
          fontSize: 9,
          position: "insideBottomRight",
        }}
      />
    </>
  ) : null;

  // Conditionally show brush control for large datasets
  const showBrush = data.length > 20 && isTimeSeries;
  const brush = showBrush ? (
    <Brush
      dataKey={xKey}
      height={20}
      stroke={`${color}30`}
      fill="#0d0d14"
      tickFormatter={tickFmt}
      travellerWidth={8}
    />
  ) : null;

  // Composed data with a 3-point rolling moving average trend line
  const composedData = data.map((d, i) => {
    const start = Math.max(0, i - 1);
    const end = Math.min(data.length - 1, i + 1);
    const subset = data.slice(start, end + 1);
    const avg =
      subset.reduce((sum, item) => sum + item.value, 0) / subset.length;
    return {
      ...d,
      movingAvg: avg,
    };
  });

  if (chartType === "area") {
    return (
      <AreaChart {...common}>
        <GradientDefs />
        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff06" />
        <XAxis
          dataKey={xKey}
          tickFormatter={tickFmt}
          {...axisProps}
          interval={interval}
        />
        <YAxis
          {...axisProps}
          tickFormatter={yFormatter}
          width={55}
          domain={["dataMin - 5%", "dataMax + 5%"]}
        />
        <Tooltip content={<ForecastTooltip forecastType={forecastType} />} />
        <ReferenceLine
          y={avgValue}
          stroke={`${color}50`}
          strokeDasharray="4 4"
          label={{ value: "Avg", fill: color, fontSize: 10, position: "right" }}
        />
        {referenceAreaAndLine}
        <Area
          type="monotone"
          dataKey="value"
          stroke={color}
          strokeWidth={2.5}
          fill={`url(#${gradientId})`}
          dot={false}
          activeDot={{ r: 5, fill: color, stroke: "#fff", strokeWidth: 2 }}
          animationDuration={600}
        />
        {brush}
      </AreaChart>
    );
  }

  if (chartType === "line") {
    return (
      <LineChart {...common}>
        <GradientDefs />
        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff06" />
        <XAxis
          dataKey={xKey}
          tickFormatter={tickFmt}
          {...axisProps}
          interval={interval}
        />
        <YAxis
          {...axisProps}
          tickFormatter={yFormatter}
          width={55}
          domain={["dataMin - 5%", "dataMax + 5%"]}
        />
        <Tooltip content={<ForecastTooltip forecastType={forecastType} />} />
        <ReferenceLine
          y={avgValue}
          stroke={`${color}40`}
          strokeDasharray="4 4"
          label={{ value: "Avg", fill: color, fontSize: 10, position: "right" }}
        />
        {referenceAreaAndLine}
        <Line
          type="monotone"
          dataKey="value"
          stroke={color}
          strokeWidth={2.5}
          dot={false}
          activeDot={{ r: 5, fill: color, stroke: "#fff", strokeWidth: 2 }}
          animationDuration={600}
        />
        {brush}
      </LineChart>
    );
  }

  if (chartType === "bar") {
    return (
      <BarChart {...common} barCategoryGap="22%">
        <GradientDefs />
        <CartesianGrid
          strokeDasharray="3 3"
          stroke="#ffffff06"
          vertical={false}
        />
        <XAxis
          dataKey={xKey}
          tickFormatter={tickFmt}
          {...axisProps}
          interval={interval}
        />
        <YAxis
          {...axisProps}
          tickFormatter={yFormatter}
          width={55}
          domain={["dataMin - 5%", "dataMax + 5%"]}
        />
        <Tooltip content={<ForecastTooltip forecastType={forecastType} />} />
        {referenceAreaAndLine}
        <Bar dataKey="value" radius={[5, 5, 0, 0]} animationDuration={600}>
          {data.map((entry, i) => (
            <Cell
              key={i}
              fill={entry.value === maxValue ? color : `${color}70`}
            />
          ))}
        </Bar>
        {brush}
      </BarChart>
    );
  }

  if (chartType === "composed") {
    return (
      <ComposedChart data={composedData} margin={common.margin}>
        <GradientDefs />
        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff06" />
        <XAxis
          dataKey={xKey}
          tickFormatter={tickFmt}
          {...axisProps}
          interval={interval}
        />
        <YAxis
          {...axisProps}
          tickFormatter={yFormatter}
          width={55}
          domain={["dataMin - 5%", "dataMax + 5%"]}
        />
        <Tooltip content={<ForecastTooltip forecastType={forecastType} />} />
        <ReferenceLine
          y={avgValue}
          stroke={`${color}40`}
          strokeDasharray="4 4"
          label={{ value: "Avg", fill: color, fontSize: 10, position: "right" }}
        />
        {referenceAreaAndLine}
        <Bar
          dataKey="value"
          fill={`${color}25`}
          radius={[3, 3, 0, 0]}
          barSize={20}
        />
        <Line
          type="monotone"
          dataKey="movingAvg"
          stroke={color}
          strokeWidth={2.5}
          dot={false}
          activeDot={{ r: 5, fill: color, stroke: "#fff", strokeWidth: 2 }}
          animationDuration={600}
        />
        {brush}
      </ComposedChart>
    );
  }

  if (chartType === "stepped") {
    return (
      <AreaChart {...common}>
        <GradientDefs />
        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff06" />
        <XAxis
          dataKey={xKey}
          tickFormatter={tickFmt}
          {...axisProps}
          interval={interval}
        />
        <YAxis
          {...axisProps}
          tickFormatter={yFormatter}
          width={55}
          domain={["dataMin - 5%", "dataMax + 5%"]}
        />
        <Tooltip content={<ForecastTooltip forecastType={forecastType} />} />
        {referenceAreaAndLine}
        <Area
          type="stepAfter"
          dataKey="value"
          stroke={color}
          strokeWidth={2.5}
          fill={`url(#${gradientId})`}
          dot={data.length <= 30}
          activeDot={{ r: 5, fill: color, stroke: "#fff", strokeWidth: 2 }}
          animationDuration={600}
        />
        {brush}
      </AreaChart>
    );
  }

  if (chartType === "scatter") {
    return (
      <ScatterChart {...common}>
        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff06" />
        <XAxis
          type="category"
          dataKey={xKey}
          tickFormatter={tickFmt}
          {...axisProps}
          interval={interval}
        />
        <YAxis
          type="number"
          dataKey="value"
          {...axisProps}
          tickFormatter={yFormatter}
          width={55}
          domain={["dataMin - 5%", "dataMax + 5%"]}
        />
        <Tooltip content={<ForecastTooltip forecastType={forecastType} />} />
        <Scatter name="Load" data={data} fill={color} animationDuration={600} />
      </ScatterChart>
    );
  }

  if (chartType === "pie") {
    const pieData = data.map((d) => ({
      name: tickFmt(d.time),
      value: d.value,
    }));
    const COLORS = [
      color,
      "#6366f1",
      "#a78bfa",
      "#22d3ee",
      "#f59e0b",
      "#10b981",
      "#ec4899",
      "#f43f5e",
    ];
    return (
      <PieChart margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
        <Tooltip
          formatter={(value, name) => [
            `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })} ${getValueUnit(forecastType)}`,
            name,
          ]}
          contentStyle={{
            backgroundColor: "#0d0d14",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 12,
          }}
          itemStyle={{ color: "#fff" }}
        />
        <Pie
          data={pieData}
          cx="50%"
          cy="48%"
          innerRadius={60}
          outerRadius={100}
          paddingAngle={2}
          dataKey="value"
        >
          {pieData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Legend
          verticalAlign="bottom"
          height={36}
          iconType="circle"
          wrapperStyle={{ fontSize: 11, color: "#9ca3af" }}
        />
      </PieChart>
    );
  }

  if (chartType === "radar") {
    const radarData = data.map((d) => ({
      subject: tickFmt(d.time),
      value: d.value,
    }));
    return (
      <RadarChart cx="50%" cy="50%" outerRadius="75%" data={radarData}>
        <PolarGrid stroke="rgba(255,255,255,0.08)" />
        <PolarAngleAxis
          dataKey="subject"
          tick={{ fill: "#9ca3af", fontSize: 11 }}
        />
        <PolarRadiusAxis
          angle={30}
          domain={[0, maxValue]}
          tick={{ fill: "#6b7280", fontSize: 9 }}
        />
        <Radar
          name="Load"
          dataKey="value"
          stroke={color}
          fill={color}
          fillOpacity={0.25}
        />
        <Tooltip
          formatter={(value) => [
            `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })} ${getValueUnit(forecastType)}`,
            "Load",
          ]}
          contentStyle={{
            backgroundColor: "#0d0d14",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 12,
          }}
          itemStyle={{ color: "#fff" }}
        />
      </RadarChart>
    );
  }

  return null;
}

// ─── Chart Type Selector ──────────────────────────────────────────────────────
function ChartTypeSelector({ allowed, active, onChange }) {
  return (
    <div className="flex flex-wrap items-center gap-1 p-1 rounded-lg bg-white/[0.04] border border-white/8">
      {allowed.map((type) => {
        const { icon: Icon, label, tooltip } = CHART_TYPE_META[type];
        const isActive = active === type;
        return (
          <button
            key={type}
            title={tooltip}
            onClick={() => onChange(type)}
            className={cn(
              "flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-all duration-200",
              isActive
                ? "bg-white/10 text-white"
                : "text-zinc-500 hover:text-zinc-300 hover:bg-white/[0.04]",
            )}
          >
            <Icon className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">{label}</span>
          </button>
        );
      })}
    </div>
  );
}

// ─── Empty State ──────────────────────────────────────────────────────────────
function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-64 gap-4 text-zinc-500">
      <div className="w-16 h-16 rounded-2xl bg-white/[0.03] border border-white/8 flex items-center justify-center">
        <BarChart3 className="w-7 h-7 opacity-30" />
      </div>
      <div className="text-center">
        <p className="text-sm font-medium text-zinc-400">
          No forecast data yet
        </p>
        <p className="text-xs text-zinc-600 mt-1">
          Select a date range and click Generate Forecast
        </p>
      </div>
    </div>
  );
}

// ─── Main Export ──────────────────────────────────────────────────────────────
/**
 * ForecastChart — universal chart with per-type dynamic view switching.
 * Allowed chart types depend on the forecast type (time-series → Area/Line/Bar).
 */
export default function ForecastChart({ data, forecastType, loading }) {
  const allowed = ALLOWED_CHART_TYPES[forecastType] || ["line", "area", "bar"];
  const [chartType, setChartType] = useState(allowed[0]);

  // When forecast type changes, reset to that type's default
  const currentAllowed = ALLOWED_CHART_TYPES[forecastType] || [
    "line",
    "area",
    "bar",
  ];
  const activeType = currentAllowed.includes(chartType)
    ? chartType
    : currentAllowed[0];

  if (loading)
    return (
      <div className="p-2">
        <ChartSkeleton />
      </div>
    );
  if (!data || data.length === 0) return <EmptyState />;

  const theme = THEMES[forecastType] || THEMES.hourly;
  const unit = getValueUnit(forecastType);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
    >
      {/* Chart type controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <div
              className="w-2 h-2 rounded-full"
              style={{ background: theme.color }}
            />
            <span className="text-xs text-zinc-400">
              {data.length} data point{data.length !== 1 ? "s" : ""} · {unit}
            </span>
          </div>
          {CHART_EXPLANATIONS[forecastType]?.[activeType] && (
            <span className="text-xs text-zinc-500 italic">
              {CHART_EXPLANATIONS[forecastType][activeType]}
            </span>
          )}
        </div>
        {currentAllowed.length > 1 && (
          <ChartTypeSelector
            allowed={currentAllowed}
            active={activeType}
            onChange={setChartType}
          />
        )}
      </div>

      {/* Animated chart area */}
      <AnimatePresence mode="wait">
        <motion.div
          key={`${forecastType}-${activeType}`}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.2 }}
        >
          <ResponsiveContainer width="100%" height={320}>
            {renderChart({ chartType: activeType, data, forecastType })}
          </ResponsiveContainer>
        </motion.div>
      </AnimatePresence>
    </motion.div>
  );
}
