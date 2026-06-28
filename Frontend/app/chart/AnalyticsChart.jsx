"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LineChart, Line,
  BarChart, Bar,
  AreaChart, Area,
  ComposedChart,
  ScatterChart, Scatter, ZAxis,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer,
} from "recharts";
import {
  LineChart as LineIcon,
  BarChart3,
  TrendingUp,
  PieChart as PieIcon,
  Target,
  Crosshair,
  Layers,
  Activity,
} from "lucide-react";
import { cn } from "@/lib/utils";

// ─── Chart Type Icons ─────────────────────────────────────────────────────────
const TYPE_ICONS = {
  line: LineIcon,
  bar: BarChart3,
  area: TrendingUp,
  pie: PieIcon,
  radar: Target,
  scatter: Crosshair,
  composed: Layers,
  stepped: Activity,
};

const TYPE_LABELS = {
  line: "Line",
  bar: "Bar",
  area: "Area",
  pie: "Donut",
  radar: "Radar",
  scatter: "Scatter",
  composed: "Composed",
  stepped: "Stepped",
};

// ─── Custom Tooltip ───────────────────────────────────────────────────────────
function CustomTooltip({ active, payload, label, valueFormatter, config }) {
  if (!active || !payload?.length) return null;
  // When a series is plotted as a normalized value, config.tooltipRaw maps its
  // dataKey -> { key, unit } so we can show the real underlying value on hover.
  const rawMap = config?.tooltipRaw;
  return (
    <div className="rounded-xl bg-[#0d0d14]/95 border border-white/10 px-3 py-2.5 shadow-2xl backdrop-blur-md text-xs">
      {label && <p className="text-zinc-400 mb-1.5 font-medium">{label}</p>}
      {payload.map((p) => {
        let display;
        const raw = rawMap?.[p.dataKey];
        if (raw && p.payload?.[raw.key] != null) {
          const v = p.payload[raw.key];
          display = `${typeof v === "number" ? v.toLocaleString() : v}${raw.unit ? ` ${raw.unit}` : ""}`;
        } else {
          display = valueFormatter ? valueFormatter(p.value) : p.value;
        }
        return (
          <div key={p.dataKey} className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full" style={{ background: p.color }} />
            <span className="text-zinc-400">{p.name}:</span>
            <span className="text-white font-semibold">{display}</span>
          </div>
        );
      })}
    </div>
  );
}

// ─── Chart Renderer ───────────────────────────────────────────────────────────
function ChartRenderer({ type, data, config }) {
  const { colors = ["#6366f1", "#22d3ee", "#a78bfa", "#f59e0b", "#34d399"] } = config;

  const commonAxis = {
    tick: { fill: "#6b7280", fontSize: 10 },
    tickLine: false,
    axisLine: false,
  };
  const grid = <CartesianGrid strokeDasharray="3 3" stroke="#ffffff06" />;
  const tooltip = <Tooltip content={<CustomTooltip valueFormatter={config.valueFormatter} config={config} />} />;

  if (type === "line") {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
          {grid}
          <XAxis dataKey={config.xKey} {...commonAxis} interval={config.xInterval ?? "preserveStartEnd"} />
          <YAxis {...commonAxis} tickFormatter={config.yFormatter} width={50} domain={['dataMin - 5%', 'dataMax + 5%']} />
          {tooltip}
          <Legend wrapperStyle={{ fontSize: 10, color: "#9ca3af" }} />
          {config.lines.map((l, i) => (
            <Line
              key={l.key}
              type="monotone"
              dataKey={l.key}
              name={l.label}
              stroke={colors[i]}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, stroke: "#fff", strokeWidth: 1.5 }}
              animationDuration={600}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    );
  }

  if (type === "area") {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
          <defs>
            {config.lines.map((l, i) => (
              <linearGradient key={l.key} id={`grad-${l.key}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={colors[i]} stopOpacity={0.35} />
                <stop offset="95%" stopColor={colors[i]} stopOpacity={0.02} />
              </linearGradient>
            ))}
          </defs>
          {grid}
          <XAxis dataKey={config.xKey} {...commonAxis} interval={config.xInterval ?? "preserveStartEnd"} />
          <YAxis {...commonAxis} tickFormatter={config.yFormatter} width={50} domain={['dataMin - 5%', 'dataMax + 5%']} />
          {tooltip}
          <Legend wrapperStyle={{ fontSize: 10, color: "#9ca3af" }} />
          {config.lines.map((l, i) => (
            <Area
              key={l.key}
              type="monotone"
              dataKey={l.key}
              name={l.label}
              stroke={colors[i]}
              fill={`url(#grad-${l.key})`}
              strokeWidth={2}
              dot={false}
              animationDuration={600}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    );
  }

  if (type === "bar") {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 0 }} barCategoryGap="25%">
          {grid}
          <XAxis dataKey={config.xKey} {...commonAxis} interval={config.xInterval ?? "preserveStartEnd"} />
          <YAxis {...commonAxis} tickFormatter={config.yFormatter} width={50} domain={['dataMin - 5%', 'dataMax + 5%']} />
          {tooltip}
          <Legend wrapperStyle={{ fontSize: 10, color: "#9ca3af" }} />
          {config.lines.map((l, i) => (
            <Bar
              key={l.key}
              dataKey={l.key}
              name={l.label}
              fill={colors[i]}
              fillOpacity={0.85}
              radius={[4, 4, 0, 0]}
              animationDuration={500}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    );
  }

  if (type === "pie") {
    const pieData = config.pieKey
      ? data
      : data.map((d) => ({ name: d[config.xKey], value: d[config.lines[0].key] }));
    return (
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={pieData}
            cx="50%"
            cy="50%"
            innerRadius="55%"
            outerRadius="80%"
            dataKey="value"
            nameKey="name"
            paddingAngle={3}
            animationDuration={600}
          >
            {pieData.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.color || colors[i % colors.length]}
                stroke="transparent"
              />
            ))}
          </Pie>
          {tooltip}
          <Legend
            formatter={(v) => <span style={{ color: "#9ca3af", fontSize: 11 }}>{v}</span>}
          />
        </PieChart>
      </ResponsiveContainer>
    );
  }

  if (type === "radar") {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={data} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
          <PolarGrid stroke="#ffffff08" />
          <PolarAngleAxis
            dataKey={config.radarAngleKey || config.xKey}
            tick={{ fill: "#9ca3af", fontSize: 10 }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 100]}
            tick={{ fill: "#6b7280", fontSize: 9 }}
          />
          {tooltip}
          <Legend wrapperStyle={{ fontSize: 10, color: "#9ca3af" }} />
          {config.lines.map((l, i) => (
            <Radar
              key={l.key}
              name={l.label}
              dataKey={l.key}
              stroke={colors[i]}
              fill={colors[i]}
              fillOpacity={0.18}
              animationDuration={600}
            />
          ))}
        </RadarChart>
      </ResponsiveContainer>
    );
  }

  if (type === "scatter") {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
          {grid}
          <XAxis
            dataKey={config.scatterX}
            name={config.scatterXLabel || config.scatterX}
            type="number"
            {...commonAxis}
            label={{ value: config.scatterXLabel, position: "insideBottom", offset: -2, fill: "#6b7280", fontSize: 10 }}
          />
          <YAxis
            dataKey={config.scatterY}
            name={config.scatterYLabel || config.scatterY}
            type="number"
            {...commonAxis}
            tickFormatter={config.yFormatter}
            width={55}
          />
          {config.scatterZ && <ZAxis dataKey={config.scatterZ} range={[40, 200]} />}
          <Tooltip
            cursor={{ stroke: "#ffffff10" }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0].payload;
              return (
                <div className="rounded-xl bg-[#0d0d14]/95 border border-white/10 px-3 py-2 text-xs">
                  <p className="text-white font-semibold mb-1">Data Point</p>
                  {Object.entries(d).map(([k, v]) => (
                    <p key={k} className="text-zinc-400">
                      <span className="text-zinc-300">{k}:</span> {v}
                    </p>
                  ))}
                </div>
              );
            }}
          />
          <Scatter
            name={config.lines[0]?.label || "Data"}
            data={data}
            fill={colors[0]}
            fillOpacity={0.7}
            animationDuration={500}
          />
        </ScatterChart>
      </ResponsiveContainer>
    );
  }

  if (type === "composed") {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
          <defs>
            {config.lines.map((l, i) => (
              <linearGradient key={l.key} id={`grad-${l.key}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={colors[i % colors.length]} stopOpacity={0.35} />
                <stop offset="95%" stopColor={colors[i % colors.length]} stopOpacity={0.02} />
              </linearGradient>
            ))}
          </defs>
          {grid}
          <XAxis dataKey={config.xKey} {...commonAxis} interval={config.xInterval ?? "preserveStartEnd"} />
          <YAxis {...commonAxis} tickFormatter={config.yFormatter} width={50} domain={['dataMin - 5%', 'dataMax + 5%']} />
          {tooltip}
          <Legend wrapperStyle={{ fontSize: 10, color: "#9ca3af" }} />
          {config.lines.map((l, i) => {
            if (i === 0) {
              return (
                <Bar
                  key={l.key}
                  dataKey={l.key}
                  name={l.label}
                  fill={colors[i % colors.length]}
                  fillOpacity={0.4}
                  radius={[3, 3, 0, 0]}
                  barSize={20}
                  animationDuration={500}
                />
              );
            }
            return (
              <Line
                key={l.key}
                type="monotone"
                dataKey={l.key}
                name={l.label}
                stroke={colors[i % colors.length]}
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 4 }}
                animationDuration={600}
              />
            );
          })}
        </ComposedChart>
      </ResponsiveContainer>
    );
  }

  if (type === "stepped") {
    return (
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
          <defs>
            {config.lines.map((l, i) => (
              <linearGradient key={l.key} id={`grad-${l.key}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={colors[i % colors.length]} stopOpacity={0.35} />
                <stop offset="95%" stopColor={colors[i % colors.length]} stopOpacity={0.02} />
              </linearGradient>
            ))}
          </defs>
          {grid}
          <XAxis dataKey={config.xKey} {...commonAxis} interval={config.xInterval ?? "preserveStartEnd"} />
          <YAxis {...commonAxis} tickFormatter={config.yFormatter} width={50} domain={['dataMin - 5%', 'dataMax + 5%']} />
          {tooltip}
          <Legend wrapperStyle={{ fontSize: 10, color: "#9ca3af" }} />
          {config.lines.map((l, i) => (
            <Area
              key={l.key}
              type="stepAfter"
              dataKey={l.key}
              name={l.label}
              stroke={colors[i % colors.length]}
              fill={`url(#grad-${l.key})`}
              strokeWidth={2}
              dot={false}
              animationDuration={600}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    );
  }

  return null;
}

/**
 * @param {object} props
 * @param {string} props.title — Card title
 * @param {string} [props.subtitle] — Card subtitle
 * @param {string[]} props.allowedTypes — e.g. ["line","area","bar"]
 * @param {string} props.defaultType — initial chart type
 * @param {any[]} props.data — the data array
 * @param {object} props.config — chart configuration object
 * @param {string} [props.badge] — optional badge text
 * @param {string} [props.badgeColor] — badge color class
 */
export default function AnalyticsChart({
  title,
  subtitle,
  allowedTypes,
  defaultType,
  data,
  config,
  badge,
  badgeColor = "bg-blue-500/15 text-blue-300 border-blue-500/20",
  height = 260,
}) {
  const [activeType, setActiveType] = useState(defaultType);

  return (
    <div className="rounded-2xl bg-white/[0.025] border border-white/8 p-5 flex flex-col">
      {/* Header */}
      <div className="flex items-start justify-between mb-4 gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2 mb-0.5">
            <h3 className="font-semibold text-white text-sm truncate">{title}</h3>
            {badge && (
              <span className={`shrink-0 text-[10px] font-medium px-2 py-0.5 rounded-full border ${badgeColor}`}>
                {badge}
              </span>
            )}
          </div>
          {subtitle && <p className="text-xs text-zinc-500">{subtitle}</p>}
        </div>

        {/* Chart type switcher */}
        <div className="flex items-center gap-1 shrink-0">
          {allowedTypes.map((t) => {
            const Icon = TYPE_ICONS[t];
            const isActive = activeType === t;
            return (
              <button
                key={t}
                title={TYPE_LABELS[t]}
                onClick={() => setActiveType(t)}
                className={cn(
                  "w-7 h-7 rounded-lg flex items-center justify-center transition-all text-xs",
                  isActive
                    ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                    : "text-zinc-600 hover:text-zinc-400 hover:bg-white/5"
                )}
              >
                {Icon ? <Icon className="w-3.5 h-3.5" /> : t[0].toUpperCase()}
              </button>
            );
          })}
        </div>
      </div>

      {/* Chart */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeType}
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.98 }}
          transition={{ duration: 0.2 }}
          style={{ height }}
        >
          <ChartRenderer type={activeType} data={data} config={config} />
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
