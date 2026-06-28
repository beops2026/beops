"use client";

import { useState, useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import {
  BarChart3, Activity, Cpu, Brain, Zap, Thermometer,
  TrendingUp, Gauge, AlertCircle, Calendar,
} from "lucide-react";
import AnalyticsChart from "@/app/chart/AnalyticsChart";
import { ChartSkeleton } from "@/components/ui/skeleton";

// ─── Color Palette ─────────────────────────────────────────────────────────────
const INDIGO = "#6366f1";
const CYAN = "#22d3ee";
const VIOLET = "#a78bfa";
const AMBER = "#f59e0b";
const EMERALD = "#34d399";
const ROSE = "#fb7185";

// ─── Helpers ────────────────────────────────────────────────────────────────────
// Local "YYYY-MM-DD" string for today + `days` offset.
function isoOffset(days = 0) {
  const d = new Date();
  d.setDate(d.getDate() + days);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

// Open-Meteo's forecast endpoint only serves roughly [today-90d, today+15d].
// Keep the picker safely inside that window so requests never 400.
const MIN_DATE = isoOffset(-88);
const MAX_DATE = isoOffset(14);

// Convert a "YYYY-MM-DD" string to a local midnight Unix epoch (seconds).
function isoToMidnightEpoch(iso) {
  return Math.floor(new Date(`${iso}T00:00:00`).getTime() / 1000);
}

// Pearson correlation coefficient between two equal-length numeric arrays.
function pearson(xs, ys) {
  const n = Math.min(xs.length, ys.length);
  if (n === 0) return 0;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    const a = xs[i] - mx;
    const b = ys[i] - my;
    num += a * b;
    dx += a * a;
    dy += b * b;
  }
  const den = Math.sqrt(dx * dy);
  return den === 0 ? 0 : num / den;
}

// ─── KPI Card ───────────────────────────────────────────────────────────────────
function KPICard({ kpi }) {
  return (
    <div className={`rounded-2xl border ${kpi.border} bg-white/[0.025] p-5 flex items-start gap-4`}>
      <div className={`w-10 h-10 rounded-xl ${kpi.bg} flex items-center justify-center shrink-0`}>
        <kpi.icon className={`h-5 w-5 ${kpi.color}`} />
      </div>
      <div className="min-w-0">
        <p className="text-xs text-zinc-500 mb-1">{kpi.label}</p>
        <p className="text-2xl font-bold text-white tabular-nums truncate">{kpi.value}</p>
        {kpi.sub && <p className="text-xs text-zinc-600 mt-0.5">{kpi.sub}</p>}
      </div>
    </div>
  );
}

function KPISkeleton() {
  return <div className="rounded-2xl border border-white/8 bg-white/[0.02] p-5 h-[92px] animate-pulse" />;
}

// ─── Section Header ───────────────────────────────────────────────────────────
function SectionHeader({ icon: Icon, title, subtitle }) {
  return (
    <div className="flex items-center gap-3 mb-6">
      <div className="w-8 h-8 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center">
        <Icon className="h-4 w-4 text-zinc-400" />
      </div>
      <div>
        <h2 className="text-sm font-semibold text-zinc-200">{title}</h2>
        {subtitle && <p className="text-xs text-zinc-600">{subtitle}</p>}
      </div>
    </div>
  );
}

// A chart slot that shows a skeleton while loading and an error card on failure.
function ChartSlot({ loading, error, hasData, children }) {
  if (loading) {
    return (
      <div className="rounded-2xl bg-white/[0.025] border border-white/8 p-5">
        <ChartSkeleton />
      </div>
    );
  }
  if (error || !hasData) {
    return (
      <div className="rounded-2xl bg-white/[0.025] border border-white/8 p-5 flex flex-col items-center justify-center gap-2 h-64 text-center">
        <AlertCircle className="h-6 w-6 text-rose-400/70" />
        <p className="text-sm text-zinc-400">{error || "No data available"}</p>
        <p className="text-xs text-zinc-600">
          Ensure the forecast backend is running and the date is within the weather horizon.
        </p>
      </div>
    );
  }
  return children;
}

// ─── Main Page ─────────────────────────────────────────────────────────────────
export default function AnalyticsPage() {
  const [date, setDate] = useState(() => isoOffset(0));
  const [env, setEnv] = useState([]);
  const [corr, setCorr] = useState([]);
  const [feat, setFeat] = useState([]);
  const [featMethod, setFeatMethod] = useState("");
  const [loading, setLoading] = useState(true); // env + correlation
  const [featLoading, setFeatLoading] = useState(true);
  const [error, setError] = useState(null);

  // Feature importance is date-independent (samples the model around "now"),
  // so fetch it once on mount.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const f = await fetch(`/api/analytics?metric=feature-importance`).then((r) => r.json());
        if (cancelled) return;
        if (f.error) throw new Error(f.error);
        setFeat(f.features || []);
        setFeatMethod(f.method || "");
      } catch (_) {
        // leave feat empty → ChartSlot shows its fallback
      } finally {
        if (!cancelled) setFeatLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Environmental + correlation refetch whenever the selected date changes.
  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const start = isoToMidnightEpoch(date);
        // Clamp the 7-day window to the weather horizon so it never overruns.
        const end = Math.min(start + 6 * 86400, isoToMidnightEpoch(MAX_DATE));

        const [e, c] = await Promise.all([
          fetch(`/api/analytics?metric=environmental&date=${start}`).then((r) => r.json()),
          fetch(`/api/analytics?metric=correlation&start_date=${start}&end_date=${end}`).then((r) => r.json()),
        ]);

        if (cancelled) return;

        const firstError = e.error || c.error;
        if (firstError) throw new Error(firstError);

        setEnv(e.data || []);
        setCorr(c.points || []);
      } catch (err) {
        if (!cancelled) {
          setError(err.message || "Failed to load analytics");
          setEnv([]);
          setCorr([]);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [date]);

  // Real KPIs derived from today's environmental (weather + load) data.
  const kpis = useMemo(() => {
    if (!env || env.length === 0) return [];
    const loads = env.map((d) => d.load);
    const temps = env.map((d) => d.temperature);
    const peak = Math.max(...loads);
    const peakHour = env[loads.indexOf(peak)]?.hour ?? "--";
    const avg = loads.reduce((a, b) => a + b, 0) / loads.length;
    const min = Math.min(...loads);
    const r = pearson(temps, loads);

    const fmtMW = (v) => `${(v / 1000).toFixed(2)}k`;

    return [
      { label: "Peak Load", value: `${fmtMW(peak)} MW`, sub: `at ${peakHour}`, icon: Zap, color: "text-rose-400", bg: "bg-rose-500/10", border: "border-rose-500/20" },
      { label: "Average Load", value: `${fmtMW(avg)} MW`, sub: `min ${fmtMW(min)} MW`, icon: Activity, color: "text-indigo-400", bg: "bg-indigo-500/10", border: "border-indigo-500/20" },
      { label: "Peak Hour", value: peakHour, sub: "highest demand", icon: TrendingUp, color: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/20" },
      { label: "Temp ↔ Load", value: r.toFixed(2), sub: "correlation (selected day)", icon: Thermometer, color: "text-cyan-400", bg: "bg-cyan-500/10", border: "border-cyan-500/20" },
    ];
  }, [env]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="space-y-10"
    >
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
              <BarChart3 className="w-4 h-4 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-white">Analytics</h1>
          </div>
          <p className="text-sm text-zinc-500 ml-11">
            Live model &amp; weather-driven analysis for the selected day
          </p>
        </div>

        {/* Forecast date selector — drives Environmental & Correlation views */}
        <div className="flex flex-col gap-1.5">
          <label htmlFor="analytics-date" className="text-xs font-medium text-zinc-400 uppercase tracking-wider">
            Forecast date
          </label>
          <div className="relative">
            <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-500 pointer-events-none" />
            <input
              id="analytics-date"
              type="date"
              value={date}
              min={MIN_DATE}
              max={MAX_DATE}
              onChange={(e) => setDate(e.target.value)}
              className="rounded-xl bg-white/5 border border-white/10 text-white text-sm pl-10 pr-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50 transition-all [color-scheme:dark]"
            />
          </div>
          <p className="text-[10px] text-zinc-600">
            Weather data available {MIN_DATE} → {MAX_DATE}
          </p>
        </div>
      </div>

      {/* ── Real KPI Cards (derived from weather + model) ── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {loading
          ? Array.from({ length: 4 }).map((_, i) => <KPISkeleton key={i} />)
          : kpis.map((kpi) => <KPICard key={kpi.label} kpi={kpi} />)}
      </div>

      {/* ── Section 1: Feature Analysis ── */}
      <div>
        <SectionHeader
          icon={Cpu}
          title="Feature Analysis"
          subtitle="What actually drives the forecast model's predictions"
        />
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
          {/* Feature Importance (real, occlusion-based) — date-independent */}
          <ChartSlot loading={featLoading} error={null} hasData={feat.length > 0}>
            <AnalyticsChart
              title="Feature Importance"
              subtitle={featMethod ? `Model-derived · ${featMethod}` : "Model-derived contribution"}
              badge="Model"
              badgeColor="bg-violet-500/15 text-violet-300 border-violet-500/20"
              allowedTypes={["bar", "pie"]}
              defaultType="bar"
              data={feat}
              config={{
                xKey: "feature",
                lines: [{ key: "importance", label: "Importance (%)" }],
                colors: [VIOLET, CYAN, AMBER, EMERALD, ROSE, INDIGO, "#8b5cf6", "#06b6d4"],
                yFormatter: (v) => `${v}%`,
              }}
              height={300}
            />
          </ChartSlot>

          {/* Environmental Factors vs Load (real) */}
          <ChartSlot loading={loading} error={error} hasData={env.length > 0}>
            <AnalyticsChart
              title="Environmental Factors vs Load"
              subtitle="Normalized 0–100 so each series' variation is visible · hover for real values"
              badge="Live"
              badgeColor="bg-emerald-500/15 text-emerald-300 border-emerald-500/20"
              allowedTypes={["line", "area", "bar", "composed", "stepped"]}
              defaultType="line"
              data={env}
              config={{
                xKey: "hour",
                xInterval: 3,
                lines: [
                  { key: "load_pct", label: "Load" },
                  { key: "temperature_pct", label: "Temp" },
                  { key: "humidity_pct", label: "Humidity" },
                ],
                colors: [INDIGO, AMBER, CYAN],
                yFormatter: (v) => `${v}%`,
                tooltipRaw: {
                  load_pct: { key: "load", unit: "MW" },
                  temperature_pct: { key: "temperature", unit: "°C" },
                  humidity_pct: { key: "humidity", unit: "%" },
                },
              }}
              height={300}
            />
          </ChartSlot>
        </div>
      </div>

      {/* ── Section 2: Weather → Load Correlation ── */}
      <div>
        <SectionHeader
          icon={Gauge}
          title="Weather × Load Correlation"
          subtitle="How weather conditions relate to predicted grid demand (7-day window from selected date)"
        />
        <div className="grid grid-cols-1 gap-5">
          {/* Correlation Scatter (real) */}
          <ChartSlot loading={loading} error={error} hasData={corr.length > 0}>
            <AnalyticsChart
              title="Temperature × Humidity → Load"
              subtitle={`${corr.length} hourly samples from live forecasts`}
              badge="Live"
              badgeColor="bg-amber-500/15 text-amber-300 border-amber-500/20"
              allowedTypes={["scatter"]}
              defaultType="scatter"
              data={corr}
              config={{
                xKey: "temperature",
                scatterX: "temperature",
                scatterY: "load",
                scatterXLabel: "Temperature (°C)",
                scatterYLabel: "Load (MW)",
                lines: [{ key: "load", label: "Load (MW)" }],
                colors: [CYAN],
                yFormatter: (v) => `${(v / 1000).toFixed(1)}k`,
              }}
              height={320}
            />
          </ChartSlot>
        </div>
      </div>
    </motion.div>
  );
}
