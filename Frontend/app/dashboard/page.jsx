"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Calendar,
  Zap,
  TrendingUp,
  TrendingDown,
  Minus,
  BarChart3,
  Activity,
  ChevronDown,
  Loader2,
  Clock,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useToast } from "@/components/ui/toast";
import { SummaryCardsSkeleton } from "@/components/ui/skeleton";
import ForecastChart from "@/app/chart/ForecastChart";
import {
  toMidnightEpoch,
  validateDateRange,
  computeSummary,
  getForecastTypeLabel,
  getValueUnit,
  formatEpochFull,
} from "@/lib/forecast";

// ─── Constants ─────────────────────────────────────────────────────────────────

const FORECAST_TYPES = [
  {
    value: "hourly",
    label: "Hourly",
    description: "24 hourly values for a single day",
    icon: Clock,
    color: "from-indigo-500 to-blue-600",
    accent: "text-indigo-400",
    border: "border-indigo-500/30",
    bg: "bg-indigo-500/10",
  },
  {
    value: "weekly",
    label: "Weekly",
    description: "Daily totals across a week",
    icon: BarChart3,
    color: "from-cyan-500 to-teal-600",
    accent: "text-cyan-400",
    border: "border-cyan-500/30",
    bg: "bg-cyan-500/10",
  },
  {
    value: "monthly",
    label: "Monthly",
    description: "Day-by-day trend for a month",
    icon: Activity,
    color: "from-violet-500 to-purple-600",
    accent: "text-violet-400",
    border: "border-violet-500/30",
    bg: "bg-violet-500/10",
  },
  {
    value: "quarterly",
    label: "Quarterly",
    description: "Monthly totals (max 108 days)",
    icon: TrendingUp,
    color: "from-amber-500 to-orange-600",
    accent: "text-amber-400",
    border: "border-amber-500/30",
    bg: "bg-amber-500/10",
  },
];

// ─── Date Input ─────────────────────────────────────────────────────────────────

function DateInput({ label, value, onChange, id }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={id} className="text-xs font-medium text-zinc-400 uppercase tracking-wider">
        {label}
      </label>
      <div className="relative">
        <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-500 pointer-events-none" />
        <input
          id={id}
          type="date"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full rounded-xl bg-white/5 border border-white/10 text-white text-sm pl-10 pr-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all [color-scheme:dark]"
        />
      </div>
    </div>
  );
}

// ─── Summary Card ───────────────────────────────────────────────────────────────

function SummaryCard({ label, value, unit, icon: Icon, accent, sublabel }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className="rounded-2xl bg-white/[0.03] border border-white/8 p-5 hover:bg-white/[0.05] transition-colors group"
    >
      <div className="flex items-start justify-between mb-3">
        <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">{label}</p>
        <div className={`p-1.5 rounded-lg ${accent.replace("text-", "bg-").replace("400", "500/15")}`}>
          <Icon className={`h-3.5 w-3.5 ${accent}`} />
        </div>
      </div>
      <p className={`text-2xl font-bold text-white tabular-nums`}>
        {typeof value === "number"
          ? value.toLocaleString(undefined, { maximumFractionDigits: 2 })
          : value}
        {unit && (
          <span className="text-sm font-normal text-zinc-500 ml-1">{unit}</span>
        )}
      </p>
      {sublabel && (
        <p className="text-xs text-zinc-600 mt-1">{sublabel}</p>
      )}
    </motion.div>
  );
}

// ─── Trend Card ─────────────────────────────────────────────────────────────────

function TrendCard({ trend }) {
  const configs = {
    rising: {
      icon: TrendingUp,
      label: "Rising",
      accent: "text-rose-400",
      bg: "bg-rose-500/10",
      border: "border-rose-500/20",
      desc: "Demand strengthening",
    },
    falling: {
      icon: TrendingDown,
      label: "Falling",
      accent: "text-blue-400",
      bg: "bg-blue-500/10",
      border: "border-blue-500/20",
      desc: "Demand easing",
    },
    stable: {
      icon: Minus,
      label: "Stable",
      accent: "text-emerald-400",
      bg: "bg-emerald-500/10",
      border: "border-emerald-500/20",
      desc: "Steady demand",
    },
  };

  const cfg = configs[trend] || configs.stable;
  const Icon = cfg.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: 0.15 }}
      className={`rounded-2xl bg-white/[0.03] border ${cfg.border} p-5 hover:bg-white/[0.05] transition-colors`}
    >
      <div className="flex items-start justify-between mb-3">
        <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Trend</p>
        <div className={`p-1.5 rounded-lg ${cfg.bg}`}>
          <Icon className={`h-3.5 w-3.5 ${cfg.accent}`} />
        </div>
      </div>
      <p className={`text-2xl font-bold ${cfg.accent}`}>{cfg.label}</p>
      <p className="text-xs text-zinc-600 mt-1">{cfg.desc}</p>
    </motion.div>
  );
}

// ─── Main Page ──────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const { addToast } = useToast();

  // Form state
  const today = new Date().toISOString().slice(0, 10);
  const [forecastType, setForecastType] = useState("hourly");
  const [startDate, setStartDate] = useState(today);
  const [endDate, setEndDate] = useState(today);
  const [substationName, setSubstationName] = useState("Grid Substation");

  // Result state
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [summary, setSummary] = useState(null);
  const [lastGenerated, setLastGenerated] = useState(null);

  // When hourly is selected, auto-set end = start
  const handleForecastTypeChange = (type) => {
    setForecastType(type);
    if (type === "hourly") setEndDate(startDate);
  };

  const handleStartDateChange = (val) => {
    setStartDate(val);
    if (forecastType === "hourly") setEndDate(val);
  };

  const handleGenerate = useCallback(async () => {
    // Build JS dates from string values
    const start = new Date(startDate + "T00:00:00");
    const end = new Date(endDate + "T00:00:00");

    // Client-side validation
    const validationError = validateDateRange(start, end, forecastType);
    if (validationError) {
      addToast(validationError, "warning");
      return;
    }

    setLoading(true);
    setPredictions(null);
    setSummary(null);

    try {
      const startEpoch = toMidnightEpoch(start);
      const endEpoch = toMidnightEpoch(end);

      const res = await fetch(
        `/api/forecast?type=${forecastType}&start_date=${startEpoch}&end_date=${endEpoch}`
      );
      const data = await res.json();

      if (!res.ok || data.error) {
        addToast(data.error || "Forecast generation failed. Please try again.", "error");
        return;
      }

      const preds = data.predictions || [];
      if (preds.length === 0) {
        addToast("No forecast data returned from the backend.", "warning");
        return;
      }

      setPredictions(preds);
      setSummary(computeSummary(preds));
      setLastGenerated(new Date());
      addToast(`${getForecastTypeLabel(forecastType)} generated successfully.`, "success", 3000);
    } catch (err) {
      addToast(`Connection error: ${err.message}`, "error");
    } finally {
      setLoading(false);
    }
  }, [forecastType, startDate, endDate, addToast]);

  const unit = getValueUnit(forecastType);
  const selectedType = FORECAST_TYPES.find((t) => t.value === forecastType);

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="space-y-8"
    >
      {/* ── Page Header ── */}
      <div>
        <div className="flex items-center gap-3 mb-1">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center">
            <Zap className="w-4 h-4 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-white">Forecast Dashboard</h1>
        </div>
        <p className="text-sm text-zinc-500 ml-11">
          Generate electricity load forecasts for the grid substation
        </p>
      </div>

      {/* ── Filter Panel ── */}
      <div className="rounded-2xl bg-white/[0.03] border border-white/8 p-6">
        <h2 className="text-sm font-semibold text-zinc-300 mb-5 flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-blue-400" />
          Configure Forecast
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-end">

          {/* Forecast Type */}
          <div className="lg:col-span-3">
            <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider mb-2">
              Forecast Type
            </p>
            <div className="grid grid-cols-2 gap-2">
              {FORECAST_TYPES.map((ft) => {
                const Icon = ft.icon;
                const isActive = forecastType === ft.value;
                return (
                  <button
                    key={ft.value}
                    onClick={() => handleForecastTypeChange(ft.value)}
                    className={`flex items-center gap-2 rounded-xl px-3 py-2.5 text-sm font-medium border transition-all duration-200 ${
                      isActive
                        ? `${ft.bg} ${ft.border} ${ft.accent}`
                        : "bg-white/[0.02] border-white/8 text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-300"
                    }`}
                  >
                    <Icon className="h-4 w-4 shrink-0" />
                    <span>{ft.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Date Range */}
          <div className="lg:col-span-4 grid grid-cols-2 gap-3">
            <DateInput
              id="start-date"
              label={forecastType === "hourly" ? "Date" : "Start Date"}
              value={startDate}
              onChange={handleStartDateChange}
            />
            {forecastType !== "hourly" && (
              <DateInput
                id="end-date"
                label="End Date"
                value={endDate}
                onChange={setEndDate}
              />
            )}
          </div>

          {/* Substation Name */}
          <div className="lg:col-span-3">
            <label className="text-xs font-medium text-zinc-400 uppercase tracking-wider block mb-1.5">
              Substation Name
            </label>
            <input
              type="text"
              value={substationName}
              onChange={(e) => setSubstationName(e.target.value)}
              placeholder="e.g. Delhi Grid Station"
              className="w-full rounded-xl bg-white/5 border border-white/10 text-white text-sm px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all placeholder:text-zinc-600"
            />
          </div>

          {/* Generate Button */}
          <div className="lg:col-span-2">
            <Button
              onClick={handleGenerate}
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 text-white font-semibold rounded-xl py-2.5 h-auto disabled:opacity-60 transition-all duration-200 shadow-lg shadow-blue-900/30"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Generating…
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  <Zap className="h-4 w-4" />
                  Generate
                </span>
              )}
            </Button>
          </div>
        </div>

        {/* Selected type description */}
        {selectedType && (
          <motion.p
            key={forecastType}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-xs text-zinc-600 mt-4 pl-1"
          >
            <span className={`font-medium ${selectedType.accent}`}>
              {selectedType.label}:
            </span>{" "}
            {selectedType.description}
          </motion.p>
        )}
      </div>

      {/* ── Chart Section ── */}
      <div className="rounded-2xl bg-white/[0.03] border border-white/8 p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-base font-semibold text-white flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-zinc-400" />
              {predictions
                ? getForecastTypeLabel(forecastType)
                : "Forecast Chart"}
            </h2>
            {lastGenerated && (
              <p className="text-xs text-zinc-600 mt-0.5">
                Generated {lastGenerated.toLocaleTimeString()} · {substationName}
              </p>
            )}
          </div>
          {predictions && (
            <span className={`text-xs font-medium px-2.5 py-1 rounded-full border ${selectedType?.bg} ${selectedType?.border} ${selectedType?.accent}`}>
              {predictions.length} data points
            </span>
          )}
        </div>

        <ForecastChart
          data={predictions}
          forecastType={forecastType}
          loading={loading}
        />
      </div>

      {/* ── Summary Cards ── */}
      <AnimatePresence>
        {loading && !summary && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <SummaryCardsSkeleton />
          </motion.div>
        )}

        {summary && !loading && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-3"
          >
            <h2 className="text-sm font-semibold text-zinc-400 flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Summary Statistics
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
              <SummaryCard
                label="Average Load"
                value={summary.average}
                unit={unit}
                icon={Activity}
                accent="text-blue-400"
              />
              <SummaryCard
                label="Peak Load"
                value={summary.maximum}
                unit={unit}
                icon={TrendingUp}
                accent="text-rose-400"
                sublabel="Maximum observed"
              />
              <SummaryCard
                label="Min Load"
                value={summary.minimum}
                unit={unit}
                icon={TrendingDown}
                accent="text-sky-400"
                sublabel="Minimum observed"
              />
              <SummaryCard
                label="Total Energy"
                value={summary.total}
                unit={unit}
                icon={Zap}
                accent="text-amber-400"
                sublabel={`${predictions?.length || 0} periods`}
              />
              <TrendCard trend={summary.trend} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Call to Action for AI Report ── */}
      {predictions && !loading && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="rounded-2xl bg-gradient-to-r from-violet-950/40 to-blue-950/40 border border-violet-700/20 p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-violet-500/20 flex items-center justify-center shrink-0">
              <Sparkles className="h-5 w-5 text-violet-400" />
            </div>
            <div>
              <p className="font-semibold text-white text-sm">Generate AI Operational Report</p>
              <p className="text-xs text-zinc-500">
                Get AI-powered insights, risk analysis, and action plan for this forecast
              </p>
            </div>
          </div>
          <a
            href="/dashboard/ai-insights"
            className="shrink-0 inline-flex items-center gap-2 rounded-xl bg-violet-600 hover:bg-violet-500 text-white text-sm font-semibold px-4 py-2.5 transition-colors"
          >
            <Sparkles className="h-4 w-4" />
            Open AI Report
          </a>
        </motion.div>
      )}
    </motion.div>
  );
}