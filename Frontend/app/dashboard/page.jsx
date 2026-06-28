"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Calendar,
  Zap,
  TrendingUp,
  TrendingDown,
  Minus,
  BarChart3,
  Activity,
  Loader2,
  Clock,
  Sparkles,
  History,
  Printer,
  Bell,
  BellOff,
  BellRing,
  AlertTriangle,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/toast";
import ForecastChart from "@/app/chart/ForecastChart";
import {
  toMidnightEpoch,
  validateDateRange,
  computeSummary,
  getForecastTypeLabel,
  getValueUnit,
} from "@/lib/forecast";
import { cn } from "@/lib/utils";

// ─── Constants ─────────────────────────────────────────────────────────────────

const FORECAST_TYPES = [
  {
    value: "hourly",
    label: "Hourly",
    description: "24 hourly values for a single day",
    icon: Clock,
    accent: "text-indigo-400",
    border: "border-indigo-500/30",
    bg: "bg-indigo-500/10",
  },
  {
    value: "weekly",
    label: "Weekly",
    description: "Daily totals across a week",
    icon: BarChart3,
    accent: "text-cyan-400",
    border: "border-cyan-500/30",
    bg: "bg-cyan-500/10",
  },
  {
    value: "monthly",
    label: "Monthly",
    description: "Day-by-day trend for a month",
    icon: Activity,
    accent: "text-violet-400",
    border: "border-violet-500/30",
    bg: "bg-violet-500/10",
  },
  {
    value: "quarterly",
    label: "Quarterly",
    description: "Monthly totals (max 108 days)",
    icon: TrendingUp,
    accent: "text-amber-400",
    border: "border-amber-500/30",
    bg: "bg-amber-500/10",
  },
];

const TREND_META = {
  rising: { icon: TrendingUp, label: "Rising", accent: "text-rose-400" },
  falling: { icon: TrendingDown, label: "Falling", accent: "text-blue-400" },
  stable: { icon: Minus, label: "Stable", accent: "text-emerald-400" },
};

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

// ─── Compact Summary Strip ──────────────────────────────────────────────────────

function SummaryStrip({ summary, unit }) {
  const trend = TREND_META[summary.trend] || TREND_META.stable;
  const TrendIcon = trend.icon;
  const stats = [
    { label: "Average", value: summary.average, accent: "text-blue-400" },
    { label: "Peak", value: summary.maximum, accent: "text-rose-400" },
    { label: "Min", value: summary.minimum, accent: "text-sky-400" },
  ];
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-5 pt-4 border-t border-white/5"
    >
      {stats.map((s) => (
        <div key={s.label}>
          <p className="text-[10px] uppercase tracking-wider text-zinc-500">{s.label}</p>
          <p className={`text-sm font-bold tabular-nums ${s.accent}`}>
            {s.value.toLocaleString(undefined, { maximumFractionDigits: 1 })}
            <span className="text-[10px] font-normal text-zinc-600 ml-1">{unit}</span>
          </p>
        </div>
      ))}
      <div>
        <p className="text-[10px] uppercase tracking-wider text-zinc-500">Trend</p>
        <p className={`text-sm font-bold flex items-center gap-1 ${trend.accent}`}>
          <TrendIcon className="h-3.5 w-3.5" />
          {trend.label}
        </p>
      </div>
    </motion.div>
  );
}

// ─── Alert ID generator ─────────────────────────────────────────────────────────
// Monotonic so two panels firing in the same millisecond never collide
// (Date.now() alone can produce duplicate React keys).
let alertSeq = 0;
const nextAlertId = () => `alert-${Date.now()}-${++alertSeq}`;

// Severity ranking — peak deviation thresholds relative to the configured %.
const PEAK_FALLBACK_THRESHOLD = 6;

// ─── Forecast Panel ─────────────────────────────────────────────────────────────
// A fully self-contained, independently-configurable forecast panel.
// Holds its own type/date/substation config, calls the backend on demand,
// and starts empty until the user generates.

function ForecastPanel({ title, subtitle, icon: HeaderIcon, headerAccent, defaultType, onResult, notifEnabled, peakThresholdPct, onAlert }) {
  const { addToast } = useToast();
  const today = new Date().toISOString().slice(0, 10);

  const [forecastType, setForecastType] = useState(defaultType);
  const [startDate, setStartDate] = useState(today);
  const [endDate, setEndDate] = useState(today);
  const [substation, setSubstation] = useState("Grid Substation");

  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [summary, setSummary] = useState(null);
  const [lastGenerated, setLastGenerated] = useState(null);

  // Keep the latest notification config in a ref so the memoized generate
  // handler always reads fresh values. Without this, toggling alerts or moving
  // the threshold slider after first render leaves handleGenerate firing on
  // stale config — the root cause of the "flaky" alerts.
  const notifRef = useRef({});
  useEffect(() => {
    notifRef.current = { notifEnabled, peakThresholdPct, onAlert, substation, title, forecastType };
  });

  const handleForecastTypeChange = (type) => {
    setForecastType(type);
    if (type === "hourly") setEndDate(startDate);
  };

  const handleStartDateChange = (val) => {
    setStartDate(val);
    if (forecastType === "hourly") setEndDate(val);
  };

  const handleGenerate = useCallback(async () => {
    const start = new Date(startDate + "T00:00:00");
    const end = new Date(endDate + "T00:00:00");

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

      const summaryData = computeSummary(preds);
      setPredictions(preds);
      setSummary(summaryData);
      setLastGenerated(new Date());
      onResult?.();
      addToast(`${getForecastTypeLabel(forecastType)} generated successfully.`, "success", 3000);

      // ── Peak demand alert ────────────────────────────────────────────────
      // Read live config from the ref so we never act on stale values.
      const cfg = notifRef.current;
      if (cfg.notifEnabled && summaryData && summaryData.average > 0) {
        const { maximum, average, trend } = summaryData;
        const deviation = ((maximum - average) / average) * 100;
        const threshold = cfg.peakThresholdPct ?? PEAK_FALLBACK_THRESHOLD;

        if (deviation >= threshold) {
          const isCritical = deviation >= threshold * 1.5 || trend === "rising";
          const severity = isCritical ? "CRITICAL" : "WARNING";
          const u = getValueUnit(cfg.forecastType);

          // 1) In-app banner + log — ALWAYS fires when alerts are enabled, so a
          //    muted OS/browser never swallows the alert silently.
          cfg.onAlert?.({
            id: nextAlertId(),
            severity,
            substation: cfg.substation,
            peak: maximum,
            avg: average,
            deviation: deviation.toFixed(1),
            trend,
            unit: u,
            forecastType: cfg.forecastType,
            panelTitle: cfg.title,
            time: new Date().toLocaleTimeString(),
          });

          // 2) Native browser push — best-effort, only when actually permitted.
          if (typeof Notification !== "undefined" && Notification.permission === "granted") {
            try {
              const n = new Notification(`${severity}: Peak Demand Alert — ${cfg.substation}`, {
                body: `Peak: ${maximum.toLocaleString(undefined, { maximumFractionDigits: 1 })} ${u} · ${deviation.toFixed(1)}% above average\nTrend: ${trend.toUpperCase()} · ${getForecastTypeLabel(cfg.forecastType)} · ${cfg.title}`,
                icon: "/favicon.ico",
                requireInteraction: isCritical,
                tag: `peak-${cfg.title.replace(/\s+/g, "-").toLowerCase()}-${cfg.forecastType}`,
              });
              n.onclick = () => {
                window.focus();
                n.close();
              };
            } catch (_) {
              // Notification constructor can throw in restricted contexts — the
              // in-app banner above already covered the user.
            }
          }
        }
      }
    } catch (err) {
      addToast(`Connection error: ${err.message}`, "error");
    } finally {
      setLoading(false);
    }
  }, [forecastType, startDate, endDate, addToast, onResult]);

  const unit = getValueUnit(forecastType);
  const selectedType = FORECAST_TYPES.find((t) => t.value === forecastType);

  return (
    <div className="print-card rounded-2xl bg-white/[0.03] border border-white/8 p-6 flex flex-col">
      {/* Panel header */}
      <div className="flex items-start justify-between gap-3 mb-5">
        <div className="flex items-center gap-3">
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${headerAccent.bg}`}>
            <HeaderIcon className={`w-4 h-4 ${headerAccent.text}`} />
          </div>
          <div>
            <h2 className="text-base font-semibold text-white">{title}</h2>
            <p className="text-xs text-zinc-500">{subtitle}</p>
          </div>
        </div>
        {predictions && (
          <span className={`shrink-0 text-xs font-medium px-2.5 py-1 rounded-full border ${selectedType?.bg} ${selectedType?.border} ${selectedType?.accent}`}>
            {predictions.length} pts
          </span>
        )}
      </div>

      {/* Config */}
      <div className="space-y-4 no-print">
        {/* Forecast type */}
        <div>
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
                  className={cn(
                    "flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium border transition-all duration-200",
                    isActive
                      ? `${ft.bg} ${ft.border} ${ft.accent}`
                      : "bg-white/[0.02] border-white/8 text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-300"
                  )}
                >
                  <Icon className="h-4 w-4 shrink-0" />
                  <span>{ft.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Date range */}
        <div className={cn("grid gap-3", forecastType === "hourly" ? "grid-cols-1" : "grid-cols-2")}>
          <DateInput
            id={`${title}-start`}
            label={forecastType === "hourly" ? "Date" : "Start Date"}
            value={startDate}
            onChange={handleStartDateChange}
          />
          {forecastType !== "hourly" && (
            <DateInput
              id={`${title}-end`}
              label="End Date"
              value={endDate}
              onChange={setEndDate}
            />
          )}
        </div>

        {/* Substation */}
        <div>
          <label className="text-xs font-medium text-zinc-400 uppercase tracking-wider block mb-1.5">
            Substation Name
          </label>
          <input
            type="text"
            value={substation}
            onChange={(e) => setSubstation(e.target.value)}
            placeholder="e.g. Delhi Grid Station"
            className="w-full rounded-xl bg-white/5 border border-white/10 text-white text-sm px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all placeholder:text-zinc-600"
          />
        </div>

        {/* Generate */}
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

        {selectedType && (
          <p className="text-xs text-zinc-600">
            <span className={`font-medium ${selectedType.accent}`}>{selectedType.label}:</span>{" "}
            {selectedType.description}
          </p>
        )}
      </div>

      {/* Chart */}
      <div className="border-t border-white/5 mt-5 pt-5">
        {lastGenerated && (
          <p className="text-xs text-zinc-600 mb-3">
            {getForecastTypeLabel(forecastType)} · Generated {lastGenerated.toLocaleTimeString()} ·{" "}
            {substation}
          </p>
        )}
        <ForecastChart data={predictions} forecastType={forecastType} loading={loading} />
        {summary && !loading && <SummaryStrip summary={summary} unit={unit} />}
      </div>
    </div>
  );
}

// ─── Main Page ──────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const { addToast } = useToast();

  const [hasResult, setHasResult] = useState(false);
  const onResult = useCallback(() => setHasResult(true), []);

  // ── Notification state ────────────────────────────────────────────────────
  const ALERTS_PREF_KEY = "loadsense-alerts";
  const [notifPermission, setNotifPermission] = useState("default");
  const [notifEnabled, setNotifEnabled] = useState(false);
  const [peakThresholdPct, setPeakThresholdPct] = useState(6);
  const [alertLog, setAlertLog] = useState([]);
  const [currentAlert, setCurrentAlert] = useState(null); // drives the top banner

  // On mount: sync permission and restore the user's saved on/off preference.
  useEffect(() => {
    if (typeof Notification === "undefined") return;
    const perm = Notification.permission;
    setNotifPermission(perm);
    let pref = null;
    try {
      pref = localStorage.getItem(ALERTS_PREF_KEY);
    } catch (_) {
      // localStorage may be unavailable (private mode / SSR) — fall back to off
    }
    // Enable on load only when permission is granted AND the user hasn't opted out.
    setNotifEnabled(perm === "granted" && pref !== "off");
  }, []);

  // Auto-dismiss the banner for WARNING alerts. Self-cleaning: the timer is
  // cleared on any alert change / unmount, and only dismisses if the SAME alert
  // is still showing — so a newer alert is never dismissed by a stale timer.
  useEffect(() => {
    if (!currentAlert || currentAlert.severity !== "WARNING") return;
    const id = currentAlert.id;
    const t = setTimeout(() => {
      setCurrentAlert((cur) => (cur && cur.id === id ? null : cur));
    }, 8000);
    return () => clearTimeout(t);
  }, [currentAlert]);

  const persistPref = (value) => {
    try {
      localStorage.setItem(ALERTS_PREF_KEY, value);
    } catch (_) {
      /* ignore unavailable storage */
    }
  };

  const handleToggleNotifications = async () => {
    if (typeof Notification === "undefined") {
      addToast("Push notifications are not supported in your browser.", "warning");
      return;
    }
    if (notifEnabled) {
      setNotifEnabled(false);
      persistPref("off");
      addToast("Peak demand alerts disabled.", "info", 3000);
      return;
    }
    if (Notification.permission === "denied") {
      addToast("Notifications are blocked. Enable them in your browser settings, then reload.", "warning");
      return;
    }
    let perm = Notification.permission;
    if (perm !== "granted") {
      perm = await Notification.requestPermission();
      setNotifPermission(perm);
    }
    if (perm === "granted") {
      setNotifEnabled(true);
      persistPref("on");
      addToast("Peak demand alerts enabled. In-app banners always show; browser pop-ups appear when your OS allows them.", "success", 4500);
    } else {
      addToast("Notification permission denied.", "warning");
    }
  };

  const addAlert = useCallback((alert) => {
    setAlertLog((prev) => [alert, ...prev].slice(0, 20));
    // Don't let a WARNING bury a CRITICAL that's still on the banner.
    setCurrentAlert((prev) =>
      prev && prev.severity === "CRITICAL" && alert.severity !== "CRITICAL" ? prev : alert
    );
  }, []);

  const dismissCurrentAlert = () => setCurrentAlert(null);

  const dismissAlert = (id) => {
    setAlertLog((prev) => prev.filter((a) => a.id !== id));
  };

  // ─────────────────────────────────────────────────────────────────────────

  const handlePrint = () => window.print();

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="space-y-8 print-root"
    >
      {/* Print stylesheet + countdown animation */}
      <style dangerouslySetInnerHTML={{ __html: `
        .print-only { display: none; }
        @keyframes countdown-bar { from { width: 100%; } to { width: 0%; } }
        @media print {
          @page { margin: 14mm; }
          html, body { background: #fff !important; }
          body * { visibility: hidden !important; }
          .print-root, .print-root * { visibility: visible !important; }
          .print-root { position: absolute !important; left: 0 !important; top: 0 !important; width: 100% !important; }
          .no-print, .no-print * { display: none !important; }
          .print-only { display: block !important; }
          .print-grid { display: block !important; }
          .print-grid > * { margin-bottom: 18px !important; break-inside: avoid; page-break-inside: avoid; }
          .print-card { background: #fff !important; border: 1px solid #e5e7eb !important; box-shadow: none !important; }
          .print-root, .print-root .text-white, .print-root h1, .print-root h2, .print-root h3 { color: #111827 !important; }
          .print-root .text-zinc-400, .print-root .text-zinc-500, .print-root .text-zinc-600 { color: #6b7280 !important; }
        }
      ` }} />

      {/* ── Top Alert Banner ── */}
      <AnimatePresence>
        {currentAlert && (
          <motion.div
            key={currentAlert.id}
            initial={{ opacity: 0, y: -24, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -24, scale: 0.97 }}
            transition={{ type: "spring", damping: 22, stiffness: 300 }}
            className={cn(
              "no-print relative overflow-hidden rounded-2xl border p-5 flex items-start gap-4",
              currentAlert.severity === "CRITICAL"
                ? "bg-rose-950/60 border-rose-500/40 shadow-xl shadow-rose-950/40"
                : "bg-amber-950/60 border-amber-500/40 shadow-xl shadow-amber-950/30"
            )}
          >
            {/* Ambient glow */}
            <div
              className={cn(
                "absolute inset-0 pointer-events-none",
                currentAlert.severity === "CRITICAL" ? "bg-rose-600/8" : "bg-amber-600/8"
              )}
            />

            {/* Icon with optional ping for CRITICAL */}
            <div className={cn(
              "relative w-11 h-11 rounded-xl flex items-center justify-center shrink-0",
              currentAlert.severity === "CRITICAL" ? "bg-rose-500/20" : "bg-amber-500/20"
            )}>
              {currentAlert.severity === "CRITICAL" && (
                <span className="absolute inset-0 rounded-xl animate-ping bg-rose-500/25" />
              )}
              <AlertTriangle className={cn(
                "h-5 w-5 relative z-10",
                currentAlert.severity === "CRITICAL" ? "text-rose-400" : "text-amber-400"
              )} />
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0 relative">

              {/* Row 1 — severity + panel source badge + time */}
              <div className="flex flex-wrap items-center gap-2 mb-2">
                {/* Severity */}
                <span className={cn(
                  "text-[10px] font-extrabold px-2.5 py-0.5 rounded-full border tracking-widest",
                  currentAlert.severity === "CRITICAL"
                    ? "text-rose-300 bg-rose-500/15 border-rose-500/30"
                    : "text-amber-300 bg-amber-500/15 border-amber-500/30"
                )}>
                  {currentAlert.severity}
                </span>

                {/* Panel source — blue for Current Prediction, violet for Historical */}
                {currentAlert.panelTitle === "Current Prediction" ? (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full border text-[11px] font-semibold bg-blue-500/15 text-blue-300 border-blue-500/30">
                    <Zap className="h-3 w-3" />
                    Current Prediction
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full border text-[11px] font-semibold bg-violet-500/15 text-violet-300 border-violet-500/30">
                    <History className="h-3 w-3" />
                    Historical / Comparison
                  </span>
                )}

                <span className="text-xs text-zinc-500">
                  {getForecastTypeLabel(currentAlert.forecastType)} &middot; {currentAlert.time}
                </span>
              </div>

              {/* Row 2 — main alert message */}
              <p className="text-sm text-zinc-200 leading-relaxed">
                <span className="font-semibold text-white">{currentAlert.substation}</span>
                {" — peak load "}
                <span className={cn(
                  "font-bold text-base",
                  currentAlert.severity === "CRITICAL" ? "text-rose-400" : "text-amber-400"
                )}>
                  {Number(currentAlert.peak).toLocaleString(undefined, { maximumFractionDigits: 1 })}{" "}
                  {currentAlert.unit}
                </span>{" "}
                <span className="text-zinc-400">({currentAlert.deviation}% above average)</span>
                {" · Trend: "}
                <span className="capitalize text-zinc-300">{currentAlert.trend}</span>
              </p>

              {/* Countdown bar — WARNING only */}
              {currentAlert.severity === "WARNING" && (
                <>
                  <p className="text-[10px] text-zinc-500 mt-2">Auto-dismissing in 8 seconds</p>
                  <div className="mt-1.5 h-0.5 w-full rounded-full bg-white/5 overflow-hidden">
                    <div
                      key={currentAlert.id}
                      className="h-full rounded-full bg-amber-400/50"
                      style={{ animation: "countdown-bar 8s linear forwards" }}
                    />
                  </div>
                </>
              )}
            </div>

            {/* Dismiss */}
            <button
              onClick={dismissCurrentAlert}
              className="relative text-zinc-500 hover:text-white transition-colors shrink-0 p-1.5 rounded-lg hover:bg-white/10"
              title="Dismiss"
            >
              <X className="h-4 w-4" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Page Header ── */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center">
              <Zap className="w-4 h-4 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-white">Forecast Dashboard</h1>
          </div>
          <p className="text-sm text-zinc-500 ml-11">
            Configure each panel independently — compare a current forecast against any past or future date.
          </p>
        </div>
        <div className="no-print flex items-center gap-2.5 shrink-0">
          {/* Notification toggle */}
          <Button
            onClick={handleToggleNotifications}
            variant="outline"
            title={
              notifPermission === "denied"
                ? "Notifications blocked — enable in browser settings"
                : notifEnabled
                ? "Peak demand alerts ON — click to disable"
                : "Enable peak demand alerts"
            }
            className={cn(
              "rounded-xl gap-2 text-sm border transition-all",
              notifEnabled
                ? "bg-amber-500/10 border-amber-500/30 text-amber-300 hover:bg-amber-500/15 hover:text-amber-200"
                : notifPermission === "denied"
                ? "bg-white/5 border-white/10 text-zinc-600 cursor-not-allowed"
                : "bg-white/5 border-white/10 text-zinc-300 hover:bg-white/10 hover:text-white"
            )}
          >
            {notifEnabled ? (
              <BellRing className="h-4 w-4" />
            ) : notifPermission === "denied" ? (
              <BellOff className="h-4 w-4" />
            ) : (
              <Bell className="h-4 w-4" />
            )}
            {notifEnabled ? "Alerts On" : "Alerts Off"}
          </Button>

          <Button
            onClick={handlePrint}
            variant="outline"
            className="bg-white/5 border-white/10 text-zinc-300 hover:bg-white/10 hover:text-white rounded-xl gap-2 text-sm"
          >
            <Printer className="h-4 w-4" />
            Print
          </Button>
        </div>
      </div>

      {/* ── Notification Settings Bar (shown when alerts are enabled) ── */}
      {notifEnabled && (
        <div className="no-print flex flex-wrap items-center gap-4 rounded-xl bg-amber-950/20 border border-amber-700/25 px-4 py-3">
          <BellRing className="h-4 w-4 text-amber-400 shrink-0" />
          <p className="text-sm text-zinc-400">
            Browser alert fires when peak demand exceeds{" "}
            <span className="text-amber-300 font-semibold">{peakThresholdPct}%</span> above the period average.
          </p>
          <div className="flex items-center gap-3 ml-auto">
            <span className="text-xs text-zinc-500 whitespace-nowrap">Alert threshold</span>
            <input
              type="range"
              min={2}
              max={30}
              step={1}
              value={peakThresholdPct}
              onChange={(e) => setPeakThresholdPct(Number(e.target.value))}
              className="w-28 accent-amber-400 cursor-pointer"
            />
            <span className="text-xs font-semibold text-amber-300 w-9 text-right">{peakThresholdPct}%</span>
          </div>
        </div>
      )}

      {/* ── Two Independent Panels ── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 print-grid">
        <ForecastPanel
          title="Current Prediction"
          subtitle="Forecast for today or an upcoming date"
          icon={Zap}
          headerAccent={{ bg: "bg-blue-500/15", text: "text-blue-400" }}
          defaultType="hourly"
          onResult={onResult}
          notifEnabled={notifEnabled}
          peakThresholdPct={peakThresholdPct}
          onAlert={addAlert}
        />
        <ForecastPanel
          title="Historical / Comparison"
          subtitle="Pick any past or future date to compare"
          icon={History}
          headerAccent={{ bg: "bg-violet-500/15", text: "text-violet-400" }}
          defaultType="hourly"
          onResult={onResult}
          notifEnabled={notifEnabled}
          peakThresholdPct={peakThresholdPct}
          onAlert={addAlert}
        />
      </div>

      {/* ── Call to Action for AI Report ── */}
      <AnimatePresence>
        {hasResult && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ delay: 0.1 }}
            className="no-print rounded-2xl bg-gradient-to-r from-violet-950/40 to-blue-950/40 border border-violet-700/20 p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-violet-500/20 flex items-center justify-center shrink-0">
                <Sparkles className="h-5 w-5 text-violet-400" />
              </div>
              <div>
                <p className="font-semibold text-white text-sm">Generate AI Operational Report</p>
                <p className="text-xs text-zinc-500">
                  Get AI-powered insights, risk analysis, and action plan for these forecasts
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
      </AnimatePresence>

      {/* ── Session Alert Log ── */}
      {alertLog.length > 0 && (
        <div className="no-print space-y-3">
          <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider flex items-center gap-2">
            <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
            Peak Demand Alerts — This Session
            <span className="ml-auto text-[10px] font-normal text-zinc-600">
              {alertLog.length} alert{alertLog.length !== 1 ? "s" : ""}
            </span>
          </h3>
          <div className="space-y-2">
            {alertLog.map((alert) => (
              <motion.div
                key={alert.id}
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                className={cn(
                  "flex items-start gap-3 p-4 rounded-xl border text-sm",
                  alert.severity === "CRITICAL"
                    ? "bg-rose-950/20 border-rose-500/20"
                    : "bg-amber-950/20 border-amber-500/20"
                )}
              >
                <AlertTriangle
                  className={cn(
                    "h-4 w-4 shrink-0 mt-0.5",
                    alert.severity === "CRITICAL" ? "text-rose-400" : "text-amber-400"
                  )}
                />
                <div className="flex-1 min-w-0">
                  <div className="flex flex-wrap items-center gap-2 mb-1">
                    <span
                      className={cn(
                        "text-[9px] font-extrabold px-2 py-0.5 rounded-full border tracking-wider",
                        alert.severity === "CRITICAL"
                          ? "text-rose-400 bg-rose-500/10 border-rose-500/20"
                          : "text-amber-400 bg-amber-500/10 border-amber-500/20"
                      )}
                    >
                      {alert.severity}
                    </span>
                    {alert.panelTitle === "Current Prediction" ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[10px] font-semibold bg-blue-500/15 text-blue-300 border-blue-500/30">
                        <Zap className="h-2.5 w-2.5" />
                        Current Prediction
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[10px] font-semibold bg-violet-500/15 text-violet-300 border-violet-500/30">
                        <History className="h-2.5 w-2.5" />
                        Historical / Comparison
                      </span>
                    )}
                    <span className="text-xs text-zinc-500">
                      {getForecastTypeLabel(alert.forecastType)} &middot; {alert.time}
                    </span>
                  </div>
                  <p className="text-zinc-300">
                    <span className="font-semibold">{alert.substation}</span>
                    {" — "}Peak load{" "}
                    <span
                      className={cn(
                        "font-bold",
                        alert.severity === "CRITICAL" ? "text-rose-400" : "text-amber-400"
                      )}
                    >
                      {Number(alert.peak).toLocaleString(undefined, { maximumFractionDigits: 1 })}{" "}
                      {alert.unit}
                    </span>{" "}
                    <span className="text-zinc-500">({alert.deviation}% above avg)</span>
                    {" · "}Trend:{" "}
                    <span className="capitalize text-zinc-400">{alert.trend}</span>
                  </p>
                </div>
                <button
                  onClick={() => dismissAlert(alert.id)}
                  className="text-zinc-600 hover:text-zinc-400 transition-colors shrink-0 p-0.5"
                  title="Dismiss"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </motion.div>
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
}
