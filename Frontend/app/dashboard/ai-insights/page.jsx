"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FileBarChart2,
  Calendar,
  Loader2,
  Sparkles,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  AlertTriangle,
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
  Zap,
  Bot,
  ShieldAlert,
  ListChecks,
  Route,
  ArrowRight,
  Printer,
  LayoutGrid,
  FileText,
  CheckCircle2,
  Square,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/toast";
import { SummaryCardsSkeleton, ReportSkeleton } from "@/components/ui/skeleton";
import {
  toMidnightEpoch,
  validateDateRange,
  parseReport,
  getValueUnit,
  getForecastTypeLabel,
} from "@/lib/forecast";
import { cn } from "@/lib/utils";

// ─── Constants ──────────────────────────────────────────────────────────────

const FORECAST_TYPES = ["hourly", "weekly", "monthly", "quarterly"];

// ─── Date Input Component ───────────────────────────────────────────────────

function DateInput({ label, value, onChange, id }) {
  return (
    <div className="flex flex-col gap-1.5 no-print">
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
          className="w-full rounded-xl bg-white/5 border border-white/10 text-white text-sm pl-10 pr-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-violet-500/50 focus:border-violet-500/50 transition-all [color-scheme:dark]"
        />
      </div>
    </div>
  );
}

// ─── Summary Card Component ─────────────────────────────────────────────────

function SummaryCard({ label, value, unit, icon: Icon, accent }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl bg-white/[0.03] border border-white/8 p-5 hover:bg-white/[0.05] transition-colors"
    >
      <div className="flex items-start justify-between mb-3">
        <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">{label}</p>
        <div className={`p-1 rounded-md bg-white/5`}>
          <Icon className={`h-3.5 w-3.5 ${accent}`} />
        </div>
      </div>
      <p className="text-2xl font-bold text-white tabular-nums">
        {typeof value === "number"
          ? value.toLocaleString(undefined, { maximumFractionDigits: 2 })
          : String(value)}
        {unit && <span className="text-sm font-normal text-zinc-500 ml-1">{unit}</span>}
      </p>
    </motion.div>
  );
}

// ─── Trend Card Component ───────────────────────────────────────────────────

function TrendCard({ trend }) {
  const map = {
    rising: { icon: TrendingUp, label: "Rising", accent: "text-rose-400", bg: "bg-rose-500/10", border: "border-rose-500/20" },
    falling: { icon: TrendingDown, label: "Falling", accent: "text-blue-400", bg: "bg-blue-500/10", border: "border-blue-500/20" },
    stable: { icon: Minus, label: "Stable", accent: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/20" },
  };
  const { icon: Icon, label, accent, bg, border } = map[trend] || map.stable;
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-2xl border ${border} ${bg} p-5`}
    >
      <div className="flex items-start justify-between mb-3">
        <p className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Trend</p>
        <Icon className={`h-4 w-4 ${accent}`} />
      </div>
      <p className={`text-2xl font-bold ${accent}`}>{label}</p>
    </motion.div>
  );
}

// ─── Severity Level Configuration ──────────────────────────────────────────

const getRiskConfig = (text) => {
  const t = text.toLowerCase();
  if (t.includes("high") || t.includes("critical") || t.includes("danger") || t.includes("overload") || t.includes("risk")) {
    return { label: "CRITICAL", color: "text-rose-400 bg-rose-500/10 border-rose-500/20" };
  }
  if (t.includes("medium") || t.includes("moderate") || t.includes("warning") || t.includes("alert")) {
    return { label: "WARNING", color: "text-amber-400 bg-amber-500/10 border-amber-500/20" };
  }
  return { label: "OPERATIONAL", color: "text-blue-400 bg-blue-500/10 border-blue-500/20" };
};

// ─── Formal Document View Component ─────────────────────────────────────────

function DocumentReportView({ reportData, parsedReport, forecastType, startDate, endDate, unit }) {
  return (
    <div className="printable-report rounded-2xl bg-white/[0.015] border border-white/8 p-6 sm:p-8 space-y-6 text-left">
      {/* Letterhead Header */}
      <div className="border-b border-zinc-800 pb-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight uppercase">BEOPS System Operational Report</h1>
          <p className="text-xs text-zinc-500 mt-1">
            Grid Substation Forecasting & Operational Analysis System
          </p>
        </div>
        <div className="text-left md:text-right text-xs text-zinc-400 space-y-1">
          <p><span className="text-zinc-600 font-medium">Substation:</span> {reportData.substation_name}</p>
          <p><span className="text-zinc-600 font-medium">Period:</span> {getForecastTypeLabel(forecastType)} ({startDate} {forecastType !== "hourly" && `to ${endDate}`})</p>
          <p><span className="text-zinc-600 font-medium">Generated:</span> {new Date().toLocaleString()}</p>
        </div>
      </div>

      {/* Metrics Summary Grid */}
      {reportData.summary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 border-b border-zinc-800 pb-5">
          <div>
            <span className="text-[10px] text-zinc-500 uppercase font-semibold">Average Load</span>
            <p className="text-lg font-bold text-white mt-0.5">
              {reportData.summary.average.toLocaleString(undefined, { maximumFractionDigits: 2 })} {unit}
            </p>
          </div>
          <div>
            <span className="text-[10px] text-zinc-500 uppercase font-semibold">Peak Load</span>
            <p className="text-lg font-bold text-rose-400 mt-0.5">
              {reportData.summary.maximum.toLocaleString(undefined, { maximumFractionDigits: 2 })} {unit}
            </p>
          </div>
          <div>
            <span className="text-[10px] text-zinc-500 uppercase font-semibold">Minimum Load</span>
            <p className="text-lg font-bold text-sky-400 mt-0.5">
              {reportData.summary.minimum.toLocaleString(undefined, { maximumFractionDigits: 2 })} {unit}
            </p>
          </div>
          <div>
            <span className="text-[10px] text-zinc-500 uppercase font-semibold">Total Energy</span>
            <p className="text-lg font-bold text-amber-400 mt-0.5">
              {reportData.summary.total.toLocaleString(undefined, { maximumFractionDigits: 2 })} {unit}
            </p>
          </div>
        </div>
      )}

      {/* Document Sections */}
      <div className="space-y-6">
        {/* Executive Summary */}
        {parsedReport.executive_summary && (
          <div className="space-y-2">
            <h2 className="text-xs font-bold text-zinc-400 uppercase tracking-widest">1. Executive Summary</h2>
            <p className="text-sm text-zinc-300 leading-relaxed font-serif italic pl-4 border-l border-zinc-700">
              "{parsedReport.executive_summary}"
            </p>
          </div>
        )}

        {/* Observations */}
        {parsedReport.key_observations && parsedReport.key_observations.length > 0 && (
          <div className="space-y-2">
            <h2 className="text-xs font-bold text-zinc-400 uppercase tracking-widest">2. Key Observations</h2>
            <ul className="space-y-2 pl-4">
              {parsedReport.key_observations.map((item, i) => (
                <li key={i} className="text-sm text-zinc-300 leading-relaxed list-disc list-inside">
                  {item}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Risks */}
        {parsedReport.operational_risks && parsedReport.operational_risks.length > 0 && (
          <div className="space-y-2">
            <h2 className="text-xs font-bold text-zinc-400 uppercase tracking-widest">3. Operational Risks</h2>
            <ul className="space-y-2 pl-4">
              {parsedReport.operational_risks.map((item, i) => (
                <li key={i} className="text-sm text-zinc-300 leading-relaxed list-disc list-inside">
                  {item}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Plan of Action */}
        {parsedReport.plan_of_action && parsedReport.plan_of_action.length > 0 && (
          <div className="space-y-2">
            <h2 className="text-xs font-bold text-zinc-400 uppercase tracking-widest">4. Operational SOP / Action Plan</h2>
            <ul className="space-y-2 pl-4">
              {parsedReport.plan_of_action.map((item, i) => (
                <li key={i} className="text-sm text-zinc-300 leading-relaxed list-decimal list-inside">
                  {item}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Next Steps */}
        {parsedReport.next_steps && parsedReport.next_steps.length > 0 && (
          <div className="space-y-2">
            <h2 className="text-xs font-bold text-zinc-400 uppercase tracking-widest">5. Timeline / Next Steps</h2>
            <ul className="space-y-2 pl-4">
              {parsedReport.next_steps.map((item, i) => (
                <li key={i} className="text-sm text-zinc-300 leading-relaxed list-disc list-inside">
                  {item}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Sign-off / Disclaimer */}
      <div className="border-t border-zinc-800 pt-5 mt-10 flex flex-col sm:flex-row sm:items-center justify-between text-[10px] text-zinc-500">
        <p>Report Generated Autonomously by BEOPS AI Forecasting Core.</p>
        <p className="mt-1 sm:mt-0">© {new Date().getFullYear()} BEOPS Grid Operations. All rights reserved.</p>
      </div>
    </div>
  );
}

// ─── Main Component ──────────────────────────────────────────────────────────

export default function AIReportPage() {
  const { addToast } = useToast();

  const today = new Date().toISOString().slice(0, 10);
  const [forecastType, setForecastType] = useState("hourly");
  const [startDate, setStartDate] = useState(today);
  const [endDate, setEndDate] = useState(today);
  const [substationName, setSubstationName] = useState("Grid Substation");

  const [loading, setLoading] = useState(false);
  const [reportData, setReportData] = useState(null);
  const [parsedReport, setParsedReport] = useState(null);
  const [copied, setCopied] = useState(false);
  const [viewMode, setViewMode] = useState("dashboard"); // "dashboard" or "document"
  const [checkedActions, setCheckedActions] = useState({});

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
    setReportData(null);
    setParsedReport(null);
    setCheckedActions({});

    const startEpoch = toMidnightEpoch(start);
    const endEpoch = toMidnightEpoch(end);

    try {
      const res = await fetch("/api/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          report_type: forecastType,
          start_date: startEpoch,
          end_date: endEpoch,
          substation_name: substationName || "Grid Substation",
        }),
      });

      const data = await res.json();

      if (!res.ok || data.error) {
        addToast(data.error || "Report generation failed. Please try again.", "error");
        return;
      }

      setReportData(data);
      setParsedReport(parseReport(data.report));
      addToast("AI operational report generated successfully.", "success", 3000);
    } catch (err) {
      addToast(`Connection error: ${err.message}`, "error");
    } finally {
      setLoading(false);
    }
  }, [forecastType, startDate, endDate, substationName, addToast]);

  const handleCopy = () => {
    if (!parsedReport) return;
    const text = [
      `AI OPERATIONAL REPORT — ${reportData?.substation_name}`,
      `Type: ${getForecastTypeLabel(forecastType)}`,
      `Period: ${startDate} ${forecastType !== "hourly" ? `to ${endDate}` : ""}`,
      "",
      "EXECUTIVE SUMMARY",
      parsedReport.executive_summary,
      "",
      "KEY OBSERVATIONS",
      ...parsedReport.key_observations.map((s) => `• ${s}`),
      "",
      "OPERATIONAL RISKS",
      ...parsedReport.operational_risks.map((s) => `• ${s}`),
      "",
      "PLAN OF ACTION",
      ...parsedReport.plan_of_action.map((s) => `• ${s}`),
      "",
      "NEXT STEPS",
      ...parsedReport.next_steps.map((s) => `• ${s}`),
    ].join("\n");

    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const handlePrint = () => {
    window.print();
  };

  const toggleAction = (idx) => {
    setCheckedActions((prev) => ({
      ...prev,
      [idx]: !prev[idx],
    }));
  };

  const unit = getValueUnit(forecastType);

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="space-y-8"
    >
      {/* Dynamic Global print css wrapper */}
      <style dangerouslySetInnerHTML={{ __html: `
        @media print {
          /* Hide everything on the page */
          body, html, #sidebar-container, header, nav, aside, footer, button, .no-print {
            display: none !important;
            visibility: hidden !important;
          }
          /* Ensure content printable-area has full width */
          main, .content-root, .main-layout {
            padding: 0 !important;
            margin: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
          }
          /* Show only the printable report */
          .printable-report {
            display: block !important;
            visibility: visible !important;
            position: absolute !important;
            left: 0 !important;
            top: 0 !important;
            width: 100% !important;
            color: #111827 !important;
            background-color: #ffffff !important;
            padding: 40px !important;
            box-shadow: none !important;
            border: none !important;
          }
          .printable-report * {
            color: #111827 !important;
            border-color: #e5e7eb !important;
          }
          .printable-report .text-white { color: #111827 !important; }
          .printable-report .text-zinc-300 { color: #374151 !important; }
          .printable-report .text-zinc-400 { color: #4b5563 !important; }
          .printable-report .text-zinc-500 { color: #6b7280 !important; }
          .printable-report .text-zinc-600 { color: #9ca3af !important; }
          .printable-report h1, .printable-report h2, .printable-report h3, .printable-report h4 {
            color: #111827 !important;
          }
          .printable-report border-zinc-800 {
            border-color: #e5e7eb !important;
          }
        }
      ` }} />

      {/* ── Page Header ── */}
      <div className="no-print">
        <div className="flex items-center gap-3 mb-1">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center">
            <FileBarChart2 className="w-4 h-4 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-white">AI Operational Report</h1>
        </div>
        <p className="text-sm text-zinc-500 ml-11">
          Generate AI-powered insights and operational recommendations for substation operators
        </p>
      </div>

      {/* ── LLM Notice ── */}
      <div className="flex items-start gap-3 rounded-xl bg-violet-950/30 border border-violet-700/20 px-4 py-3 no-print">
        <Bot className="h-5 w-5 text-violet-400 shrink-0 mt-0.5 animate-pulse" />
        <p className="text-sm text-zinc-400">
          Reports are generated using{" "}
          <span className="text-violet-300 font-medium">Qwen/Qwen2.5-7B-Instruct</span> via
          HuggingFace. Generation may take up to{" "}
          <span className="text-zinc-300 font-medium">2 minutes</span>. A rule-based fallback is
          used if the model is unavailable.
        </p>
      </div>

      {/* ── Filter Panel ── */}
      <div className="rounded-2xl bg-white/[0.03] border border-white/8 p-6 no-print">
        <h2 className="text-sm font-semibold text-zinc-300 mb-5 flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-violet-400" />
          Report Configuration
        </h2>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-end">
          {/* Forecast Type */}
          <div className="lg:col-span-3">
            <p className="text-xs font-medium text-zinc-400 uppercase tracking-wider mb-2">
              Report Type
            </p>
            <div className="grid grid-cols-2 gap-2">
              {FORECAST_TYPES.map((ft) => (
                <button
                  key={ft}
                  onClick={() => handleForecastTypeChange(ft)}
                  className={`rounded-xl px-3 py-2.5 text-sm font-medium border transition-all duration-200 capitalize ${
                    forecastType === ft
                      ? "bg-violet-500/15 border-violet-500/30 text-violet-300 animate-pulse"
                      : "bg-white/[0.02] border-white/8 text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-300"
                  }`}
                >
                  {ft}
                </button>
              ))}
            </div>
          </div>

          {/* Dates */}
          <div className="lg:col-span-4 grid grid-cols-2 gap-3">
            <DateInput
              id="report-start-date"
              label={forecastType === "hourly" ? "Date" : "Start Date"}
              value={startDate}
              onChange={handleStartDateChange}
            />
            {forecastType !== "hourly" && (
              <DateInput
                id="report-end-date"
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
              className="w-full rounded-xl bg-white/5 border border-white/10 text-white text-sm px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-violet-500/50 focus:border-violet-500/50 transition-all placeholder:text-zinc-600"
            />
          </div>

          {/* Generate Button */}
          <div className="lg:col-span-2">
            <Button
              onClick={handleGenerate}
              disabled={loading}
              className="w-full bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-500 hover:to-purple-500 text-white font-semibold rounded-xl py-2.5 h-auto disabled:opacity-60 transition-all shadow-lg shadow-violet-900/30"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Generating…
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4" />
                  Generate
                </span>
              )}
            </Button>
          </div>
        </div>
      </div>

      {/* ── Loading State ── */}
      {loading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-6 no-print"
        >
          <div className="rounded-2xl bg-violet-950/20 border border-violet-800/20 p-6 flex flex-col items-center gap-4">
            <div className="relative">
              <div className="w-14 h-14 rounded-full bg-violet-500/10 flex items-center justify-center">
                <Bot className="w-7 h-7 text-violet-400 animate-pulse" />
              </div>
              <Loader2 className="absolute -top-1 -right-1 w-5 h-5 text-violet-400 animate-spin" />
            </div>
            <div className="text-center">
              <p className="font-semibold text-white">AI is analysing forecast data…</p>
              <p className="text-sm text-zinc-500 mt-1">
                Generating operational report using Qwen2.5. This may take up to 2 minutes.
              </p>
            </div>
          </div>
          <SummaryCardsSkeleton />
          <ReportSkeleton />
        </motion.div>
      )}

      {/* ── Report Results ── */}
      <AnimatePresence>
        {reportData && !loading && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* Report Utility Action Bar */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 rounded-2xl bg-white/[0.03] border border-white/8 p-5 no-print">
              <div>
                <h2 className="text-base font-semibold text-white">
                  {reportData.substation_name}
                </h2>
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-xs text-zinc-500 capitalize">
                    {getForecastTypeLabel(forecastType)} ·{" "}
                  </span>
                  <span
                    className={`text-[10px] px-2.5 py-0.5 rounded-full font-semibold ${
                      reportData.report_source === "huggingface_router"
                        ? "bg-violet-500/15 text-violet-300 border border-violet-500/20 animate-pulse"
                        : "bg-amber-500/15 text-amber-300 border border-amber-500/20"
                    }`}
                  >
                    {reportData.report_source === "huggingface_router"
                      ? "🤖 Qwen AI Active"
                      : "📋 Rule-Based Core"}
                  </span>
                </div>
              </div>

              {/* Action Buttons & Switcher */}
              <div className="flex flex-wrap items-center gap-2.5">
                {/* View Switcher */}
                <div className="flex items-center gap-1 p-1 rounded-xl bg-white/[0.04] border border-white/8">
                  <button
                    onClick={() => setViewMode("dashboard")}
                    className={cn(
                      "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200",
                      viewMode === "dashboard"
                        ? "bg-white/10 text-white shadow-sm"
                        : "text-zinc-500 hover:text-zinc-300"
                    )}
                  >
                    <LayoutGrid className="w-3.5 h-3.5" />
                    Dashboard
                  </button>
                  <button
                    onClick={() => setViewMode("document")}
                    className={cn(
                      "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200",
                      viewMode === "document"
                        ? "bg-white/10 text-white shadow-sm"
                        : "text-zinc-500 hover:text-zinc-300"
                    )}
                  >
                    <FileText className="w-3.5 h-3.5" />
                    Document
                  </button>
                </div>

                {/* Print/Download PDF Button */}
                <Button
                  onClick={handlePrint}
                  className="bg-violet-600 hover:bg-violet-500 text-white rounded-xl gap-2 text-xs py-1.5 px-3.5 shadow-md shadow-violet-900/35 border-none"
                >
                  <Printer className="h-3.5 w-3.5" />
                  Print / Save PDF
                </Button>

                {/* Copy Button */}
                <Button
                  onClick={handleCopy}
                  variant="outline"
                  className="bg-white/5 border-white/10 text-zinc-300 hover:bg-white/10 hover:text-white rounded-xl gap-2 text-xs py-1.5 px-3.5"
                >
                  {copied ? (
                    <>
                      <Check className="h-3.5 w-3.5 text-emerald-400" />
                      Copied!
                    </>
                  ) : (
                    <>
                      <Copy className="h-3.5 w-3.5" />
                      Copy Text
                    </>
                  )}
                </Button>
              </div>
            </div>

            {/* Summary Statistics Cards */}
            {reportData.summary && (
              <div className="space-y-3 no-print">
                <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider flex items-center gap-2">
                  <Activity className="h-3.5 w-3.5 text-zinc-500" />
                  Operational Load Summary
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                  <SummaryCard
                    label="Average"
                    value={reportData.summary.average}
                    unit={unit}
                    icon={Activity}
                    accent="text-blue-400"
                  />
                  <SummaryCard
                    label="Peak"
                    value={reportData.summary.maximum}
                    unit={unit}
                    icon={TrendingUp}
                    accent="text-rose-400"
                  />
                  <SummaryCard
                    label="Minimum"
                    value={reportData.summary.minimum}
                    unit={unit}
                    icon={TrendingDown}
                    accent="text-sky-400"
                  />
                  <SummaryCard
                    label="Total"
                    value={reportData.summary.total}
                    unit={unit}
                    icon={Zap}
                    accent="text-amber-400"
                  />
                  <TrendCard trend={reportData.summary.trend} />
                </div>
              </div>
            )}

            {/* ── Interactive Dashboard Mode / Dynamic Grid Layout ── */}
            {parsedReport && viewMode === "dashboard" && (
              <div className="space-y-6 no-print">
                {/* Executive Summary Header */}
                {parsedReport.executive_summary && (
                  <div className="rounded-2xl border border-violet-500/20 bg-gradient-to-r from-violet-950/20 to-blue-950/20 p-6 relative overflow-hidden">
                    <div className="absolute right-0 top-0 w-64 h-64 bg-violet-500/5 rounded-full blur-3xl -z-10" />
                    <div className="flex items-center gap-2.5 mb-3">
                      <Bot className="h-4.5 w-4.5 text-violet-400 animate-pulse" />
                      <h3 className="font-bold text-white text-xs uppercase tracking-wider">AI Executive Brief</h3>
                    </div>
                    <p className="text-sm text-zinc-300 leading-relaxed italic font-serif">
                      "{parsedReport.executive_summary}"
                    </p>
                  </div>
                )}

                {/* Dashboard Grid columns */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Left Column */}
                  <div className="space-y-6">
                    {/* Key Observations */}
                    {parsedReport.key_observations && parsedReport.key_observations.length > 0 && (
                      <div className="rounded-2xl border border-white/8 bg-white/[0.015] p-5 space-y-4">
                        <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-wider flex items-center gap-2 border-b border-white/5 pb-3">
                          <Activity className="h-4 w-4 text-emerald-400" />
                          Key Observations
                        </h3>
                        <ul className="space-y-3.5">
                          {parsedReport.key_observations.map((obs, idx) => (
                            <li key={idx} className="flex items-start gap-3">
                              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-2 shrink-0 animate-pulse" />
                              <p className="text-sm text-zinc-300 leading-relaxed">{obs}</p>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Plan of Action SOP (Interactive Checklist) */}
                    {parsedReport.plan_of_action && parsedReport.plan_of_action.length > 0 && (
                      <div className="rounded-2xl border border-white/8 bg-white/[0.015] p-5 space-y-4">
                        <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-wider flex items-center gap-2 border-b border-white/5 pb-3">
                          <ListChecks className="h-4 w-4 text-amber-400" />
                          Operational Action Checklist
                        </h3>
                        <p className="text-[10px] text-zinc-500 italic mb-1">Click items below to mark off SOP actions completed:</p>
                        <div className="space-y-2.5">
                          {parsedReport.plan_of_action.map((act, idx) => {
                            const isChecked = checkedActions[idx];
                            return (
                              <button
                                key={idx}
                                onClick={() => toggleAction(idx)}
                                className={cn(
                                  "w-full text-left flex items-start gap-3 p-3 rounded-xl border transition-all duration-200 group",
                                  isChecked
                                    ? "bg-emerald-950/10 border-emerald-950/20"
                                    : "bg-white/[0.01] border-white/5 hover:bg-white/[0.02] hover:border-white/10"
                                )}
                              >
                                <div className="mt-0.5 shrink-0">
                                  {isChecked ? (
                                    <CheckCircle2 className="h-4.5 w-4.5 text-emerald-400 fill-emerald-500/10" />
                                  ) : (
                                    <Square className="h-4.5 w-4.5 text-zinc-600 group-hover:text-zinc-400 transition-colors" />
                                  )}
                                </div>
                                <p className={cn(
                                  "text-sm leading-relaxed transition-colors duration-200",
                                  isChecked ? "text-zinc-500 line-through" : "text-zinc-300"
                                )}>
                                  {act}
                                </p>
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Right Column */}
                  <div className="space-y-6">
                    {/* Operational Risks with Severity badge analyzer */}
                    {parsedReport.operational_risks && parsedReport.operational_risks.length > 0 && (
                      <div className="rounded-2xl border border-white/8 bg-white/[0.015] p-5 space-y-4">
                        <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-wider flex items-center gap-2 border-b border-white/5 pb-3">
                          <ShieldAlert className="h-4 w-4 text-rose-400" />
                          Risk Assessment
                        </h3>
                        <div className="space-y-3.5">
                          {parsedReport.operational_risks.map((risk, idx) => {
                            const riskCfg = getRiskConfig(risk);
                            return (
                              <div key={idx} className="flex flex-col gap-2 p-3 rounded-xl bg-white/[0.01] border border-white/5">
                                <div className="flex items-center justify-between">
                                  <span className={`text-[9px] font-extrabold px-2.5 py-0.5 rounded-full border tracking-wider ${riskCfg.color}`}>
                                    {riskCfg.label}
                                  </span>
                                  <span className="text-[10px] text-zinc-600">Risk ID: #{idx + 1}</span>
                                </div>
                                <p className="text-sm text-zinc-300 leading-relaxed">{risk}</p>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* Next Steps (Timeline View) */}
                    {parsedReport.next_steps && parsedReport.next_steps.length > 0 && (
                      <div className="rounded-2xl border border-white/8 bg-white/[0.015] p-5 space-y-4">
                        <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-wider flex items-center gap-2 border-b border-white/5 pb-3">
                          <Route className="h-4 w-4 text-violet-400" />
                          Implementation Steps
                        </h3>
                        <div className="relative pl-6 space-y-6 border-l border-white/10 ml-3 py-1">
                          {parsedReport.next_steps.map((step, idx) => (
                            <div key={idx} className="relative group">
                              {/* Outer timeline indicator */}
                              <div className="absolute -left-[31px] top-0 w-4.5 h-4.5 rounded-full bg-[#0d0d14] border-2 border-violet-500 flex items-center justify-center group-hover:scale-115 transition-all">
                                <span className="text-[8px] font-extrabold text-violet-400">{idx + 1}</span>
                              </div>
                              <h4 className="text-xs font-semibold text-violet-400 uppercase tracking-wider mb-1">
                                Stage {idx + 1}
                              </h4>
                              <p className="text-sm text-zinc-300 leading-relaxed">{step}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* ── Document/Standard Paper View (Visible on screen if selected, always visible to printer) ── */}
            {parsedReport && (viewMode === "document" || typeof window === "undefined") && (
              <div className="no-print">
                <DocumentReportView
                  reportData={reportData}
                  parsedReport={parsedReport}
                  forecastType={forecastType}
                  startDate={startDate}
                  endDate={endDate}
                  unit={unit}
                />
              </div>
            )}

            {/* Always-visible printable wrapper strictly for window.print() output */}
            {parsedReport && (
              <div className="hidden">
                <DocumentReportView
                  reportData={reportData}
                  parsedReport={parsedReport}
                  forecastType={forecastType}
                  startDate={startDate}
                  endDate={endDate}
                  unit={unit}
                />
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}