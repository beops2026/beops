"use client";

import { motion } from "framer-motion";
import {
  BarChart3, TrendingUp, Activity, Cpu, Brain,
  Shield, Thermometer, Radio, AlertTriangle, History,
  FlaskConical, Info,
} from "lucide-react";
import AnalyticsChart from "@/app/chart/AnalyticsChart";
import {
  predictionAccuracyData,
  featureImportanceData,
  environmentalImpactData,
  sensorContributionData,
  modelConfidenceData,
  correlationData,
  resourceUtilisationData,
  systemHealthData,
  riskRadarData,
  historicalPredictionData,
} from "@/lib/analyticsData";

// ─── Color Palette ─────────────────────────────────────────────────────────────
const INDIGO = "#6366f1";
const CYAN = "#22d3ee";
const VIOLET = "#a78bfa";
const AMBER = "#f59e0b";
const EMERALD = "#34d399";
const ROSE = "#fb7185";

// ─── KPI Cards ────────────────────────────────────────────────────────────────
const kpis = [
  { label: "Model Accuracy", value: "96.8%", change: "+0.4%", icon: Brain, color: "text-indigo-400", bg: "bg-indigo-500/10", border: "border-indigo-500/20", up: true },
  { label: "Avg MAPE", value: "3.21%", change: "-0.18%", icon: Activity, color: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/20", up: true },
  { label: "System Health", value: "91.4", change: "+1.2", icon: Shield, color: "text-cyan-400", bg: "bg-cyan-500/10", border: "border-cyan-500/20", up: true },
  { label: "Prediction Drift", value: "1.07%", change: "+0.03%", icon: AlertTriangle, color: "text-amber-400", bg: "bg-amber-500/10", border: "border-amber-500/20", up: false },
];

function KPICard({ kpi }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-2xl border ${kpi.border} bg-white/[0.025] p-5 flex items-start gap-4`}
    >
      <div className={`w-10 h-10 rounded-xl ${kpi.bg} flex items-center justify-center shrink-0`}>
        <kpi.icon className={`h-5 w-5 ${kpi.color}`} />
      </div>
      <div>
        <p className="text-xs text-zinc-500 mb-1">{kpi.label}</p>
        <p className="text-2xl font-bold text-white tabular-nums">{kpi.value}</p>
        <p className={`text-xs font-medium mt-0.5 ${kpi.up ? "text-emerald-400" : "text-rose-400"}`}>
          {kpi.change} vs last week
        </p>
      </div>
    </motion.div>
  );
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

// ─── Main Page ─────────────────────────────────────────────────────────────────
export default function AnalyticsPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="space-y-10"
    >
      {/* Page Header */}
      <div>
        <div className="flex items-center gap-3 mb-1">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
            <BarChart3 className="w-4 h-4 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-white">Analytics</h1>
        </div>
        <p className="text-sm text-zinc-500 ml-11">
          Model performance, environmental analysis, and system health metrics
        </p>
      </div>

      {/* Dummy Data Notice */}
      <div className="flex items-start gap-3 rounded-xl bg-amber-950/30 border border-amber-700/20 px-4 py-3">
        <FlaskConical className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
        <p className="text-sm text-zinc-400">
          <span className="text-amber-300 font-medium">Demo Data — </span>
          These visualisations use realistic simulated data based on Delhi grid operational ranges.
          They will be replaced with live backend data once the analytics API endpoints are ready.
        </p>
      </div>

      {/* ── KPI Cards ── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {kpis.map((kpi, i) => (
          <motion.div key={kpi.label} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.07 }}>
            <KPICard kpi={kpi} />
          </motion.div>
        ))}
      </div>

      {/* ── Section 1: Model Performance ── */}
      <div>
        <SectionHeader icon={Brain} title="Model Performance" subtitle="Prediction accuracy and historical comparison over 30 days" />
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
          {/* Prediction Accuracy Trend */}
          <AnalyticsChart
            title="Prediction Accuracy Trend"
            subtitle="Daily MAPE & accuracy over 30 days"
            badge="Time-series"
            badgeColor="bg-indigo-500/15 text-indigo-300 border-indigo-500/20"
            allowedTypes={["line", "area", "bar", "composed", "stepped"]}
            defaultType="line"
            data={predictionAccuracyData}
            config={{
              xKey: "date",
              xInterval: 4,
              lines: [
                { key: "accuracy", label: "Accuracy (%)" },
                { key: "mape", label: "MAPE (%)" },
              ],
              colors: [INDIGO, ROSE],
              yFormatter: (v) => `${v}%`,
            }}
          />

          {/* Historical Prediction vs Actual */}
          <AnalyticsChart
            title="Predicted vs Actual Load"
            subtitle="30-day comparison (Apr 2026)"
            badge="Time-series"
            badgeColor="bg-cyan-500/15 text-cyan-300 border-cyan-500/20"
            allowedTypes={["line", "area", "bar", "composed", "stepped"]}
            defaultType="area"
            data={historicalPredictionData}
            config={{
              xKey: "date",
              xInterval: 4,
              lines: [
                { key: "actual", label: "Actual (MW)" },
                { key: "predicted", label: "Predicted (MW)" },
              ],
              colors: [EMERALD, VIOLET],
              yFormatter: (v) => `${(v / 1000).toFixed(0)}k`,
            }}
          />
        </div>
      </div>

      {/* ── Section 2: Feature Analysis ── */}
      <div>
        <SectionHeader icon={Cpu} title="Feature Analysis" subtitle="Factors driving the forecast model's predictions" />
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
          {/* Feature Importance */}
          <AnalyticsChart
            title="Feature Importance"
            subtitle="Relative contribution to forecast accuracy"
            badge="Category"
            badgeColor="bg-violet-500/15 text-violet-300 border-violet-500/20"
            allowedTypes={["bar", "pie"]}
            defaultType="bar"
            data={featureImportanceData}
            config={{
              xKey: "feature",
              lines: [{ key: "importance", label: "Importance (%)" }],
              colors: [VIOLET, CYAN, AMBER, EMERALD, ROSE, INDIGO, "#8b5cf6", "#06b6d4"],
              yFormatter: (v) => `${v}%`,
            }}
            height={280}
          />

          {/* Environmental Impact */}
          <AnalyticsChart
            title="Environmental Factors vs Load"
            subtitle="Temperature, humidity, and load across 24h"
            badge="Time-series"
            badgeColor="bg-amber-500/15 text-amber-300 border-amber-500/20"
            allowedTypes={["line", "area", "bar", "composed", "stepped"]}
            defaultType="area"
            data={environmentalImpactData}
            config={{
              xKey: "hour",
              xInterval: 3,
              lines: [
                { key: "load", label: "Load (MW)" },
                { key: "temperature", label: "Temp (°C)" },
                { key: "humidity", label: "Humidity (%)" },
              ],
              colors: [INDIGO, AMBER, CYAN],
              yFormatter: (v) => v.toLocaleString(),
            }}
            height={280}
          />
        </div>
      </div>

      {/* ── Section 3: Sensor & Resource ── */}
      <div>
        <SectionHeader icon={Radio} title="Sensor & Resource Analysis" subtitle="Zone contributions and infrastructure utilisation" />
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
          {/* Sensor Contribution */}
          <AnalyticsChart
            title="Zone Sensor Contribution"
            subtitle="Percentage share of total load by distribution zone"
            badge="Proportions"
            badgeColor="bg-emerald-500/15 text-emerald-300 border-emerald-500/20"
            allowedTypes={["pie", "bar"]}
            defaultType="pie"
            data={sensorContributionData}
            config={{
              xKey: "name",
              lines: [{ key: "value", label: "Share (%)" }],
              colors: [INDIGO, CYAN, VIOLET, AMBER, EMERALD],
              pieKey: true,
            }}
          />

          {/* Resource Utilisation */}
          <AnalyticsChart
            title="Resource Utilisation"
            subtitle="Weekly transformer, feeder, and bus loading (%)"
            badge="Category"
            badgeColor="bg-cyan-500/15 text-cyan-300 border-cyan-500/20"
            allowedTypes={["bar", "line", "area", "composed", "stepped"]}
            defaultType="bar"
            data={resourceUtilisationData}
            config={{
              xKey: "day",
              lines: [
                { key: "transformer", label: "Transformer %" },
                { key: "feeder", label: "Feeder %" },
                { key: "bus", label: "Bus Bar %" },
              ],
              colors: [INDIGO, CYAN, EMERALD],
              yFormatter: (v) => `${v}%`,
            }}
          />
        </div>
      </div>

      {/* ── Section 4: Model Health & Confidence ── */}
      <div>
        <SectionHeader icon={Activity} title="Model Health & Confidence" subtitle="Real-time model reliability and system health metrics" />
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
          {/* Model Confidence */}
          <AnalyticsChart
            title="Model Confidence Over Time"
            subtitle="Confidence interval narrows for near-term predictions"
            badge="Time-series"
            badgeColor="bg-indigo-500/15 text-indigo-300 border-indigo-500/20"
            allowedTypes={["line", "area"]}
            defaultType="area"
            data={modelConfidenceData}
            config={{
              xKey: "hour",
              xInterval: 3,
              lines: [
                { key: "confidence", label: "Confidence (%)" },
                { key: "upperBound", label: "Upper Bound" },
                { key: "lowerBound", label: "Lower Bound" },
              ],
              colors: [INDIGO, "#818cf8", "#4f46e5"],
              yFormatter: (v) => `${v}%`,
            }}
          />

          {/* System Health */}
          <AnalyticsChart
            title="System Health Score"
            subtitle="Composite operational health over 30 days"
            badge="Time-series"
            badgeColor="bg-emerald-500/15 text-emerald-300 border-emerald-500/20"
            allowedTypes={["line", "area", "bar", "composed", "stepped"]}
            defaultType="line"
            data={systemHealthData}
            config={{
              xKey: "date",
              xInterval: 4,
              lines: [
                { key: "score", label: "Health Score" },
                { key: "threshold", label: "Min Threshold" },
              ],
              colors: [EMERALD, ROSE],
              yFormatter: (v) => v,
            }}
          />
        </div>
      </div>

      {/* ── Section 5: Risk & Correlation ── */}
      <div>
        <SectionHeader icon={Shield} title="Risk & Correlation Analysis" subtitle="Multi-factor risk radar and environmental correlations" />
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
          {/* Risk Factor Radar */}
          <AnalyticsChart
            title="Risk Factor Analysis"
            subtitle="Current risk vs operational threshold by metric"
            badge="Multi-metric"
            badgeColor="bg-rose-500/15 text-rose-300 border-rose-500/20"
            allowedTypes={["radar", "bar"]}
            defaultType="radar"
            data={riskRadarData}
            config={{
              xKey: "subject",
              radarAngleKey: "subject",
              lines: [
                { key: "current", label: "Current Level" },
                { key: "threshold", label: "Threshold" },
              ],
              colors: [ROSE, AMBER],
            }}
            height={300}
          />

          {/* Correlation Scatter */}
          <AnalyticsChart
            title="Temperature × Humidity → Load"
            subtitle="How weather conditions correlate with grid demand"
            badge="Correlation"
            badgeColor="bg-amber-500/15 text-amber-300 border-amber-500/20"
            allowedTypes={["scatter"]}
            defaultType="scatter"
            data={correlationData}
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
            height={300}
          />
        </div>
      </div>
    </motion.div>
  );
}
