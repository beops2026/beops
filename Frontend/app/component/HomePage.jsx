"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import {
  Zap, BarChart3, Brain, Cloud, Calendar, Activity,
  TrendingUp, Shield, ArrowRight, CheckCircle2, Cpu,
  LineChart as LineChartIcon
} from "lucide-react";

// ─── Animated Background ─────────────────────────────────────────────────────
function AnimatedGrid() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {/* Grid lines */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage: `
            linear-gradient(rgba(99,102,241,0.8) 1px, transparent 1px),
            linear-gradient(90deg, rgba(99,102,241,0.8) 1px, transparent 1px)
          `,
          backgroundSize: "60px 60px",
        }}
      />
      {/* Radial glow center */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full bg-blue-600/10 blur-[120px]" />
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[400px] h-[400px] rounded-full bg-violet-600/10 blur-[80px]" />
      {/* Floating orbs */}
      {[...Array(6)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute rounded-full mix-blend-screen"
          style={{
            width: `${[120, 80, 100, 60, 140, 90][i]}px`,
            height: `${[120, 80, 100, 60, 140, 90][i]}px`,
            left: `${[10, 70, 30, 85, 50, 15][i]}%`,
            top: `${[20, 50, 70, 25, 80, 60][i]}%`,
            background: [
              "radial-gradient(circle, rgba(99,102,241,0.15), transparent)",
              "radial-gradient(circle, rgba(139,92,246,0.15), transparent)",
              "radial-gradient(circle, rgba(34,211,238,0.10), transparent)",
              "radial-gradient(circle, rgba(245,158,11,0.10), transparent)",
              "radial-gradient(circle, rgba(99,102,241,0.12), transparent)",
              "radial-gradient(circle, rgba(167,139,250,0.12), transparent)",
            ][i],
          }}
          animate={{
            y: [0, -30, 0],
            x: [0, 10, 0],
            scale: [1, 1.1, 1],
          }}
          transition={{
            duration: [8, 6, 10, 7, 9, 5][i],
            repeat: Infinity,
            ease: "easeInOut",
            delay: i * 0.8,
          }}
        />
      ))}
    </div>
  );
}

// ─── Stat Counter Card ────────────────────────────────────────────────────────
function StatCard({ value, label, color }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      whileInView={{ opacity: 1, scale: 1 }}
      viewport={{ once: true }}
      className="text-center"
    >
      <p className={`text-3xl lg:text-4xl font-bold ${color}`}>{value}</p>
      <p className="text-sm text-zinc-500 mt-1">{label}</p>
    </motion.div>
  );
}

// ─── Feature Card ─────────────────────────────────────────────────────────────
const features = [
  {
    icon: BarChart3,
    title: "Hourly Forecasting",
    desc: "24 hourly load predictions for a single day with MW-level accuracy.",
    color: "from-blue-500/10 to-blue-600/5",
    border: "border-blue-500/20",
    accent: "text-blue-400",
    bg: "bg-blue-500/10",
  },
  {
    icon: LineChartIcon,
    title: "Weekly Trends",
    desc: "Daily load totals across 7-day windows for operational planning.",
    color: "from-cyan-500/10 to-cyan-600/5",
    border: "border-cyan-500/20",
    accent: "text-cyan-400",
    bg: "bg-cyan-500/10",
  },
  {
    icon: TrendingUp,
    title: "Monthly Analysis",
    desc: "Day-by-day demand curves across an entire month for capacity planning.",
    color: "from-violet-500/10 to-violet-600/5",
    border: "border-violet-500/20",
    accent: "text-violet-400",
    bg: "bg-violet-500/10",
  },
  {
    icon: Calendar,
    title: "Quarterly Outlook",
    desc: "Monthly aggregated totals up to 108 days ahead using weather forecasts.",
    color: "from-amber-500/10 to-amber-600/5",
    border: "border-amber-500/20",
    accent: "text-amber-400",
    bg: "bg-amber-500/10",
  },
  {
    icon: Brain,
    title: "AI Operational Reports",
    desc: "Qwen LLM-powered reports with risks, action plans, and next steps.",
    color: "from-rose-500/10 to-rose-600/5",
    border: "border-rose-500/20",
    accent: "text-rose-400",
    bg: "bg-rose-500/10",
  },
  {
    icon: Cloud,
    title: "Weather Integration",
    desc: "Real-time Open-Meteo weather data fused into every forecast point.",
    color: "from-emerald-500/10 to-emerald-600/5",
    border: "border-emerald-500/20",
    accent: "text-emerald-400",
    bg: "bg-emerald-500/10",
  },
];

// Mini animated chart for hero visual
function MiniChart() {
  const bars = [65, 48, 72, 55, 80, 63, 91, 74, 68, 85, 59, 78];
  return (
    <div className="flex items-end gap-1 h-16">
      {bars.map((h, i) => (
        <motion.div
          key={i}
          className="flex-1 rounded-t-sm"
          style={{
            background: `linear-gradient(to top, rgba(99,102,241,0.8), rgba(139,92,246,0.4))`,
          }}
          initial={{ height: 0 }}
          animate={{ height: `${h}%` }}
          transition={{ duration: 0.6, delay: i * 0.07, ease: "easeOut" }}
        />
      ))}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────
export function HomePage() {
  return (
    <div className="min-h-screen bg-[#040408] text-white">
      {/* ── Hero Section ── */}
      <section className="relative min-h-screen flex flex-col items-center justify-center px-4 pt-20 pb-16 overflow-hidden">
        <AnimatedGrid />

        <div className="relative z-10 max-w-5xl mx-auto text-center">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center gap-2 rounded-full border border-blue-500/30 bg-blue-500/10 px-4 py-1.5 mb-6"
          >
            <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
            <span className="text-xs font-medium text-blue-300">
              AI-Powered Grid Substation Forecasting
            </span>
          </motion.div>

          {/* Headline */}
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-4xl sm:text-5xl lg:text-7xl font-bold tracking-tight mb-6"
          >
            <span className="text-white">Predict Grid Load.</span>
            <br />
            <span className="bg-gradient-to-r from-blue-400 via-violet-400 to-cyan-400 bg-clip-text text-transparent">
              Operate Smarter.
            </span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-lg text-zinc-400 max-w-2xl mx-auto mb-10 leading-relaxed"
          >
            LoadSense provides accurate electricity load forecasting for grid substation operators —
            from hourly to quarterly horizons — powered by machine learning and real-time weather data.
          </motion.p>

          {/* CTAs */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="flex flex-col sm:flex-row gap-4 justify-center"
          >
            <Link
              href="/dashboard"
              className="inline-flex items-center gap-2 px-6 py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 text-white font-semibold text-sm transition-all shadow-lg shadow-blue-900/40 hover:shadow-blue-900/60 hover:scale-105"
            >
              <Zap className="w-4 h-4" />
              Generate Forecast
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              href="/dashboard/analytics"
              className="inline-flex items-center gap-2 px-6 py-3.5 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 text-white font-semibold text-sm transition-all"
            >
              <BarChart3 className="w-4 h-4" />
              View Analytics
            </Link>
          </motion.div>

          {/* Mini dashboard preview */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="mt-16 mx-auto max-w-2xl rounded-2xl border border-white/10 bg-white/[0.03] backdrop-blur-sm p-6 shadow-2xl"
          >
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Today's Forecast</p>
                <p className="text-sm text-zinc-500">Hourly Load — Delhi Grid Station</p>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-xs text-emerald-400 font-medium">Live</span>
              </div>
            </div>
            <MiniChart />
            <div className="grid grid-cols-3 gap-4 mt-4 pt-4 border-t border-white/5">
              {[
                { label: "Peak Load", value: "4,279 MW", color: "text-rose-400" },
                { label: "Average", value: "3,360 MW", color: "text-blue-400" },
                { label: "Trend", value: "Rising ↑", color: "text-amber-400" },
              ].map((s) => (
                <div key={s.label} className="text-center">
                  <p className={`text-sm font-bold ${s.color}`}>{s.value}</p>
                  <p className="text-[10px] text-zinc-600">{s.label}</p>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* ── Stats ── */}
      <section className="relative py-16 border-y border-white/5 bg-white/[0.01]">
        <div className="max-w-5xl mx-auto px-4">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
            <StatCard value="±3.2%" label="Forecast Error (MAPE)" color="text-blue-400" />
            <StatCard value="108 days" label="Max Forecast Horizon" color="text-violet-400" />
            <StatCard value="24×7" label="Automated Monitoring" color="text-cyan-400" />
            <StatCard value="< 2 min" label="AI Report Generation" color="text-amber-400" />
          </div>
        </div>
      </section>

      {/* ── Features Grid ── */}
      <section className="py-24 px-4">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
              Everything You Need to{" "}
              <span className="bg-gradient-to-r from-blue-400 to-violet-400 bg-clip-text text-transparent">
                Operate the Grid
              </span>
            </h2>
            <p className="text-zinc-400 max-w-2xl mx-auto">
              From short-term hourly predictions to quarterly outlooks, LoadSense gives substation
              operators the intelligence they need to manage load effectively.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {features.map((f, i) => (
              <motion.div
                key={f.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.08 }}
                whileHover={{ y: -4 }}
                className={`relative rounded-2xl border ${f.border} bg-gradient-to-br ${f.color} p-6 group cursor-default`}
              >
                <div className={`w-10 h-10 rounded-xl ${f.bg} flex items-center justify-center mb-4`}>
                  <f.icon className={`w-5 h-5 ${f.accent}`} />
                </div>
                <h3 className="font-semibold text-white mb-2">{f.title}</h3>
                <p className="text-sm text-zinc-400 leading-relaxed">{f.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How it Works ── */}
      <section className="py-24 px-4 bg-white/[0.01] border-y border-white/5">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl font-bold text-white mb-4">How It Works</h2>
            <p className="text-zinc-400">Three steps from date selection to actionable insights</p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative">
            {/* Connector line */}
            <div className="hidden md:block absolute top-8 left-[calc(33%-20px)] right-[calc(33%-20px)] h-px bg-gradient-to-r from-transparent via-blue-500/30 to-transparent" />

            {[
              {
                step: "01",
                title: "Select & Configure",
                desc: "Choose forecast type (Hourly/Weekly/Monthly/Quarterly), pick your date range, and enter the substation name.",
                icon: Calendar,
                color: "text-blue-400",
                bg: "bg-blue-500/10",
              },
              {
                step: "02",
                title: "ML Model Runs",
                desc: "The backend fetches real-time weather data from Open-Meteo and runs the trained LSTM model to generate predictions.",
                icon: Cpu,
                color: "text-violet-400",
                bg: "bg-violet-500/10",
              },
              {
                step: "03",
                title: "AI Report Generated",
                desc: "Qwen LLM analyses the forecast and generates an operational report with risks, action plans, and next steps.",
                icon: Brain,
                color: "text-emerald-400",
                bg: "bg-emerald-500/10",
              },
            ].map((s, i) => (
              <motion.div
                key={s.step}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.15 }}
                className="relative text-center"
              >
                <div className={`w-16 h-16 mx-auto rounded-2xl ${s.bg} flex items-center justify-center mb-4 border border-white/5`}>
                  <s.icon className={`w-7 h-7 ${s.color}`} />
                </div>
                <span className="text-xs font-bold text-zinc-600 uppercase tracking-widest block mb-2">{s.step}</span>
                <h3 className="font-semibold text-white mb-2">{s.title}</h3>
                <p className="text-sm text-zinc-400 leading-relaxed">{s.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA Banner ── */}
      <section className="py-24 px-4">
        <div className="max-w-3xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="rounded-3xl bg-gradient-to-br from-blue-950/60 to-violet-950/60 border border-blue-700/20 p-12"
          >
            <div className="w-14 h-14 mx-auto rounded-2xl bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center mb-6">
              <Zap className="w-7 h-7 text-white" />
            </div>
            <h2 className="text-3xl font-bold text-white mb-4">
              Ready to Optimise Your Grid Operations?
            </h2>
            <p className="text-zinc-400 mb-8 max-w-lg mx-auto">
              Start generating accurate load forecasts and AI-powered operational reports today.
            </p>
            <div className="flex flex-col sm:flex-row gap-3 justify-center">
              <Link
                href="/dashboard"
                className="inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 text-white font-semibold text-sm transition-all shadow-lg shadow-blue-900/40"
              >
                <Zap className="w-4 h-4" />
                Generate Forecast
              </Link>
              <Link
                href="/dashboard/ai-insights"
                className="inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl border border-white/10 bg-white/5 hover:bg-white/10 text-white font-semibold text-sm transition-all"
              >
                <Brain className="w-4 h-4" />
                AI Reports
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="border-t border-white/5 py-8 px-4">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-md bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center">
              <Zap className="w-3 h-3 text-white" />
            </div>
            <p className="text-sm font-semibold text-white">LoadSense</p>
            <span className="text-zinc-600 text-sm">Grid Substation Forecasting System</span>
          </div>
          <p className="text-xs text-zinc-600">
            Powered by Open-Meteo · HuggingFace · Next.js
          </p>
        </div>
      </footer>
    </div>
  );
}
