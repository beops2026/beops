"use client";

import { motion } from "framer-motion";
import { 
  Zap, 
  LineChart, 
  Globe2, 
  Brain, 
  BarChart3, 
  AlertCircle 
} from "lucide-react";

const services = [
  {
    icon: LineChart,
    title: "Load Forecasting",
    description: "Advanced ML-powered forecasting for accurate prediction of electricity demand patterns.",
    color: "from-blue-500/10 to-blue-500/30"
  },
  {
    icon: Brain,
    title: "AI-Powered Analysis",
    description: "Intelligent analysis of load patterns and anomaly detection using state-of-the-art AI.",
    color: "from-purple-500/10 to-purple-500/30"
  },
  {
    icon: Globe2,
    title: "Geographic Insights",
    description: "Interactive 3D visualization of load distribution across different regions of Delhi.",
    color: "from-green-500/10 to-green-500/30"
  },
  {
    icon: BarChart3,
    title: "Real-time Monitoring",
    description: "Live monitoring of electricity consumption with instant updates and alerts.",
    color: "from-orange-500/10 to-orange-500/30"
  },
  {
    icon: AlertCircle,
    title: "Early Warning System",
    description: "Proactive alerts for potential peak loads and system anomalies.",
    color: "from-red-500/10 to-red-500/30"
  },
  {
    icon: Zap,
    title: "Load Management",
    description: "Smart recommendations for optimal load distribution and energy efficiency.",
    color: "from-yellow-500/10 to-yellow-500/30"
  }
];

export function Services() {
  return (
    <section className="relative py-20 bg-black">
      {/* Background Gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-black via-zinc-900/50 to-black" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6">
        {/* Section Header */}
        <div className="text-center mb-20">
          <motion.h2 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-4xl font-bold bg-gradient-to-r from-blue-500 to-violet-500 bg-clip-text text-transparent mb-4"
          >
            Our Services
          </motion.h2>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-zinc-400 max-w-2xl mx-auto"
          >
            Empowering the future of energy management with cutting-edge technology and intelligent solutions
          </motion.p>
        </div>

        {/* Services Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {services.map((service, index) => (
            <motion.div
              key={service.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              whileHover={{ y: -5 }}
              className="relative group"
            >
              <div className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${service.color} opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />
              <div className="relative p-8 rounded-2xl bg-zinc-900 border border-zinc-800 hover:border-zinc-700 transition-colors duration-300">
                <div className="w-12 h-12 rounded-lg bg-zinc-800 flex items-center justify-center mb-6">
                  <service.icon className="w-6 h-6 text-blue-400" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-3">
                  {service.title}
                </h3>
                <p className="text-zinc-400">
                  {service.description}
                </p>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Call to Action */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="text-center mt-20"
        >
          <p className="text-zinc-400 mb-6">
            Want to learn more about our services?
          </p>
          <a 
            href="/dashboard" 
            className="inline-flex items-center justify-center px-6 py-3 rounded-lg bg-blue-500 hover:bg-blue-600 text-white font-medium transition-colors duration-300"
          >
            Explore Dashboard
          </a>
        </motion.div>
      </div>
    </section>
  );
} 