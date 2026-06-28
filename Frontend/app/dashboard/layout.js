"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sidebar } from "../components/dashboard/sidebar";
import { ToastProvider } from "@/components/ui/toast";

export default function DashboardLayout({ children }) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 1024;
      setIsMobile(mobile);
      if (!mobile) setIsSidebarOpen(true);
    };

    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const toggleSidebar = () => setIsSidebarOpen((prev) => !prev);

  return (
    <ToastProvider>
      <div className="relative min-h-screen bg-[#0A0A0F]">
        {/* Mobile Overlay */}
        <AnimatePresence>
          {isMobile && isSidebarOpen && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="fixed inset-0 bg-black/60 z-40 backdrop-blur-sm"
              onClick={toggleSidebar}
            />
          )}
        </AnimatePresence>

        {/* Sidebar Container */}
        <div className="fixed top-0 left-0 bottom-0 z-50 w-72">
          <motion.div
            className="absolute top-0 left-0 bottom-0 w-full overflow-hidden"
            animate={{
              width: isSidebarOpen ? "100%" : isMobile ? "0%" : "72px",
              x: isMobile && !isSidebarOpen ? "-100%" : "0%",
            }}
            transition={{
              type: "spring",
              damping: 25,
              stiffness: 120,
              mass: 0.8,
            }}
          >
            <Sidebar isSidebarOpen={isSidebarOpen} toggleSidebar={toggleSidebar} />
          </motion.div>
        </div>

        {/* Main Content */}
        <motion.div
          className="min-h-screen"
          animate={{
            marginLeft: isMobile ? 0 : isSidebarOpen ? "288px" : "72px",
          }}
          transition={{
            type: "spring",
            damping: 25,
            stiffness: 120,
            mass: 0.8,
          }}
        >
          {/* Top bar for mobile */}
          {isMobile && (
            <div className="sticky top-0 z-30 bg-[#0A0A0F]/90 backdrop-blur-md border-b border-white/5 px-4 py-3 flex items-center gap-3">
              <button
                onClick={toggleSidebar}
                className="p-2 rounded-lg hover:bg-white/10 transition-colors"
              >
                <svg className="w-5 h-5 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
              <p className="font-semibold text-white text-sm">LoadSense</p>
            </div>
          )}

          <main className="py-8 px-6 lg:px-8 max-w-[1600px]">
            {children}
          </main>
        </motion.div>
      </div>
    </ToastProvider>
  );
}