"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  FileBarChart2,
  BarChart3,
  Home,
  Menu,
  ArrowLeftFromLine,
  ArrowRightFromLine,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { UserButton, useUser } from "@clerk/nextjs";

const routes = [
  {
    label: "Forecast Dashboard",
    icon: LayoutDashboard,
    href: "/dashboard",
    color: "text-sky-400",
    description: "Generate load forecasts",
  },
  {
    label: "AI Operational Report",
    icon: FileBarChart2,
    href: "/dashboard/ai-insights",
    color: "text-violet-400",
    description: "AI-powered analysis",
  },
  {
    label: "Analytics",
    icon: BarChart3,
    href: "/dashboard/analytics",
    color: "text-cyan-400",
    description: "Charts & model metrics",
  },
];

const externalLinks = [
  {
    label: "Home",
    icon: Home,
    href: "/",
    color: "text-zinc-400",
    description: "Back to homepage",
  },
];

export function Sidebar({ isSidebarOpen, toggleSidebar }) {
  const pathname = usePathname();
  const { user } = useUser();
  return (
    <div className="flex flex-col h-full bg-[#111116] text-white border-r border-white/5">
      {/* Header with Logo and Toggle */}
      <div className="p-5 border-b border-white/5 flex items-center justify-between">
        <Link href="/dashboard" className="flex items-center">
          <motion.div
            animate={{ width: isSidebarOpen ? "auto" : "40px" }}
            className="overflow-hidden flex items-center gap-3"
          >
            {/* Logo Icon */}
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center shrink-0">
              <Zap className="w-5 h-5 text-white" />
            </div>

            {isSidebarOpen && (
              <motion.div
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                className="min-w-0"
              >
                <p className="font-semibold text-sm text-white leading-tight whitespace-nowrap">
                  LoadSense
                </p>
                <p className="text-[10px] text-zinc-500 whitespace-nowrap">
                  Grid Forecasting System
                </p>
              </motion.div>
            )}
          </motion.div>
        </Link>

        {/* Desktop Toggle Button */}
        <motion.div animate={{ opacity: 1 }} className="hidden lg:block shrink-0">
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleSidebar}
            className="p-2 hover:bg-white/8 transition-colors rounded-lg"
          >
            {isSidebarOpen ? (
              <ArrowLeftFromLine className="h-4 w-4 text-zinc-400" />
            ) : (
              <ArrowRightFromLine className="h-4 w-4 text-zinc-400" />
            )}
          </Button>
        </motion.div>

        {/* Mobile Menu Button */}
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleSidebar}
          className="lg:hidden p-2 hover:bg-white/8 transition-colors"
        >
          <Menu className="h-4 w-4 text-zinc-400" />
        </Button>
      </div>

      {/* Navigation Links */}
      <div className="flex-1 px-3 py-5 overflow-y-auto">
        {isSidebarOpen && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-[10px] font-semibold uppercase tracking-widest text-zinc-600 px-3 mb-3"
          >
            Navigation
          </motion.p>
        )}
        <div className="space-y-1">
          {routes.map((route) => {
            const isActive = pathname === route.href;
            return (
              <Link key={route.href} href={route.href}>
                <motion.div
                  className={cn(
                    "flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-all duration-200 cursor-pointer",
                    isActive
                      ? "bg-white/10 text-white"
                      : "text-zinc-400 hover:bg-white/5 hover:text-zinc-200"
                  )}
                  whileHover={{ x: isSidebarOpen ? 3 : 0 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {/* Active indicator */}
                  {isActive && (
                    <div className={cn("absolute left-0 w-0.5 h-8 rounded-r-full", route.color.replace("text-", "bg-"))} />
                  )}
                  <route.icon
                    className={cn("h-5 w-5 shrink-0", isActive ? route.color : "text-zinc-500")}
                  />
                  <AnimatePresence mode="wait">
                    {isSidebarOpen && (
                      <motion.div
                        initial={{ opacity: 0, width: 0 }}
                        animate={{ opacity: 1, width: "auto" }}
                        exit={{ opacity: 0, width: 0 }}
                        transition={{ type: "spring", damping: 25, stiffness: 120 }}
                        className="overflow-hidden"
                      >
                        <p className="whitespace-nowrap font-medium text-sm">
                          {route.label}
                        </p>
                        {isActive && (
                          <p className="whitespace-nowrap text-[10px] text-zinc-500 mt-0.5">
                            {route.description}
                          </p>
                        )}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              </Link>
            );
          })}
        </div>

        {/* External Links */}
        {isSidebarOpen && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-[10px] font-semibold uppercase tracking-widest text-zinc-600 px-3 mb-3 mt-6"
          >
            General
          </motion.p>
        )}
        <div className="space-y-1">
          {externalLinks.map((route) => (
            <Link key={route.href} href={route.href}>
              <motion.div
                className={cn(
                  "flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-all duration-200 cursor-pointer",
                  "text-zinc-500 hover:bg-white/5 hover:text-zinc-300"
                )}
                whileHover={{ x: isSidebarOpen ? 3 : 0 }}
                whileTap={{ scale: 0.98 }}
              >
                <route.icon className="h-5 w-5 shrink-0 text-zinc-600" />
                <AnimatePresence mode="wait">
                  {isSidebarOpen && (
                    <motion.div
                      initial={{ opacity: 0, width: 0 }}
                      animate={{ opacity: 1, width: "auto" }}
                      exit={{ opacity: 0, width: 0 }}
                      transition={{ type: "spring", damping: 25, stiffness: 120 }}
                      className="overflow-hidden"
                    >
                      <p className="whitespace-nowrap font-medium text-sm">{route.label}</p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            </Link>
          ))}
        </div>
      </div>

      {/* Status Section */}
      <AnimatePresence>
        {isSidebarOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ type: "spring", damping: 25, stiffness: 120 }}
            className="p-4 border-t border-white/5"
          >
            <div className="rounded-xl bg-gradient-to-br from-blue-950/40 to-violet-950/40 border border-blue-900/30 p-3">
              <div className="flex items-center gap-2 mb-2">
                <div className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <p className="text-xs font-medium text-zinc-300">Backend Connected</p>
              </div>
              <p className="text-[10px] text-zinc-600">localhost:5000</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      {/* User Section */}
<AnimatePresence>
  {isSidebarOpen && (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: "auto" }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ type: "spring", damping: 25, stiffness: 120 }}
      className="p-4 border-t border-white/5"
    >
      <div className="flex items-center justify-between rounded-xl bg-white/5 border border-white/10 p-3">
        <div className="min-w-0">
          <p className="text-sm font-medium text-white truncate">
            {user?.fullName}
          </p>
          <p className="text-xs text-zinc-500 truncate">
            {user?.primaryEmailAddress?.emailAddress}
          </p>
        </div>

        <UserButton
          afterSignOutUrl="/sign-in"
          appearance={{
            elements: {
              avatarBox: "w-10 h-10",
            },
          }}
        />
      </div>
    </motion.div>
  )}
</AnimatePresence>
    </div>
  );
}