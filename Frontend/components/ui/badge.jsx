import { cn } from "@/lib/utils";
import {
  TrendingUp,
  TrendingDown,
  Minus,
} from "lucide-react";

export function Badge({ children, variant = "default", className, ...props }) {
  const variants = {
    default: "bg-white/10 text-white border-white/20",
    primary: "bg-blue-500/20 text-blue-300 border-blue-500/30",
    success: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
    warning: "bg-amber-500/20 text-amber-300 border-amber-500/30",
    danger: "bg-red-500/20 text-red-300 border-red-500/30",
    violet: "bg-violet-500/20 text-violet-300 border-violet-500/30",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium",
        variants[variant] || variants.default,
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
}

export function TrendBadge({ trend }) {
  if (trend === "rising") {
    return (
      <Badge variant="danger" className="gap-1">
        <TrendingUp className="h-3 w-3" />
        Rising
      </Badge>
    );
  }
  if (trend === "falling") {
    return (
      <Badge variant="primary" className="gap-1">
        <TrendingDown className="h-3 w-3" />
        Falling
      </Badge>
    );
  }
  return (
    <Badge variant="default" className="gap-1">
      <Minus className="h-3 w-3" />
      Stable
    </Badge>
  );
}
