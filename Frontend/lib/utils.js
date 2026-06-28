import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merge Tailwind CSS class names, resolving conflicts intelligently.
 * Used throughout the component library.
 */
export function cn(...inputs) {
  return twMerge(clsx(inputs));
}
