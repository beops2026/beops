import { redirect } from "next/navigation";

// Globe visualization page — temporarily disabled pending three.js dependency resolution.
export default function GlobePage() {
  redirect("/dashboard/analytics");
}
