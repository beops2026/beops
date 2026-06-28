import { redirect } from "next/navigation";

// Old chart page — replaced by the new Forecast Dashboard.
export default function OldChartPage() {
  redirect("/dashboard");
}