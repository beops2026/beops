import { redirect } from "next/navigation";

// This page was for the old backend (market price analysis).
// Redirecting to the new dashboard.
export default function ResolvePage() {
  redirect("/dashboard");
}