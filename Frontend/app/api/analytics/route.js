import { NextResponse } from "next/server";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:5000";

// Maps the ?metric= query param to the backend analytics endpoint.
const ENDPOINT_MAP = {
  environmental: "/analytics/environmental",
  correlation: "/analytics/correlation",
  "feature-importance": "/analytics/feature-importance",
};

export async function GET(request) {
  const { searchParams } = new URL(request.url);

  const metric = searchParams.get("metric");
  const endpoint = ENDPOINT_MAP[metric];
  if (!endpoint) {
    return NextResponse.json(
      { error: `Unknown analytics metric: ${metric}` },
      { status: 400 }
    );
  }

  // Forward every other param (date, start_date, end_date, refresh) as-is.
  const forwarded = new URLSearchParams(searchParams);
  forwarded.delete("metric");
  const qs = forwarded.toString();
  const backendUrl = `${API_BASE}${endpoint}${qs ? `?${qs}` : ""}`;

  try {
    const response = await fetch(backendUrl, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
      // Feature-importance / multi-day correlation can be slow on first call.
      signal: AbortSignal.timeout(300000),
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error.name === "TimeoutError"
            ? "Analytics request timed out."
            : `Backend connection failed: ${error.message}`,
      },
      { status: 502 }
    );
  }
}
