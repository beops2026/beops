import { NextResponse } from "next/server";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:5000";

const ENDPOINT_MAP = {
  hourly: "/predict",
  weekly: "/predict-weekly",
  monthly: "/predict-monthly",
  quarterly: "/predict-quarterly",
};

export async function GET(request) {
  const { searchParams } = new URL(request.url);

  const forecastType = searchParams.get("type") || "hourly";
  const startDate = searchParams.get("start_date");
  const endDate = searchParams.get("end_date");

  if (!startDate || !endDate) {
    return NextResponse.json(
      { error: "start_date and end_date required" },
      { status: 400 }
    );
  }

  const endpoint = ENDPOINT_MAP[forecastType];
  if (!endpoint) {
    return NextResponse.json(
      { error: `Unknown forecast type: ${forecastType}` },
      { status: 400 }
    );
  }

  const backendUrl = `${API_BASE}${endpoint}?start_date=${startDate}&end_date=${endDate}`;

  try {
    const response = await fetch(backendUrl, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
      // 5-minute timeout for long forecasts
      signal: AbortSignal.timeout(300000),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Forecast proxy error:", error);
    return NextResponse.json(
      {
        error:
          error.name === "TimeoutError"
            ? "Request timed out. The forecast computation is taking longer than expected."
            : `Backend connection failed: ${error.message}`,
      },
      { status: 502 }
    );
  }
}
