import { NextResponse } from "next/server";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:5000";

export async function POST(request) {
  try {
    const body = await request.json();
    const { report_type, start_date, end_date, substation_name } = body;

    if (!start_date || !end_date) {
      return NextResponse.json(
        { error: "start_date and end_date required" },
        { status: 400 }
      );
    }

    if (!["hourly", "weekly", "monthly", "quarterly"].includes(report_type)) {
      return NextResponse.json(
        { error: "report_type must be hourly, weekly, monthly, or quarterly" },
        { status: 400 }
      );
    }

    const backendUrl = `${API_BASE}/report`;

    const response = await fetch(backendUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        report_type,
        start_date,
        end_date,
        substation_name: substation_name || "Grid Substation",
      }),
      // 3-minute timeout for LLM generation
      signal: AbortSignal.timeout(180000),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Report proxy error:", error);
    return NextResponse.json(
      {
        error:
          error.name === "TimeoutError"
            ? "AI report generation timed out. The model is taking longer than expected. Please try again."
            : `Backend connection failed: ${error.message}`,
      },
      { status: 502 }
    );
  }
}
