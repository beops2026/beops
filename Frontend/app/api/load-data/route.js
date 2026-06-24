import { NextResponse } from "next/server";

// This endpoint was for the old Delhi SLDC scraper backend.
// It is no longer used. Use /api/forecast instead.
export async function GET() {
  return NextResponse.json(
    { error: "This endpoint is deprecated. Use /api/forecast instead." },
    { status: 410 }
  );
}