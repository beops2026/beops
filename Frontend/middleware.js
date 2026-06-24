import { NextResponse } from "next/server";

export function middleware(request) {
  // No authentication required — pass all requests through
  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!static|.*\\..*|_next|favicon.ico).*)"],
};
