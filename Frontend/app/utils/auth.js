// Auth utilities - no authentication required with the new backend.
// This file is kept for compatibility but auth is not enforced.

export function requireAuth() {
  return null; // No-op — always passes
}