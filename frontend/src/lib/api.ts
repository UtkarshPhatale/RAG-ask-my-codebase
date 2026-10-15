import type { AskResponse, LogEntry, Role } from '../types'
import { getAccessToken } from './supabase'

const TIMEOUT_MS = 60_000
const RETRY_DELAY_MS = 2_000

async function send(token: string, path: string, init: RequestInit): Promise<Response> {
  const headers = new Headers(init.headers)
  headers.set('Authorization', `Bearer ${token}`)
  try {
    return await fetch(`/api${path}`, {
      ...init,
      headers,
      // Without this, a hung backend leaves the UI waiting forever.
      signal: AbortSignal.timeout(TIMEOUT_MS),
    })
  } catch (e) {
    if (e instanceof DOMException && e.name === 'TimeoutError') {
      throw new Error(`No response after ${TIMEOUT_MS / 1000} s. The backend may be down or stuck.`)
    }
    throw new Error(`Request failed: ${e instanceof Error ? e.message : String(e)}`)
  }
}

/** Call the backend as `role`. The role is decided server-side from the JWT, not sent by us. */
async function call<T>(role: Role, path: string, init: RequestInit = {}): Promise<T> {
  const token = await getAccessToken(role)

  let res = await send(token, path, init)
  if (res.status === 503) {
    // The backend returns 503 only when its role lookup fails, before any retrieval or
    // generation happens, so retrying cannot repeat a model call. The usual cause is a
    // freshly issued token rejected by PostgREST ("JWT issued at future": a small clock
    // difference between Supabase services); a short wait clears it.
    await new Promise((resolve) => setTimeout(resolve, RETRY_DELAY_MS))
    res = await send(token, path, init)
  }

  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${detail.slice(0, 200)}`)
  }
  return (await res.json()) as T
}

export function ask(role: Role, question: string): Promise<AskResponse> {
  return call<AskResponse>(role, '/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
}

/** Recent queries as `role` sees them. Postgres RLS decides which rows come back. */
export function fetchLogs(role: Role, limit = 50): Promise<LogEntry[]> {
  return call<LogEntry[]>(role, `/logs?limit=${limit}`)
}
