import type { AskResponse, Role } from '../types'
import { getAccessToken } from './supabase'

const TIMEOUT_MS = 60_000

/** Ask the backend as `role`. The role is decided server-side from the JWT, not sent by us. */
export async function ask(role: Role, question: string): Promise<AskResponse> {
  const token = await getAccessToken(role)

  let res: Response
  try {
    res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ question }),
      // Without this, a hung backend leaves the UI on "Asking…" forever.
      signal: AbortSignal.timeout(TIMEOUT_MS),
    })
  } catch (e) {
    if (e instanceof DOMException && e.name === 'TimeoutError') {
      throw new Error(`No response after ${TIMEOUT_MS / 1000} s. The backend may be down or stuck.`)
    }
    throw new Error(`Request failed: ${e instanceof Error ? e.message : String(e)}`)
  }

  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${detail.slice(0, 200)}`)
  }
  return (await res.json()) as AskResponse
}
