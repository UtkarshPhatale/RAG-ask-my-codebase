import { createClient, type SupabaseClient } from '@supabase/supabase-js'
import type { Role } from '../types'

const url = import.meta.env.VITE_SUPABASE_URL as string
// The PUBLISHABLE key is designed to be public. The secret key must never be
// referenced anywhere in this folder: it bypasses RLS.
const publishableKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY as string

if (!url || !publishableKey) {
  throw new Error('Missing VITE_SUPABASE_URL / VITE_SUPABASE_PUBLISHABLE_KEY (copy .env.example to .env.local)')
}

// Demo accounts, one per role. This is a demo harness: a real product would have a
// login form and a single role per user.
const DEMO: Record<Role, { email: string; password: string }> = {
  contractor: {
    email: import.meta.env.VITE_DEMO_CONTRACTOR_EMAIL as string,
    password: import.meta.env.VITE_DEMO_CONTRACTOR_PASSWORD as string,
  },
  senior_engineer: {
    email: import.meta.env.VITE_DEMO_SENIOR_EMAIL as string,
    password: import.meta.env.VITE_DEMO_SENIOR_PASSWORD as string,
  },
}

// One client per role, sessions kept in memory only, with distinct storage keys so
// the two logins can never overwrite each other.
function makeClient(role: Role): SupabaseClient {
  return createClient(url, publishableKey, {
    auth: {
      persistSession: false,
      autoRefreshToken: true,
      detectSessionInUrl: false,
      storageKey: `rag-demo-${role}`,
    },
  })
}

const clients: Record<Role, SupabaseClient> = {
  contractor: makeClient('contractor'),
  senior_engineer: makeClient('senior_engineer'),
}

/** JWT for this role's demo user; signs in on first use, refreshes when expired. */
export async function getAccessToken(role: Role): Promise<string> {
  const client = clients[role]
  const { data: { session } } = await client.auth.getSession()
  if (session) return session.access_token

  const { data, error } = await client.auth.signInWithPassword(DEMO[role])
  if (error || !data.session) {
    throw new Error(`Demo sign-in failed for ${role}: ${error?.message ?? 'no session'}`)
  }
  return data.session.access_token
}
