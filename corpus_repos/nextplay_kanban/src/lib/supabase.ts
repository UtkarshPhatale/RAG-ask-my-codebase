import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables. Copy .env.example to .env and fill in your keys.');
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

/** Sign in anonymously if no session exists, return user id */
export async function ensureGuestSession(): Promise<string> {
  const { data: { session } } = await supabase.auth.getSession();
  if (session?.user) return session.user.id;

  const { data, error } = await supabase.auth.signInAnonymously();
  if (error || !data.user) throw new Error('Failed to create guest session');
  return data.user.id;
}
