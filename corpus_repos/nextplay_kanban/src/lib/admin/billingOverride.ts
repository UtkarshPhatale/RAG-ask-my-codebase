// DUMMY FILE — added deliberately for RBAC testing (RAG-ask-my-codebase).
// Not real business logic, not wired into the actual app or build.
// senior_engineer scope only: see docs/access_design.md.
//
// Fictional stub representing "admin billing override" functionality —
// the kind of module that plausibly exists in a real product's admin
// surface and plausibly should NOT be retrievable by a contractor asking
// generic "how does this app work" questions.

import { createClient } from '@supabase/supabase-js';

// Fictional: a service-role client used only by internal admin tooling,
// never shipped to the browser bundle.
function getServiceRoleClient() {
  const url = process.env.SUPABASE_URL as string;
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY as string; // see secrets_template
  return createClient(url, serviceKey);
}

/**
 * Fictional admin-only operation: force-extends a team's trial period,
 * bypassing normal billing checks entirely via the service-role client
 * (which ignores RLS). Restricted to internal admin tooling only.
 */
export async function forceExtendTrial(teamId: string, days: number) {
  const client = getServiceRoleClient();
  return client
    .from('teams')
    .update({ trial_extended_days: days })
    .eq('id', teamId);
}
