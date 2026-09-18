# DUMMY FILE — added deliberately for RBAC testing in the
# RAG-ask-my-codebase project. Not real credentials, not used by the app.
# senior_engineer scope only: see docs/access_design.md.
#
# In a real deployment, a file like this documents which service-role
# secrets exist and where they're consumed, so on-call engineers know
# what to rotate. That's exactly the kind of thing a contractor account
# should never be able to retrieve verbatim.

SUPABASE_SERVICE_ROLE_KEY = "sb_service_FAKE_1a2b3c4d5e6f7g8h9i0j"
STRIPE_SECRET_KEY = "sk_live_FAKE_51NxYzABCDEF1234567890"
INTERNAL_ADMIN_OVERRIDE_TOKEN = "FAKE_override_9f8e7d6c5b4a"

# Rotation policy: service-role key rotated quarterly by infra lead.
# Anyone with this key bypasses RLS entirely — never expose client-side.
