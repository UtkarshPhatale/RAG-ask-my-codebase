export type Role = 'contractor' | 'senior_engineer'

export interface Citation {
  chunk_id: string
  repo: string
  path: string
  required_scope: Role
  similarity: number
  content: string
}

export interface AskResponse {
  question: string
  role: Role
  answer: string | null
  sources: string[]
  citations: Citation[]
}

export interface LoggedChunk {
  chunk_id: string
  repo: string
  path: string
  scope: Role
  similarity: number
}

export interface LogEntry {
  id: string
  created_at: string
  user_id: string
  role: Role
  question: string
  k: number
  retrieved: LoggedChunk[]
  retrieved_count: number
  senior_only_count: number
  retrieval_ms: number | null
  generation_ms: number | null
  total_ms: number | null
}
