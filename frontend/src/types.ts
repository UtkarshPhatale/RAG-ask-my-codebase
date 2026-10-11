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
