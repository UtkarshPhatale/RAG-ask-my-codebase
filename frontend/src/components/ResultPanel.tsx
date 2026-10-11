import type { AskResponse, Role } from '../types'

export type PanelState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'done'; data: AskResponse; ms: number }

const TITLES: Record<Role, string> = {
  contractor: 'Contractor',
  senior_engineer: 'Senior engineer',
}

export function ResultPanel({ role, state }: { role: Role; state: PanelState }) {
  return (
    <section className={`panel panel-${role}`} aria-label={`${TITLES[role]} result`}>
      <h2>{TITLES[role]}</h2>

      {state.status === 'idle' && <p className="muted">Ask a question to see this role's answer.</p>}
      {state.status === 'loading' && <p className="muted">Retrieving and generating…</p>}
      {state.status === 'error' && <p className="error">{state.message}</p>}

      {state.status === 'done' && <Result data={state.data} ms={state.ms} />}
    </section>
  )
}

function Result({ data, ms }: { data: AskResponse; ms: number }) {
  const seniorOnly = data.citations.filter((c) => c.required_scope === 'senior_engineer').length
  return (
    <>
      <p className="answer">{data.answer ?? '(no answer returned)'}</p>

      <h3>
        Retrieved context <span className="muted">({data.citations.length} chunks, {ms} ms)</span>
      </h3>
      <p className="muted small">
        Postgres filtered these before the app saw them. Senior-only chunks: <strong>{seniorOnly}</strong>
      </p>

      <ol className="citations">
        {data.citations.map((c) => (
          <li key={c.chunk_id}>
            <details>
              <summary>
                <span className="path">{c.repo}/{c.path}</span>
                <span className={`badge badge-${c.required_scope}`}>
                  {c.required_scope === 'senior_engineer' ? 'senior-only' : 'contractor'}
                </span>
                <span className="muted small">sim {c.similarity.toFixed(3)}</span>
              </summary>
              <pre>{c.content}</pre>
            </details>
          </li>
        ))}
      </ol>
    </>
  )
}
