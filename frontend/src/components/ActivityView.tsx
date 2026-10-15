import { useEffect, useState } from 'react'
import { fetchLogs } from '../lib/api'
import type { LogEntry, LoggedChunk, Role } from '../types'

const ROLES: Role[] = ['contractor', 'senior_engineer']
const LABEL: Record<Role, string> = { contractor: 'Contractor', senior_engineer: 'Senior engineer' }

type State =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'done'; rows: LogEntry[] }

function sum(xs: (number | null)[]): number {
  return xs.reduce<number>((a, b) => a + (b ?? 0), 0)
}

function Summary({ rows }: { rows: LogEntry[] }) {
  const timed = rows.filter((r) => r.total_ms != null)
  if (timed.length === 0) return null
  const n = timed.length
  const total = sum(timed.map((r) => r.total_ms))
  const gen = sum(timed.map((r) => r.generation_ms))
  const ret = sum(timed.map((r) => r.retrieval_ms))
  return (
    <p className="summary">
      <strong>{rows.length}</strong> queries · avg total <strong>{Math.round(total / n)} ms</strong> · avg
      retrieval <strong>{Math.round(ret / n)} ms</strong> · avg generation{' '}
      <strong>{Math.round(gen / n)} ms</strong>
      {total > 0 && <> ({Math.round((gen / total) * 100)}% of the time is generation)</>}
    </p>
  )
}

function LatencyBar({ r }: { r: LogEntry }) {
  if (!r.total_ms) return <span className="muted small">n/a</span>
  const ret = ((r.retrieval_ms ?? 0) / r.total_ms) * 100
  const gen = ((r.generation_ms ?? 0) / r.total_ms) * 100
  return (
    <div className="latency" title={`retrieval ${r.retrieval_ms} ms, generation ${r.generation_ms} ms`}>
      <div className="bar">
        <span className="bar-retrieval" style={{ width: `${ret}%` }} />
        <span className="bar-generation" style={{ width: `${gen}%` }} />
      </div>
      <span className="muted small">{r.total_ms} ms</span>
    </div>
  )
}

function Files({ chunks }: { chunks: LoggedChunk[] }) {
  const files = new Map<string, LoggedChunk>()
  for (const c of chunks) files.set(`${c.repo}/${c.path}`, c)
  return (
    <details>
      <summary>{files.size} files</summary>
      <ul className="files">
        {[...files.entries()].map(([name, c]) => (
          <li key={name}>
            <span className="path">{name}</span>{' '}
            <span className={`badge badge-${c.scope}`}>
              {c.scope === 'senior_engineer' ? 'senior-only' : 'contractor'}
            </span>
          </li>
        ))}
      </ul>
    </details>
  )
}

export function ActivityView() {
  const [viewAs, setViewAs] = useState<Role>('contractor')
  const [refreshKey, setRefreshKey] = useState(0)
  const [state, setState] = useState<State>({ status: 'loading' })

  useEffect(() => {
    let cancelled = false // ignore a slow response if the role was switched meanwhile
    setState({ status: 'loading' })
    fetchLogs(viewAs)
      .then((rows) => {
        if (!cancelled) setState({ status: 'done', rows })
      })
      .catch((e) => {
        if (!cancelled) setState({ status: 'error', message: e instanceof Error ? e.message : String(e) })
      })
    return () => {
      cancelled = true
    }
  }, [viewAs, refreshKey])

  return (
    <section className="activity">
      <div className="activity-controls">
        <span className="muted">View as:</span>
        {ROLES.map((r) => (
          <button
            key={r}
            type="button"
            className={`pill pill-${r} ${viewAs === r ? 'active' : ''}`}
            onClick={() => setViewAs(r)}
          >
            {LABEL[r]}
          </button>
        ))}
        <button type="button" className="pill" onClick={() => setRefreshKey((k) => k + 1)}>
          Refresh
        </button>
      </div>

      {state.status === 'loading' && <p className="muted">Loading…</p>}
      {state.status === 'error' && <p className="error">{state.message}</p>}

      {state.status === 'done' && (
        <>
          <p>
            As <strong>{LABEL[viewAs]}</strong>, the database returned <strong>{state.rows.length}</strong>{' '}
            {state.rows.length === 1 ? 'query' : 'queries'}. Contractors can read only their own; senior engineers
            can read everyone's. This page has no role logic: it calls <code>/logs</code> with that role's token and
            shows whatever Postgres returns.
          </p>
          <Summary rows={state.rows} />

          {state.rows.length === 0 ? (
            <p className="muted">No queries yet. Ask something in the other tab.</p>
          ) : (
            <table className="log-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Who</th>
                  <th>Question</th>
                  <th>Senior-only chunks</th>
                  <th>Files</th>
                  <th>Latency (retrieval | generation)</th>
                </tr>
              </thead>
              <tbody>
                {state.rows.map((r) => (
                  <tr key={r.id}>
                    <td className="small">{new Date(r.created_at).toLocaleString()}</td>
                    <td>
                      <span className={`badge badge-${r.role}`}>{LABEL[r.role]}</span>
                      <div className="muted small">{r.user_id.slice(0, 8)}</div>
                    </td>
                    <td className="question">{r.question}</td>
                    <td className={r.senior_only_count > 0 ? 'hot' : ''}>
                      {r.senior_only_count} of {r.retrieved_count}
                    </td>
                    <td>
                      <Files chunks={r.retrieved} />
                    </td>
                    <td>
                      <LatencyBar r={r} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <p className="muted small">
            Chunks that row-level security blocked are not logged: the application never receives them, so it
            cannot record them.
          </p>
        </>
      )}
    </section>
  )
}
