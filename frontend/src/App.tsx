import { useState, type FormEvent } from 'react'
import { ask } from './lib/api'
import { ResultPanel, type PanelState } from './components/ResultPanel'
import type { Role } from './types'
import './App.css'

const ROLES: Role[] = ['contractor', 'senior_engineer']

const EXAMPLES = [
  'How does the agent decide which domains and routes it may visit, and when does it require confirmation for irreversible actions?',
  'What is the PTO accrual policy and how much is the on-call stipend?',
  'How does the replay engine decide whether a run was a success, a business outcome, or a hard failure?',
]

const IDLE: Record<Role, PanelState> = {
  contractor: { status: 'idle' },
  senior_engineer: { status: 'idle' },
}

export default function App() {
  const [question, setQuestion] = useState('')
  const [panels, setPanels] = useState<Record<Role, PanelState>>(IDLE)
  const busy = ROLES.some((r) => panels[r].status === 'loading')

  async function runRole(role: Role, q: string) {
    const started = performance.now()
    try {
      const data = await ask(role, q)
      const ms = Math.round(performance.now() - started)
      setPanels((p) => ({ ...p, [role]: { status: 'done', data, ms } }))
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e)
      setPanels((p) => ({ ...p, [role]: { status: 'error', message } }))
    }
  }

  function submit(q: string) {
    const trimmed = q.trim()
    if (!trimmed || busy) return
    setQuestion(trimmed)
    setPanels({ contractor: { status: 'loading' }, senior_engineer: { status: 'loading' } })
    // Same question, both roles, in parallel. Each call carries that role's own JWT.
    void Promise.all(ROLES.map((r) => runRole(r, trimmed)))
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault()
    submit(question)
  }

  return (
    <main className="app">
      <h1>Ask my codebase</h1>
      <p className="tagline">
        Same question, two roles. Access is enforced by Postgres row-level security, not by this UI.
      </p>

      <form className="ask" onSubmit={onSubmit}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask how something works…"
          aria-label="Question"
        />
        <button type="submit" disabled={busy || !question.trim()}>
          {busy ? 'Asking…' : 'Ask both'}
        </button>
      </form>

      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button key={ex} type="button" onClick={() => submit(ex)} disabled={busy}>
            {ex.length > 70 ? ex.slice(0, 67) + '…' : ex}
          </button>
        ))}
      </div>

      <div className="panels">
        {ROLES.map((r) => (
          <ResultPanel key={r} role={r} state={panels[r]} />
        ))}
      </div>
    </main>
  )
}
