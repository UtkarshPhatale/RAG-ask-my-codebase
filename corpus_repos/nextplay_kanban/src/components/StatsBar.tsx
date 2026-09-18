import { CheckCircle2, Clock, AlertCircle, LayoutGrid } from 'lucide-react';
import { isAfter, startOfDay } from 'date-fns';
import type { Task } from '../types';

export function StatsBar({ tasks }: { tasks: Task[] }) {
  const total = tasks.length;
  const done = tasks.filter(t => t.status === 'done').length;
  const overdue = tasks.filter(t =>
    t.due_date && t.status !== 'done' && isAfter(startOfDay(new Date()), new Date(t.due_date))
  ).length;
  const inProgress = tasks.filter(t => t.status === 'in_progress').length;

  const pct = total > 0 ? Math.round((done / total) * 100) : 0;

  return (
    <div style={{
      background: 'var(--surface)',
      borderBottom: '1px solid var(--border)',
      padding: '0 24px',
      height: 40,
      display: 'flex',
      alignItems: 'center',
      gap: 24,
      flexShrink: 0,
    }}>
      <Stat icon={<LayoutGrid size={12} />} label="Total" value={total} />
      <Stat icon={<CheckCircle2 size={12} color="var(--done)" />} label="Done" value={`${done} (${pct}%)`} />
      <Stat icon={<Clock size={12} color="var(--progress)" />} label="In Progress" value={inProgress} />
      {overdue > 0 && <Stat icon={<AlertCircle size={12} color="var(--red)" />} label="Overdue" value={overdue} color="var(--red)" />}

      {/* Progress bar */}
      <div style={{ flex: 1, height: 4, background: 'var(--surface3)', borderRadius: 99, overflow: 'hidden', maxWidth: 200, marginLeft: 'auto' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: 'var(--done)', borderRadius: 99, transition: 'width 0.4s ease' }} />
      </div>
      <span style={{ fontSize: 11, color: 'var(--text-dim)', whiteSpace: 'nowrap' }}>{pct}% complete</span>
    </div>
  );
}

function Stat({ icon, label, value, color }: { icon: React.ReactNode; label: string; value: string | number; color?: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <span style={{ color: 'var(--text-dim)' }}>{icon}</span>
      <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>{label}</span>
      <span style={{ fontSize: 12, fontWeight: 600, color: color ?? 'var(--text)' }}>{value}</span>
    </div>
  );
}
