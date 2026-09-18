import { Search, Users, Tag, Plus, SlidersHorizontal } from 'lucide-react';
import type { Label, TeamMember } from '../types';
import type { Panel } from '../App';

interface Props {
  search: string; onSearch: (v: string) => void;
  filterPriority: string; onFilterPriority: (v: string) => void;
  filterLabel: string; onFilterLabel: (v: string) => void;
  filterAssignee: string; onFilterAssignee: (v: string) => void;
  labels: Label[]; members: TeamMember[];
  activePanel: Panel; onPanel: (p: Panel) => void;
  onNewTask: () => void;
}

export function Header({ search, onSearch, filterPriority, onFilterPriority, filterLabel, onFilterLabel, filterAssignee, onFilterAssignee, labels, members, activePanel, onPanel, onNewTask }: Props) {
  const hasFilters = filterPriority || filterLabel || filterAssignee;

  return (
    <header style={{
      height: 'var(--header-h)',
      borderBottom: '1px solid var(--border)',
      background: 'var(--surface)',
      display: 'flex',
      alignItems: 'center',
      padding: '0 20px',
      gap: 12,
      flexShrink: 0,
    }}>
      {/* Logo */}
      <div style={{ fontFamily: 'var(--font-display)', fontWeight: 800, fontSize: 18, color: 'var(--accent-light)', whiteSpace: 'nowrap', marginRight: 8 }}>
        Next<span style={{ color: 'var(--text)' }}>Play</span>
      </div>

      {/* Search */}
      <div style={{ position: 'relative', flex: '0 0 220px' }}>
        <Search size={14} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-dim)', pointerEvents: 'none' }} />
        <input
          value={search}
          onChange={e => onSearch(e.target.value)}
          placeholder="Search tasks…"
          style={{ paddingLeft: 32, height: 36, borderRadius: 99 }}
        />
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <SlidersHorizontal size={14} style={{ color: hasFilters ? 'var(--accent)' : 'var(--text-dim)' }} />
        <select
          value={filterPriority}
          onChange={e => onFilterPriority(e.target.value)}
          style={{ width: 'auto', height: 32, padding: '0 10px', borderRadius: 99, fontSize: 12 }}
        >
          <option value="">All Priorities</option>
          <option value="low">Low</option>
          <option value="normal">Normal</option>
          <option value="high">High</option>
        </select>

        {labels.length > 0 && (
          <select
            value={filterLabel}
            onChange={e => onFilterLabel(e.target.value)}
            style={{ width: 'auto', height: 32, padding: '0 10px', borderRadius: 99, fontSize: 12 }}
          >
            <option value="">All Labels</option>
            {labels.map(l => <option key={l.id} value={l.id}>{l.name}</option>)}
          </select>
        )}

        {members.length > 0 && (
          <select
            value={filterAssignee}
            onChange={e => onFilterAssignee(e.target.value)}
            style={{ width: 'auto', height: 32, padding: '0 10px', borderRadius: 99, fontSize: 12 }}
          >
            <option value="">All Members</option>
            {members.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
          </select>
        )}
      </div>

      <div style={{ flex: 1 }} />

      {/* Right actions */}
      <div style={{ display: 'flex', gap: 8 }}>
        <button
          className={`btn btn-ghost`}
          style={{ height: 36, padding: '0 12px', fontSize: 13, borderColor: activePanel === 'labels' ? 'var(--accent)' : undefined, color: activePanel === 'labels' ? 'var(--accent)' : undefined }}
          onClick={() => onPanel(activePanel === 'labels' ? null : 'labels')}
        >
          <Tag size={14} /> Labels
        </button>
        <button
          className="btn btn-ghost"
          style={{ height: 36, padding: '0 12px', fontSize: 13, borderColor: activePanel === 'team' ? 'var(--accent)' : undefined, color: activePanel === 'team' ? 'var(--accent)' : undefined }}
          onClick={() => onPanel(activePanel === 'team' ? null : 'team')}
        >
          <Users size={14} /> Team
        </button>
        <button className="btn btn-primary" style={{ height: 36, padding: '0 14px', fontSize: 13 }} onClick={onNewTask}>
          <Plus size={14} /> New Task
        </button>
      </div>
    </header>
  );
}
