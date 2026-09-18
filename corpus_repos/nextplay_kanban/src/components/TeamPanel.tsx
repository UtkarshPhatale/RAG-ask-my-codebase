import { useState } from 'react';
import { X, Plus, Trash2 } from 'lucide-react';
import type { TeamMember } from '../types';
import { LABEL_COLORS } from '../types';

interface Props {
  members: TeamMember[];
  onCreate: (name: string, color: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
  onClose: () => void;
}

export function TeamPanel({ members, onCreate, onDelete, onClose }: Props) {
  const [name, setName] = useState('');
  const [color, setColor] = useState(LABEL_COLORS[5]);
  const [saving, setSaving] = useState(false);

  const handleAdd = async () => {
    if (!name.trim()) return;
    setSaving(true);
    try { await onCreate(name.trim(), color); setName(''); }
    finally { setSaving(false); }
  };

  return (
    <div style={{
      position: 'absolute', top: 0, right: 0, bottom: 0,
      width: 300,
      background: 'var(--surface)',
      borderLeft: '1px solid var(--border)',
      display: 'flex', flexDirection: 'column',
      animation: 'slideInRight 0.2s ease',
      zIndex: 10,
      boxShadow: '-8px 0 32px rgba(0,0,0,0.3)',
    }}>
      <style>{`@keyframes slideInRight { from { transform: translateX(100%); } to { transform: translateX(0); } }`}</style>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px 18px', borderBottom: '1px solid var(--border)' }}>
        <span style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 14 }}>Team Members</span>
        <button onClick={onClose} style={{ color: 'var(--text-dim)' }}><X size={16} /></button>
      </div>

      {/* Member list */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px 18px', display: 'flex', flexDirection: 'column', gap: 8 }}>
        {members.length === 0 && (
          <p style={{ fontSize: 12, color: 'var(--text-dim)', textAlign: 'center', marginTop: 24 }}>No team members yet.</p>
        )}
        {members.map(m => (
          <div key={m.id} style={{
            display: 'flex', alignItems: 'center', gap: 10,
            background: 'var(--surface2)', borderRadius: 'var(--radius-sm)',
            padding: '8px 12px',
          }}>
            <div className="avatar" style={{ background: m.color, width: 30, height: 30, fontSize: 12 }}>
              {m.name.charAt(0).toUpperCase()}
            </div>
            <span style={{ flex: 1, fontSize: 13, fontWeight: 500 }}>{m.name}</span>
            <button onClick={() => onDelete(m.id)} style={{ color: 'var(--text-dim)', opacity: 0.5 }}
              onMouseOver={e => (e.currentTarget.style.opacity = '1')}
              onMouseOut={e => (e.currentTarget.style.opacity = '0.5')}>
              <Trash2 size={13} />
            </button>
          </div>
        ))}
      </div>

      {/* Add form */}
      <div style={{ padding: '12px 18px', borderTop: '1px solid var(--border)', display: 'flex', flexDirection: 'column', gap: 10 }}>
        <p style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Add Member</p>
        <input
          value={name} onChange={e => setName(e.target.value)}
          placeholder="Member name…"
          onKeyDown={e => e.key === 'Enter' && handleAdd()}
        />
        {/* Color picker */}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {LABEL_COLORS.map(c => (
            <button key={c} onClick={() => setColor(c)} style={{
              width: 22, height: 22, borderRadius: '50%', background: c,
              border: color === c ? '3px solid var(--text)' : '3px solid transparent',
              transition: 'border 0.1s',
            }} />
          ))}
        </div>
        <button className="btn btn-primary" onClick={handleAdd} disabled={saving || !name.trim()} style={{ width: '100%', justifyContent: 'center' }}>
          <Plus size={14} /> Add Member
        </button>
      </div>
    </div>
  );
}
