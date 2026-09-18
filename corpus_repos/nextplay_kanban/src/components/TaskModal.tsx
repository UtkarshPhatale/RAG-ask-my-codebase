import { useState, useEffect, useRef } from 'react';
import { X, Trash2, Send, Calendar, Flag, User, Tag, ChevronDown } from 'lucide-react';
import { format } from 'date-fns';
import type { Task, TeamMember, Label, Comment, Status, Priority } from '../types';
import { COLUMNS, PRIORITY_CONFIG, LABEL_COLORS } from '../types';

interface Props {
  task: Task | null;
  defaultStatus: Status;
  members: TeamMember[];
  labels: Label[];
  onCreate: (payload: Omit<Task, 'id' | 'user_id' | 'created_at'>) => Promise<Task>;
  onUpdate: (id: string, updates: Partial<Task>) => Promise<Task>;
  onDelete: (id: string) => Promise<void>;
  fetchComments: (taskId: string) => Promise<Comment[]>;
  addComment: (taskId: string, body: string) => Promise<Comment>;
  onClose: () => void;
}

export function TaskModal({
  task, defaultStatus, members, labels,
  onCreate, onUpdate, onDelete,
  fetchComments, addComment, onClose,
}: Props) {
  const isNew = !task;

  const [title, setTitle] = useState(task?.title ?? '');
  const [description, setDescription] = useState(task?.description ?? '');
  const [status, setStatus] = useState<Status>(task?.status ?? defaultStatus);
  const [priority, setPriority] = useState<Priority>(task?.priority ?? 'normal');
  const [dueDate, setDueDate] = useState(task?.due_date ?? '');
  const [assigneeIds, setAssigneeIds] = useState<string[]>(task?.assignee_ids ?? []);
  const [labelIds, setLabelIds] = useState<string[]>(task?.label_ids ?? []);

  const [comments, setComments] = useState<Comment[]>([]);
  const [commentText, setCommentText] = useState('');
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [sendingComment, setSendingComment] = useState(false);
  const [activeTab, setActiveTab] = useState<'details' | 'comments'>('details');

  const titleRef = useRef<HTMLInputElement>(null);

  useEffect(() => { titleRef.current?.focus(); }, []);

  useEffect(() => {
    if (task) {
      fetchComments(task.id).then(setComments).catch(() => {});
    }
  }, [task?.id]);

  const toggleAssignee = (id: string) =>
    setAssigneeIds(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);

  const toggleLabel = (id: string) =>
    setLabelIds(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);

  const handleSave = async () => {
    if (!title.trim()) return;
    setSaving(true);
    try {
      const payload = {
        title: title.trim(),
        description: description || null,
        status, priority,
        due_date: dueDate || null,
        assignee_ids: assigneeIds,
        label_ids: labelIds,
      };
      if (isNew) {
        await onCreate(payload);
      } else {
        await onUpdate(task!.id, payload);
      }
      onClose();
    } catch (e) {
      console.error(e);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!task || !confirm('Delete this task?')) return;
    setDeleting(true);
    try { await onDelete(task.id); onClose(); }
    catch (e) { console.error(e); setDeleting(false); }
  };

  const handleComment = async () => {
    if (!task || !commentText.trim()) return;
    setSendingComment(true);
    try {
      const c = await addComment(task.id, commentText.trim());
      setComments(prev => [...prev, c]);
      setCommentText('');
    } finally {
      setSendingComment(false);
    }
  };

  return (
    <div className="overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal" style={{ width: 580 }}>
        {/* Modal header */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '18px 20px 0',
        }}>
          <span style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 15, color: 'var(--text-muted)' }}>
            {isNew ? 'New Task' : 'Edit Task'}
          </span>
          <div style={{ display: 'flex', gap: 8 }}>
            {!isNew && (
              <button className="btn btn-ghost" style={{ padding: '6px 10px', fontSize: 12 }}
                onClick={handleDelete} disabled={deleting}>
                <Trash2 size={13} color="var(--red)" />
              </button>
            )}
            <button onClick={onClose} style={{ color: 'var(--text-dim)', padding: 4, borderRadius: 6 }}>
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Title */}
        <div style={{ padding: '14px 20px 0' }}>
          <input
            ref={titleRef}
            value={title}
            onChange={e => setTitle(e.target.value)}
            placeholder="Task title…"
            style={{
              background: 'transparent', border: 'none', borderBottom: '1px solid var(--border)',
              borderRadius: 0, fontSize: 20, fontWeight: 600, fontFamily: 'var(--font-display)',
              padding: '4px 0 10px', color: 'var(--text)',
            }}
            onFocus={e => e.currentTarget.style.borderBottomColor = 'var(--accent)'}
            onBlur={e => e.currentTarget.style.borderBottomColor = 'var(--border)'}
            onKeyDown={e => e.key === 'Enter' && handleSave()}
          />
        </div>

        {/* Tabs (only for existing tasks) */}
        {!isNew && (
          <div style={{ display: 'flex', gap: 0, padding: '12px 20px 0', borderBottom: '1px solid var(--border)' }}>
            {(['details', 'comments'] as const).map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)} style={{
                padding: '6px 14px', fontSize: 12, fontWeight: 600,
                color: activeTab === tab ? 'var(--accent)' : 'var(--text-dim)',
                borderBottom: activeTab === tab ? '2px solid var(--accent)' : '2px solid transparent',
                marginBottom: -1, borderRadius: 0, textTransform: 'capitalize',
                transition: 'color 0.15s',
              }}>
                {tab} {tab === 'comments' && comments.length > 0 && `(${comments.length})`}
              </button>
            ))}
          </div>
        )}

        {/* Tab content */}
        <div style={{ padding: '16px 20px' }}>
          {(isNew || activeTab === 'details') && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {/* Description */}
              <div className="form-group">
                <label className="form-label">Description</label>
                <textarea
                  value={description}
                  onChange={e => setDescription(e.target.value)}
                  placeholder="Add more context…"
                  rows={3}
                />
              </div>

              {/* Status + Priority */}
              <div className="form-row">
                <div className="form-group">
                  <label className="form-label"><Flag size={10} style={{ display: 'inline' }} /> Status</label>
                  <select value={status} onChange={e => setStatus(e.target.value as Status)}>
                    {COLUMNS.map(c => <option key={c.id} value={c.id}>{c.label}</option>)}
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Priority</label>
                  <select value={priority} onChange={e => setPriority(e.target.value as Priority)}>
                    {Object.entries(PRIORITY_CONFIG).map(([k, v]) => (
                      <option key={k} value={k}>{v.label}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Due date */}
              <div className="form-group" style={{ maxWidth: 200 }}>
                <label className="form-label"><Calendar size={10} style={{ display: 'inline' }} /> Due Date</label>
                <input type="date" value={dueDate} onChange={e => setDueDate(e.target.value)} />
              </div>

              {/* Assignees */}
              {members.length > 0 && (
                <div className="form-group">
                  <label className="form-label"><User size={10} style={{ display: 'inline' }} /> Assignees</label>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {members.map(m => {
                      const selected = assigneeIds.includes(m.id);
                      return (
                        <button key={m.id} onClick={() => toggleAssignee(m.id)} style={{
                          display: 'flex', alignItems: 'center', gap: 6,
                          padding: '5px 10px', borderRadius: 99,
                          border: `1.5px solid ${selected ? m.color : 'var(--border)'}`,
                          background: selected ? m.color + '22' : 'transparent',
                          color: selected ? m.color : 'var(--text-muted)',
                          fontSize: 12, fontWeight: 500, transition: 'all 0.15s',
                        }}>
                          <div className="avatar" style={{ background: m.color, width: 18, height: 18, fontSize: 8 }}>
                            {m.name.charAt(0).toUpperCase()}
                          </div>
                          {m.name}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Labels */}
              {labels.length > 0 && (
                <div className="form-group">
                  <label className="form-label"><Tag size={10} style={{ display: 'inline' }} /> Labels</label>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {labels.map(l => {
                      const selected = labelIds.includes(l.id);
                      return (
                        <button key={l.id} onClick={() => toggleLabel(l.id)} className="tag" style={{
                          background: selected ? l.color + '30' : 'var(--surface2)',
                          color: selected ? l.color : 'var(--text-muted)',
                          border: `1.5px solid ${selected ? l.color : 'transparent'}`,
                          cursor: 'pointer', transition: 'all 0.15s',
                        }}>
                          {l.name}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Comments tab */}
          {!isNew && activeTab === 'comments' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {comments.length === 0 && (
                <div style={{ color: 'var(--text-dim)', fontSize: 12, textAlign: 'center', padding: '24px 0' }}>
                  No comments yet — start the conversation.
                </div>
              )}
              {comments.map(c => (
                <div key={c.id} style={{
                  background: 'var(--surface2)', borderRadius: 'var(--radius-sm)',
                  padding: '10px 14px',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent-light)' }}>Guest</span>
                    <span style={{ fontSize: 10, color: 'var(--text-dim)' }}>
                      {format(new Date(c.created_at), 'MMM d, h:mm a')}
                    </span>
                  </div>
                  <p style={{ fontSize: 13, color: 'var(--text)', lineHeight: 1.5 }}>{c.body}</p>
                </div>
              ))}
              <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                <textarea
                  value={commentText}
                  onChange={e => setCommentText(e.target.value)}
                  placeholder="Write a comment…"
                  rows={2}
                  style={{ flex: 1, minHeight: 'unset' }}
                  onKeyDown={e => { if (e.key === 'Enter' && e.metaKey) handleComment(); }}
                />
                <button className="btn btn-primary" onClick={handleComment} disabled={sendingComment || !commentText.trim()}
                  style={{ alignSelf: 'flex-end', padding: '8px 12px' }}>
                  <Send size={14} />
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{
          display: 'flex', justifyContent: 'flex-end', gap: 8,
          padding: '12px 20px 18px',
          borderTop: '1px solid var(--border)',
        }}>
          <button className="btn btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" onClick={handleSave} disabled={saving || !title.trim()}>
            {saving ? 'Saving…' : isNew ? 'Create Task' : 'Save Changes'}
          </button>
        </div>
      </div>
    </div>
  );
}
