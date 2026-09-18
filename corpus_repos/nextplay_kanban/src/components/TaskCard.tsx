import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { Calendar, MessageSquare, AlertCircle } from 'lucide-react';
import { format, isAfter, isBefore, addDays, startOfDay } from 'date-fns';
import type { Task, TeamMember, Label } from '../types';
import { PRIORITY_CONFIG } from '../types';

interface Props {
  task: Task;
  members: TeamMember[];
  labels: Label[];
  onEdit: (task: Task) => void;
  isDragging?: boolean;
}

export function TaskCard({ task, members, labels, onEdit, isDragging }: Props) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging: isSortDragging } = useSortable({ id: task.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isSortDragging ? 0.3 : 1,
  };

  const assignees = members.filter(m => task.assignee_ids?.includes(m.id));
  const taskLabels = labels.filter(l => task.label_ids?.includes(l.id));
  const priority = PRIORITY_CONFIG[task.priority];

  // Due date indicators
  const today = startOfDay(new Date());
  const dueDate = task.due_date ? new Date(task.due_date) : null;
  const isOverdue = dueDate && task.status !== 'done' && isAfter(today, dueDate);
  const isDueSoon = dueDate && task.status !== 'done' && !isOverdue && isBefore(dueDate, addDays(today, 3));

  return (
    <div
      ref={setNodeRef}
      {...attributes}
      {...listeners}
      onClick={() => !isSortDragging && onEdit(task)}
      className={isDragging ? '' : 'task-card'}
      style={{
        ...style,
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)',
        padding: '12px',
        cursor: 'grab',
        transition: 'border-color 0.15s, box-shadow 0.15s, transform 0.15s',
        boxShadow: isDragging ? 'var(--shadow-lg)' : 'none',
        userSelect: 'none',
      }}
      onMouseOver={e => {
        (e.currentTarget as HTMLDivElement).style.borderColor = 'var(--border-hover)';
        (e.currentTarget as HTMLDivElement).style.boxShadow = '0 2px 12px rgba(0,0,0,0.3)';
      }}
      onMouseOut={e => {
        (e.currentTarget as HTMLDivElement).style.borderColor = 'var(--border)';
        (e.currentTarget as HTMLDivElement).style.boxShadow = 'none';
      }}
    >
      {/* Labels row */}
      {taskLabels.length > 0 && (
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 8 }}>
          {taskLabels.map(l => (
            <span key={l.id} className="tag" style={{ background: l.color + '22', color: l.color, fontSize: 10 }}>
              {l.name}
            </span>
          ))}
        </div>
      )}

      {/* Title */}
      <p style={{
        fontSize: 13, fontWeight: 500, lineHeight: 1.4,
        color: task.status === 'done' ? 'var(--text-muted)' : 'var(--text)',
        textDecoration: task.status === 'done' ? 'line-through' : 'none',
        marginBottom: 10,
      }}>
        {task.title}
      </p>

      {/* Footer row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
        {/* Priority dot */}
        <span className="tag" style={{
          background: priority.color + '20', color: priority.color,
          fontSize: 10, padding: '1px 6px',
        }}>
          {priority.label}
        </span>

        {/* Due date */}
        {dueDate && (
          <span style={{
            display: 'inline-flex', alignItems: 'center', gap: 3,
            fontSize: 11, fontWeight: 500,
            color: isOverdue ? 'var(--red)' : isDueSoon ? 'var(--orange)' : 'var(--text-dim)',
            background: isOverdue ? 'rgba(239,68,68,0.1)' : isDueSoon ? 'rgba(249,115,22,0.1)' : 'transparent',
            padding: isOverdue || isDueSoon ? '1px 6px' : 0,
            borderRadius: 99,
          }}>
            {isOverdue && <AlertCircle size={10} />}
            <Calendar size={10} />
            {format(dueDate, 'MMM d')}
          </span>
        )}

        <div style={{ flex: 1 }} />

        {/* Assignee avatars */}
        {assignees.length > 0 && (
          <div style={{ display: 'flex', marginRight: -4 }}>
            {assignees.slice(0, 3).map(m => (
              <div key={m.id} className="avatar" style={{ background: m.color, width: 22, height: 22, fontSize: 9 }}>
                {m.name.charAt(0).toUpperCase()}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
