import { useDroppable } from '@dnd-kit/core';
import { SortableContext, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { Plus } from 'lucide-react';
import { TaskCard } from './TaskCard';
import type { Task, TeamMember, Label, Status } from '../types';

const COL_COLORS: Record<Status, string> = {
  todo: 'var(--todo)',
  in_progress: 'var(--progress)',
  in_review: 'var(--review)',
  done: 'var(--done)',
};

interface Props {
  column: { id: Status; label: string };
  tasks: Task[];
  members: TeamMember[];
  labels: Label[];
  onEditTask: (task: Task) => void;
  onCreateTask: () => void;
  isDragging: boolean;
}

export function Column({ column, tasks, members, labels, onEditTask, onCreateTask, isDragging }: Props) {
  const { setNodeRef, isOver } = useDroppable({ id: column.id });
  const color = COL_COLORS[column.id];

  return (
    <div style={{
      width: 'var(--col-width)',
      flexShrink: 0,
      display: 'flex',
      flexDirection: 'column',
      gap: 0,
      maxHeight: '100%',
    }}>
      {/* Column header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 4px 10px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: color }} />
          <span style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 13, letterSpacing: '0.03em' }}>
            {column.label}
          </span>
          <span style={{
            background: 'var(--surface3)', color: 'var(--text-muted)',
            fontSize: 11, fontWeight: 600,
            padding: '1px 7px', borderRadius: 99,
          }}>{tasks.length}</span>
        </div>
        <button
          onClick={onCreateTask}
          style={{ color: 'var(--text-dim)', padding: 4, borderRadius: 6, transition: 'all 0.15s' }}
          onMouseOver={e => (e.currentTarget.style.color = 'var(--accent)')}
          onMouseOut={e => (e.currentTarget.style.color = 'var(--text-dim)')}
        >
          <Plus size={15} />
        </button>
      </div>

      {/* Drop zone */}
      <div
        ref={setNodeRef}
        style={{
          flex: 1,
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: 8,
          padding: '4px 2px',
          borderRadius: 'var(--radius)',
          transition: 'background 0.15s',
          background: isOver ? `${color}12` : 'transparent',
          border: isOver ? `1px dashed ${color}40` : '1px solid transparent',
          minHeight: 80,
        }}
      >
        <SortableContext items={tasks.map(t => t.id)} strategy={verticalListSortingStrategy}>
          {tasks.map(task => (
            <TaskCard key={task.id} task={task} members={members} labels={labels} onEdit={onEditTask} />
          ))}
        </SortableContext>

        {tasks.length === 0 && !isDragging && (
          <div style={{
            flex: 1, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            gap: 8, minHeight: 100,
            color: 'var(--text-dim)', fontSize: 12,
          }}>
            <div style={{ width: 32, height: 32, borderRadius: 99, border: `1.5px dashed ${color}50`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Plus size={14} color={`${color}80`} />
            </div>
            <span>Drop tasks here</span>
          </div>
        )}
      </div>
    </div>
  );
}
