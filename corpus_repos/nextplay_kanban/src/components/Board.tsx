import { useState } from 'react';
import {
  DndContext, DragOverlay, PointerSensor, useSensor, useSensors,
  type DragStartEvent, type DragEndEvent,
} from '@dnd-kit/core';
import { Column } from './Column';
import { TaskCard } from './TaskCard';
import type { Task, TeamMember, Label, Status } from '../types';
import { COLUMNS } from '../types';

interface Props {
  tasks: Task[];
  members: TeamMember[];
  labels: Label[];
  onMoveTask: (id: string, status: Status) => Promise<void>;
  onEditTask: (task: Task) => void;
  onCreateTask: (status: Status) => void;
}

export function Board({ tasks, members, labels, onMoveTask, onEditTask, onCreateTask }: Props) {
  const [activeTask, setActiveTask] = useState<Task | null>(null);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } })
  );

  const handleDragStart = (e: DragStartEvent) => {
    const task = tasks.find(t => t.id === e.active.id);
    setActiveTask(task ?? null);
  };

  const handleDragEnd = (e: DragEndEvent) => {
    setActiveTask(null);
    const { active, over } = e;
    if (!over) return;
    const newStatus = over.id as Status;
    const task = tasks.find(t => t.id === active.id);
    if (task && task.status !== newStatus) {
      onMoveTask(task.id, newStatus);
    }
  };

  return (
    <DndContext sensors={sensors} onDragStart={handleDragStart} onDragEnd={handleDragEnd}>
      <div style={{
        display: 'flex',
        gap: 14,
        padding: '16px 20px 16px',
        height: '100%',
        overflowX: 'auto',
        overflowY: 'hidden',
      }}>
        {COLUMNS.map(col => (
          <Column
            key={col.id}
            column={col}
            tasks={tasks.filter(t => t.status === col.id)}
            members={members}
            labels={labels}
            onEditTask={onEditTask}
            onCreateTask={() => onCreateTask(col.id)}
            isDragging={activeTask?.status === col.id}
          />
        ))}
      </div>

      <DragOverlay dropAnimation={{ duration: 150, easing: 'ease' }}>
        {activeTask && (
          <div style={{ transform: 'rotate(2deg)', opacity: 0.95 }}>
            <TaskCard task={activeTask} members={members} labels={labels} onEdit={() => {}} isDragging />
          </div>
        )}
      </DragOverlay>
    </DndContext>
  );
}
