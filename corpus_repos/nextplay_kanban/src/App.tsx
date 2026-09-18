import { useState, useMemo } from 'react';
import { useBoard } from './hooks/useBoard';
import { Header } from './components/Header';
import { Board } from './components/Board';
import { TaskModal } from './components/TaskModal';
import { TeamPanel } from './components/TeamPanel';
import { LabelsPanel } from './components/LabelsPanel';
import { StatsBar } from './components/StatsBar';
import type { Task, Status } from './types';

export type Panel = 'team' | 'labels' | null;

export default function App() {
  const board = useBoard();
  const [editingTask, setEditingTask] = useState<Task | null>(null);
  const [creatingStatus, setCreatingStatus] = useState<Status | null>(null);
  const [activePanel, setActivePanel] = useState<Panel>(null);
  const [search, setSearch] = useState('');
  const [filterPriority, setFilterPriority] = useState<string>('');
  const [filterLabel, setFilterLabel] = useState<string>('');
  const [filterAssignee, setFilterAssignee] = useState<string>('');

  const filteredTasks = useMemo(() => {
    return board.tasks.filter(t => {
      if (search && !t.title.toLowerCase().includes(search.toLowerCase())) return false;
      if (filterPriority && t.priority !== filterPriority) return false;
      if (filterLabel && !t.label_ids.includes(filterLabel)) return false;
      if (filterAssignee && !t.assignee_ids.includes(filterAssignee)) return false;
      return true;
    });
  }, [board.tasks, search, filterPriority, filterLabel, filterAssignee]);

  if (board.error) return (
    <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:'100vh', flexDirection:'column', gap:16 }}>
      <div style={{ fontSize:48 }}>⚠️</div>
      <p style={{ color:'var(--text-muted)', textAlign:'center', maxWidth:400 }}>{board.error}</p>
      <p style={{ color:'var(--text-dim)', fontSize:12 }}>Check your .env file has valid Supabase credentials.</p>
    </div>
  );

  return (
    <div style={{ display:'flex', flexDirection:'column', height:'100vh', overflow:'hidden' }}>
      <Header
        search={search} onSearch={setSearch}
        filterPriority={filterPriority} onFilterPriority={setFilterPriority}
        filterLabel={filterLabel} onFilterLabel={setFilterLabel}
        filterAssignee={filterAssignee} onFilterAssignee={setFilterAssignee}
        labels={board.labels} members={board.members}
        activePanel={activePanel} onPanel={setActivePanel}
        onNewTask={() => setCreatingStatus('todo')}
      />

      <StatsBar tasks={board.tasks} />

      <div style={{ flex:1, overflow:'hidden', position:'relative' }}>
        {board.loading ? (
          <SkeletonBoard />
        ) : (
          <Board
            tasks={filteredTasks}
            members={board.members}
            labels={board.labels}
            onMoveTask={board.moveTask}
            onEditTask={setEditingTask}
            onCreateTask={setCreatingStatus}
          />
        )}

        {/* Side panels */}
        {activePanel === 'team' && (
          <TeamPanel
            members={board.members}
            onCreate={board.createMember}
            onDelete={board.deleteMember}
            onClose={() => setActivePanel(null)}
          />
        )}
        {activePanel === 'labels' && (
          <LabelsPanel
            labels={board.labels}
            onCreate={board.createLabel}
            onDelete={board.deleteLabel}
            onClose={() => setActivePanel(null)}
          />
        )}
      </div>

      {/* Task create/edit modal */}
      {(editingTask || creatingStatus) && (
        <TaskModal
          task={editingTask}
          defaultStatus={creatingStatus ?? 'todo'}
          members={board.members}
          labels={board.labels}
          onCreate={board.createTask}
          onUpdate={board.updateTask}
          onDelete={board.deleteTask}
          fetchComments={board.fetchComments}
          addComment={board.addComment}
          onClose={() => { setEditingTask(null); setCreatingStatus(null); }}
        />
      )}
    </div>
  );
}

function SkeletonBoard() {
  return (
    <div style={{ display:'flex', gap:16, padding:'16px 24px', height:'100%', overflow:'hidden' }}>
      {[0,1,2,3].map(i => (
        <div key={i} style={{ width:'var(--col-width)', flexShrink:0, display:'flex', flexDirection:'column', gap:10 }}>
          <div className="skeleton" style={{ height:36, width:'70%' }} />
          {[0,1,2].map(j => (
            <div key={j} className="skeleton" style={{ height: 90 + j * 20 }} />
          ))}
        </div>
      ))}
    </div>
  );
}
