export type Status = 'todo' | 'in_progress' | 'in_review' | 'done';
export type Priority = 'low' | 'normal' | 'high';

export interface Label {
  id: string;
  name: string;
  color: string;
  user_id: string;
}

export interface TeamMember {
  id: string;
  name: string;
  color: string;
  user_id: string;
  created_at: string;
}

export interface Task {
  id: string;
  title: string;
  description: string | null;
  status: Status;
  priority: Priority;
  due_date: string | null;
  user_id: string;
  created_at: string;
  assignee_ids: string[];
  label_ids: string[];
}

export interface Comment {
  id: string;
  task_id: string;
  user_id: string;
  body: string;
  created_at: string;
}

export interface ActivityLog {
  id: string;
  task_id: string;
  user_id: string;
  action: string;
  created_at: string;
}

export const COLUMNS: { id: Status; label: string }[] = [
  { id: 'todo', label: 'To Do' },
  { id: 'in_progress', label: 'In Progress' },
  { id: 'in_review', label: 'In Review' },
  { id: 'done', label: 'Done' },
];

export const PRIORITY_CONFIG: Record<Priority, { label: string; color: string }> = {
  low: { label: 'Low', color: '#6ee7b7' },
  normal: { label: 'Normal', color: '#93c5fd' },
  high: { label: 'High', color: '#fca5a5' },
};

export const LABEL_COLORS = [
  '#f87171', '#fb923c', '#fbbf24', '#a3e635',
  '#34d399', '#22d3ee', '#60a5fa', '#a78bfa',
  '#f472b6', '#94a3b8',
];
