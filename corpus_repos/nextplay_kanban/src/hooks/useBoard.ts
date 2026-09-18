import { useState, useEffect, useCallback } from 'react';
import { supabase, ensureGuestSession } from '../lib/supabase';
import type { Task, TeamMember, Label, Comment, Status } from '../types';

export function useBoard() {
  const [userId, setUserId] = useState<string | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [members, setMembers] = useState<TeamMember[]>([]);
  const [labels, setLabels] = useState<Label[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Init guest session
  useEffect(() => {
    ensureGuestSession()
      .then(setUserId)
      .catch(e => setError(e.message));
  }, []);

  // Load all data once we have a user
  useEffect(() => {
    if (!userId) return;
    Promise.all([fetchTasks(), fetchMembers(), fetchLabels()])
      .finally(() => setLoading(false));
  }, [userId]);

  const fetchTasks = useCallback(async () => {
    const { data, error } = await supabase
      .from('tasks')
      .select('*')
      .order('created_at', { ascending: true });
    if (error) { setError(error.message); return; }
    setTasks(data as Task[]);
  }, []);

  const fetchMembers = useCallback(async () => {
    const { data, error } = await supabase
      .from('team_members')
      .select('*')
      .order('created_at', { ascending: true });
    if (error) return;
    setMembers(data as TeamMember[]);
  }, []);

  const fetchLabels = useCallback(async () => {
    const { data, error } = await supabase
      .from('labels')
      .select('*')
      .order('created_at', { ascending: true });
    if (error) return;
    setLabels(data as Label[]);
  }, []);

  // ─── Tasks ─────────────────────────────────────────────────────────────────

  const createTask = useCallback(async (payload: Omit<Task, 'id' | 'user_id' | 'created_at'>) => {
    const { data, error } = await supabase
      .from('tasks')
      .insert({ ...payload, user_id: userId })
      .select()
      .single();
    if (error) throw error;
    setTasks(prev => [...prev, data as Task]);
    return data as Task;
  }, [userId]);

  const updateTask = useCallback(async (id: string, updates: Partial<Task>) => {
    const { data, error } = await supabase
      .from('tasks')
      .update(updates)
      .eq('id', id)
      .select()
      .single();
    if (error) throw error;
    setTasks(prev => prev.map(t => t.id === id ? data as Task : t));
    return data as Task;
  }, []);

  const deleteTask = useCallback(async (id: string) => {
    const { error } = await supabase.from('tasks').delete().eq('id', id);
    if (error) throw error;
    setTasks(prev => prev.filter(t => t.id !== id));
  }, []);

  const moveTask = useCallback(async (id: string, status: Status) => {
    // Optimistic update
    setTasks(prev => prev.map(t => t.id === id ? { ...t, status } : t));
    const { error } = await supabase.from('tasks').update({ status }).eq('id', id);
    if (error) {
      // Revert on failure
      await fetchTasks();
      throw error;
    }
  }, [fetchTasks]);

  // ─── Team Members ───────────────────────────────────────────────────────────

  const createMember = useCallback(async (name: string, color: string) => {
    const { data, error } = await supabase
      .from('team_members')
      .insert({ name, color, user_id: userId })
      .select()
      .single();
    if (error) throw error;
    setMembers(prev => [...prev, data as TeamMember]);
  }, [userId]);

  const deleteMember = useCallback(async (id: string) => {
    const { error } = await supabase.from('team_members').delete().eq('id', id);
    if (error) throw error;
    setMembers(prev => prev.filter(m => m.id !== id));
  }, []);

  // ─── Labels ─────────────────────────────────────────────────────────────────

  const createLabel = useCallback(async (name: string, color: string) => {
    const { data, error } = await supabase
      .from('labels')
      .insert({ name, color, user_id: userId })
      .select()
      .single();
    if (error) throw error;
    setLabels(prev => [...prev, data as Label]);
  }, [userId]);

  const deleteLabel = useCallback(async (id: string) => {
    const { error } = await supabase.from('labels').delete().eq('id', id);
    if (error) throw error;
    setLabels(prev => prev.filter(l => l.id !== id));
  }, []);

  // ─── Comments ───────────────────────────────────────────────────────────────

  const fetchComments = useCallback(async (taskId: string): Promise<Comment[]> => {
    const { data, error } = await supabase
      .from('comments')
      .select('*')
      .eq('task_id', taskId)
      .order('created_at', { ascending: true });
    if (error) throw error;
    return data as Comment[];
  }, []);

  const addComment = useCallback(async (taskId: string, body: string): Promise<Comment> => {
    const { data, error } = await supabase
      .from('comments')
      .insert({ task_id: taskId, body, user_id: userId })
      .select()
      .single();
    if (error) throw error;
    return data as Comment;
  }, [userId]);

  return {
    userId, tasks, members, labels, loading, error,
    createTask, updateTask, deleteTask, moveTask,
    createMember, deleteMember,
    createLabel, deleteLabel,
    fetchComments, addComment,
    refetch: fetchTasks,
  };
}
