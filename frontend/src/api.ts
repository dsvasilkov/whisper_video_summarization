// Относительный /api — Vite proxy на бэкенд. Иначе VITE_API_URL (например http://localhost:8000/api).
const _raw = (import.meta.env.VITE_API_URL as string) || '';
const API_BASE = _raw
  ? (_raw.endsWith('/api') ? _raw.replace(/\/+$/, '') : _raw.replace(/\/+$/, '') + '/api')
  : '/api';

const TOKEN_KEY = 'whisper_auth_token';

export function getStoredToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setStoredToken(token: string | null): void {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

export interface TaskStatusResponse {
  task_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  task_type: string;
  result_transcription: unknown | null;
  result_summary: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

function authHeaders(token: string): HeadersInit {
  return { Authorization: `Bearer ${token}` };
}

async function errorText(res: Response): Promise<string> {
  const text = await res.text();
  try {
    const j = JSON.parse(text) as { detail?: unknown };
    const d = j.detail;
    if (typeof d === 'string') return d;
    if (Array.isArray(d))
      return d.map((x: { msg?: string }) => x.msg ?? String(x)).join(', ');
  } catch {
    /* not JSON */
  }
  return text.trim() || `HTTP ${res.status}`;
}

export async function register(email: string, password: string): Promise<{ access_token: string }> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export async function login(email: string, password: string): Promise<{ access_token: string }> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export async function forgotPassword(
  email: string,
): Promise<{ message: string; reset_token?: string | null }> {
  const res = await fetch(`${API_BASE}/auth/forgot-password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email }),
  });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export async function resetPassword(token: string, newPassword: string): Promise<void> {
  const res = await fetch(`${API_BASE}/auth/reset-password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token, new_password: newPassword }),
  });
  if (!res.ok) throw new Error(await errorText(res));
}

/** Загрузить аудио (требуется JWT). */
export async function uploadAudio(
  file: File,
  token: string,
  options?: { forceDisableDiarization?: boolean },
): Promise<{ task_id: string }> {
  const form = new FormData();
  form.append('file', file);
  if (options?.forceDisableDiarization) {
    form.append('force_disable_diarization', 'true');
  }
  const res = await fetch(`${API_BASE}/uploads/audio`, {
    method: 'POST',
    headers: authHeaders(token),
    body: form,
  });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export async function getTaskStatus(taskId: string, token: string): Promise<TaskStatusResponse> {
  const res = await fetch(`${API_BASE}/tasks/${taskId}`, { headers: authHeaders(token) });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export async function listTasks(token: string, limit = 100): Promise<TaskStatusResponse[]> {
  const res = await fetch(`${API_BASE}/tasks?limit=${limit}`, { headers: authHeaders(token) });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}
