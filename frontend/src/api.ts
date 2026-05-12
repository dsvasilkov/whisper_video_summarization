// Относительный /api — Vite proxy на бэкенд. Иначе VITE_API_URL (например http://localhost:8000/api).
const _raw = (import.meta.env.VITE_API_URL as string) || '';
const API_BASE = _raw
  ? (_raw.endsWith('/api') ? _raw.replace(/\/+$/, '') : _raw.replace(/\/+$/, '') + '/api')
  : '/api';

import { sha256 } from '@noble/hashes/sha2.js';
import { bytesToHex } from '@noble/hashes/utils.js';

const TOKEN_KEY = 'whisper_auth_token';

export function getStoredToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setStoredToken(token: string | null): void {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

export interface TopicGraphNode {
  id: string;
  label: string;
  description?: string;
  community?: number;
  summary?: string;
  /** Сырой текст всех реплик сообщества (для подтем); с бэкенда как camelCase JSON. */
  communityBody?: string;
  /** Секунды: временной диапазон узла в записи лекции.
   * Доступно для подтем, тем и узла «Лекция» (если в графе есть t0/t1).
   */
  communityTimeStart?: number | null;
  communityTimeEnd?: number | null;
  keywords?: string[];
  position?: { x: number; y: number };
  kind?: 'lecture' | 'theme' | 'subtopic' | 'micro' | 'macro' | 'topic';
  parentId?: string | null;
}

export interface TopicGraphLink {
  source: string;
  target: string;
  type?: string;
  weight?: number;
}

export interface TopicGraphPayload {
  nodes: TopicGraphNode[];
  links: TopicGraphLink[];
}

export interface TaskStatusResponse {
  task_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  task_type: string;
  result_transcription: unknown | null;
  result_summary: string | null;
  result_topic_graph?: TopicGraphPayload | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface TaskEventPayload {
  task_id: string;
  status: TaskStatusResponse['status'];
  task_type: string;
  error_message: string | null;
  updated_at: string | null;
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

async function sha256Hex(file: Blob): Promise<string> {
  const buf = await file.arrayBuffer();
  const subtle = globalThis.crypto?.subtle;
  if (subtle?.digest) {
    const digest = await subtle.digest('SHA-256', buf);
    const bytes = new Uint8Array(digest);
    let hex = '';
    for (const b of bytes) hex += b.toString(16).padStart(2, '0');
    return hex;
  }
  // Fallback for non-secure contexts / older browsers without WebCrypto.
  return bytesToHex(sha256(new Uint8Array(buf)));
}

/** Загрузить аудио (требуется JWT). */
export async function uploadAudio(
  file: File,
  token: string,
): Promise<{ task_id: string }> {
  const sha256 = await sha256Hex(file);

  // 1) Get presigned PUT URL (upload goes directly to MinIO/S3)
  const presign = await fetch(`${API_BASE}/uploads/audio/presign`, {
    method: 'POST',
    headers: { ...authHeaders(token), 'Content-Type': 'application/json' },
    body: JSON.stringify({
      filename: file.name || 'audio.wav',
      content_type: file.type || 'application/octet-stream',
      sha256,
    }),
  });
  if (!presign.ok) throw new Error(await errorText(presign));
  const presigned = (await presign.json()) as {
    task_id: string;
    upload_url: string;
    required_headers?: Record<string, string>;
    s3_uri: string;
  };

  // 2) Upload file directly to MinIO/S3
  const putRes = await fetch(presigned.upload_url, {
    method: 'PUT',
    headers: presigned.required_headers || { 'Content-Type': file.type || 'application/octet-stream' },
    body: file,
  });
  if (!putRes.ok) {
    const t = (await putRes.text()).trim();
    throw new Error(t || `Upload to object storage failed (HTTP ${putRes.status})`);
  }

  // 3) MinIO webhook will notify backend to enqueue ASR automatically.
  return { task_id: presigned.task_id };
}

export async function getTaskStatus(taskId: string, token: string): Promise<TaskStatusResponse> {
  const res = await fetch(`${API_BASE}/tasks/${taskId}`, {
    headers: authHeaders(token),
    cache: 'no-store',
  });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export async function listTasks(token: string, limit = 100): Promise<TaskStatusResponse[]> {
  const res = await fetch(`${API_BASE}/tasks?limit=${limit}`, {
    headers: authHeaders(token),
    cache: 'no-store',
  });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

/** SSE: нативный EventSource не шлёт Authorization — fetch + stream. */
export function subscribeTaskEvents(
  taskId: string,
  token: string,
  onEvent: (ev: TaskEventPayload) => void,
  onError?: (err: unknown) => void,
): { close: () => void } {
  const controller = new AbortController();
  let sawTerminalStatus = false;

  void (async () => {
    const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
    let backoffMs = 500;

    const readStreamOnce = async (): Promise<void> => {
      const res = await fetch(`${API_BASE}/tasks/${taskId}/events`, {
        method: 'GET',
        headers: { ...authHeaders(token), Accept: 'text/event-stream' },
        cache: 'no-store',
        signal: controller.signal,
      });
      if (!res.ok) throw new Error(await errorText(res));
      if (!res.body) throw new Error('SSE stream is not supported by this browser');

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buf = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        buf = buf.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

        while (true) {
          const sep = buf.indexOf('\n\n');
          if (sep === -1) break;
          const frame = buf.slice(0, sep);
          buf = buf.slice(sep + 2);

          const lines = frame.split('\n');
          const dataLines = lines
            .map((l) => l.trimEnd())
            .filter((l) => l.startsWith('data:'))
            .map((l) => l.slice('data:'.length).trim());

          if (dataLines.length === 0) continue;
          const dataText = dataLines.join('\n');
          try {
            const payload = JSON.parse(dataText) as TaskEventPayload;
            if (payload && typeof payload.task_id === 'string') {
              if (payload.status === 'completed' || payload.status === 'failed') {
                sawTerminalStatus = true;
              }
              onEvent(payload);
            }
          } catch (e) {
            onError?.(e);
          }
        }
      }
    };

    while (!controller.signal.aborted) {
      try {
        await readStreamOnce();
        if (sawTerminalStatus) break;
        backoffMs = 500;
      } catch (e) {
        if (e instanceof DOMException && e.name === 'AbortError') return;
        onError?.(e);
        backoffMs = Math.min(15_000, Math.round(backoffMs * 1.6));
      }
      if (controller.signal.aborted || sawTerminalStatus) break;
      await sleep(backoffMs);
    }
  })();

  return {
    close: () => controller.abort(),
  };
}

export interface TaskQuestionAnswerResponse {
  answer: string;
}

/** RAG-QA по завершённой задаче (должны быть Redis result backend и воркер очереди `rag`). */
export async function askTaskQuestion(
  taskId: string,
  question: string,
  token: string,
): Promise<TaskQuestionAnswerResponse> {
  const res = await fetch(`${API_BASE}/tasks/${taskId}/qa`, {
    method: 'POST',
    headers: { ...authHeaders(token), 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export interface ChunkEmbeddingItem {
  chunk_id: number;
  embedding: number[];
}

export interface TaskChunkEmbeddingsResponse {
  chunks: ChunkEmbeddingItem[];
}

/** Встраивания чанков транскрипта (тяжёлая задача на воркере `rag`). */
export async function getTaskChunkEmbeddings(
  taskId: string,
  token: string,
): Promise<TaskChunkEmbeddingsResponse> {
  const res = await fetch(`${API_BASE}/tasks/${taskId}/chunks/embeddings`, {
    headers: authHeaders(token),
  });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}
