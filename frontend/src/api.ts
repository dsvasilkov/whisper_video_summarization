const API_BASE = (import.meta.env.VITE_API_URL as string) || '/api';

export interface TaskStatusResponse {
  task_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  task_type: string;
  result_transcription: string | null;
  result_summary: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

/** Поставить задачу инференса видео в очередь; результат — через getTaskStatus. */
export async function inferVideoUpload(file: File): Promise<{ task_id: string }> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/infer/video/upload`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

/** Получить статус задачи из БД (для опроса после постановки в очередь). */
export async function getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
  const res = await fetch(`${API_BASE}/tasks/${taskId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function uploadDataset(file: File): Promise<{ path: string }> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/upload/dataset`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function startTraining(configPath: string, datasetPath: string | null): Promise<{ status: string }> {
  const body: { config_path: string; dataset_path?: string } = { config_path: configPath };
  if (datasetPath) body.dataset_path = datasetPath;
  const res = await fetch(`${API_BASE}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
