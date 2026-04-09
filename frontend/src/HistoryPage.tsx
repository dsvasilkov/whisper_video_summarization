import { useCallback, useEffect, useState } from 'react';
import type { TaskStatusResponse } from './api';
import { listTasks } from './api';
import { useAuth } from './AuthContext';
import { ResultViewSwitch } from './TranscriptViewer';

const statusLabel: Record<TaskStatusResponse['status'], string> = {
  pending: 'В очереди',
  processing: 'Обработка',
  completed: 'Готово',
  failed: 'Ошибка',
};

function formatDate(iso: string) {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export function HistoryPage() {
  const { token, logout } = useAuth();
  const [rows, setRows] = useState<TaskStatusResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<TaskStatusResponse | null>(null);

  const load = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    setError(null);
    try {
      const data = await listTasks(token);
      setRows(data);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      if (
        msg.includes('Not authenticated') ||
        msg.includes('Invalid or expired token')
      ) {
        logout();
      }
    } finally {
      setLoading(false);
    }
  }, [token, logout]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="card">
      <div className="history-header">
        <h2>История задач</h2>
        <button type="button" className="btn btn-secondary btn-small" onClick={() => void load()}>
          Обновить
        </button>
      </div>
      {loading && <p className="loading">Загрузка…</p>}
      {error && <p className="error">{error}</p>}
      {!loading && !error && rows.length === 0 && (
        <p className="muted">Пока нет задач. Запустите инференс на вкладке «Инференс».</p>
      )}
      {!loading && rows.length > 0 && (
        <div className="table-wrap">
          <table className="task-table">
            <thead>
              <tr>
                <th>Создано</th>
                <th>Статус</th>
                <th>Тип</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {rows.map((t) => (
                <tr key={t.task_id}>
                  <td>{formatDate(t.created_at)}</td>
                  <td>{statusLabel[t.status]}</td>
                  <td>{t.task_type}</td>
                  <td>
                    <button
                      type="button"
                      className="btn btn-small"
                      onClick={() => setOpen(t)}
                    >
                      Подробнее
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {open && (
        <div
          className="modal-backdrop"
          role="presentation"
          onClick={() => setOpen(null)}
        >
          <div
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="modal-title"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-head">
              <h3 id="modal-title">Задача</h3>
              <button type="button" className="modal-close" onClick={() => setOpen(null)}>
                ×
              </button>
            </div>
            <p className="muted small">
              {formatDate(open.created_at)} · {statusLabel[open.status]}
            </p>
            {open.error_message && (
              <div className="result-block">
                <h4>Ошибка</h4>
                <pre>{open.error_message}</pre>
              </div>
            )}
            {(open.result_transcription != null || open.result_summary != null) && (
              <div className="result-block">
                <h4>Результат</h4>
                <ResultViewSwitch
                  key={open.task_id}
                  transcription={open.result_transcription}
                  summary={open.result_summary}
                />
              </div>
            )}
            {open.status === 'completed' &&
              open.result_transcription == null &&
              open.result_summary == null && (
                <p className="muted">Результаты ещё не записаны.</p>
              )}
          </div>
        </div>
      )}
    </div>
  );
}
