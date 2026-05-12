import { useCallback, useState } from 'react';
import { askTaskQuestion } from './api';
import { LlmMarkdown } from './LlmMarkdown';

type TaskQaFormProps = {
  taskId: string;
  token: string;
  onAuthError: () => void;
};

export function TaskQaForm({ taskId, token, onAuthError }: TaskQaFormProps) {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = useCallback(async () => {
    const q = question.trim();
    if (!q || loading) return;
    setLoading(true);
    setError(null);
    setAnswer(null);
    try {
      const { answer: a } = await askTaskQuestion(taskId, q, token);
      setAnswer(a);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes('Not authenticated') || msg.includes('Invalid or expired token')) {
        onAuthError();
      }
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [taskId, token, question, loading, onAuthError]);

  return (
    <div className="result-block qa-block">
      <h3>Вопрос по тексту транскрипта</h3>
      <p className="muted small" style={{ marginTop: 0 }}>
        Модель отвечает только по найденным фрагментам записи и должна опираться на формулировки спикера, без
        додуманного «конспекта курса». Задайте конкретный вопрос (что сказано про …, как определён …): при одном
        общем слове в выдачу попадает меньше контекста. Первый запрос после обработки может быть дольше обычного.
      </p>
      <div className="input-group">
        <label htmlFor={`qa-q-${taskId}`}>Ваш вопрос по записи</label>
        <textarea
          id={`qa-q-${taskId}`}
          className="input-textarea"
          rows={3}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading}
          placeholder='Например: что лектор сказал про условное распределение X при известном Y?'
        />
      </div>
      <button
        type="button"
        className="btn"
        disabled={!question.trim() || loading}
        onClick={() => void submit()}
      >
        {loading ? 'Ответ…' : 'Спросить'}
      </button>
      {error && <p className="error">{error}</p>}
      {answer != null && (
        <div className="qa-answer">
          <h4>Ответ</h4>
          <LlmMarkdown text={answer} />
        </div>
      )}
    </div>
  );
}
