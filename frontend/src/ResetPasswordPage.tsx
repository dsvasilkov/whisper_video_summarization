import { FormEvent, useMemo, useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { resetPassword } from './api';

export function ResetPasswordPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const tokenFromUrl = useMemo(() => searchParams.get('token') ?? '', [searchParams]);
  const [token, setToken] = useState(tokenFromUrl);
  const [password, setPassword] = useState('');
  const [password2, setPassword2] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [loading, setLoading] = useState(false);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    if (password !== password2) {
      setError('Пароли не совпадают');
      return;
    }
    if (!token.trim()) {
      setError('Нужен токен из ссылки или письма');
      return;
    }
    setLoading(true);
    try {
      await resetPassword(token.trim(), password);
      setDone(true);
      setTimeout(() => navigate('/login', { replace: true }), 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="card auth-card">
        <h2>Новый пароль</h2>
        {done ? (
          <p className="success-msg">Пароль изменён. Перенаправление на вход…</p>
        ) : (
          <form onSubmit={onSubmit}>
            <div className="input-group">
              <label htmlFor="reset-token">Токен сброса</label>
              <input
                id="reset-token"
                type="text"
                autoComplete="off"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="Из ссылки ?token=…"
                required
              />
            </div>
            <div className="input-group">
              <label htmlFor="reset-pass">Новый пароль</label>
              <input
                id="reset-pass"
                type="password"
                autoComplete="new-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                minLength={8}
                required
              />
            </div>
            <div className="input-group">
              <label htmlFor="reset-pass2">Повтор пароля</label>
              <input
                id="reset-pass2"
                type="password"
                autoComplete="new-password"
                value={password2}
                onChange={(e) => setPassword2(e.target.value)}
                minLength={8}
                required
              />
            </div>
            {error && <p className="error">{error}</p>}
            <button type="submit" className="btn" disabled={loading}>
              {loading ? 'Сохранение…' : 'Сохранить пароль'}
            </button>
          </form>
        )}
        <p className="auth-links">
          <Link to="/login">Вход</Link>
        </p>
      </div>
    </div>
  );
}
