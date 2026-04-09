import { FormEvent, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { forgotPassword } from './api';

export function ForgotPasswordPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState<string | null>(null);
  const [devToken, setDevToken] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setMessage(null);
    setDevToken(null);
    setLoading(true);
    try {
      const res = await forgotPassword(email);
      setMessage(res.message);
      if (res.reset_token) setDevToken(res.reset_token);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const goReset = (token: string) => {
    navigate(`/reset-password?token=${encodeURIComponent(token)}`);
  };

  return (
    <div className="auth-page">
      <div className="card auth-card">
        <h2>Сброс пароля</h2>
        <p className="auth-hint">
          Укажите почту аккаунта. После отправки при необходимости используйте ссылку из письма; при{' '}
          <code className="inline-code">DEBUG=1</code> на сервере токен сброса может появиться ниже.
        </p>
        <form onSubmit={onSubmit}>
          <div className="input-group">
            <label htmlFor="forgot-email">Почта</label>
            <input
              id="forgot-email"
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
          {error && <p className="error">{error}</p>}
          {message && <p className="success-msg">{message}</p>}
          <button type="submit" className="btn" disabled={loading}>
            {loading ? 'Отправка…' : 'Отправить'}
          </button>
        </form>
        {devToken && (
          <div className="dev-token-box">
            <p>DEBUG: токен сброса (не для продакшена)</p>
            <button type="button" className="btn btn-secondary" onClick={() => goReset(devToken)}>
              Перейти к смене пароля
            </button>
          </div>
        )}
        <p className="auth-links">
          <Link to="/login">Назад ко входу</Link>
        </p>
      </div>
    </div>
  );
}
