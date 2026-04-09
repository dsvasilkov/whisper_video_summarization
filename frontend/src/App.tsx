import type { ReactNode } from 'react'
import { Navigate, Outlet, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { InferencePage } from './InferencePage'
import { ForgotPasswordPage } from './ForgotPasswordPage'
import { HistoryPage } from './HistoryPage'
import { LoginPage } from './LoginPage'
import { RegisterPage } from './RegisterPage'
import { ResetPasswordPage } from './ResetPasswordPage'
import { useAuth } from './AuthContext'
import './App.css'

function ProtectedRoute({ children }: { children: ReactNode }) {
  const { token } = useAuth()
  const loc = useLocation()
  if (!token) {
    return <Navigate to="/login" replace state={{ from: loc.pathname }} />
  }
  return <>{children}</>
}

function MainLayout() {
  const { logout } = useAuth()
  const navigate = useNavigate()
  const loc = useLocation()

  return (
    <div className="app">
      <header className="header">
        <h1>Whisper Video Summarization</h1>
        <nav className="nav">
          <button
            type="button"
            className={loc.pathname === '/' ? 'active' : ''}
            onClick={() => navigate('/')}
          >
            Инференс
          </button>
          <button
            type="button"
            className={loc.pathname === '/history' ? 'active' : ''}
            onClick={() => navigate('/history')}
          >
            История
          </button>
          <button type="button" className="nav-ghost" onClick={() => logout()}>
            Выйти
          </button>
        </nav>
      </header>
      <main className="main">
        <Outlet />
      </main>
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />
      <Route path="/forgot-password" element={<ForgotPasswordPage />} />
      <Route path="/reset-password" element={<ResetPasswordPage />} />
      <Route
        element={
          <ProtectedRoute>
            <MainLayout />
          </ProtectedRoute>
        }
      >
        <Route path="/" element={<InferencePage />} />
        <Route path="/history" element={<HistoryPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
