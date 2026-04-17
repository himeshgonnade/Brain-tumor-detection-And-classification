import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { BrainCircuit, Lock, Mail, ActivitySquare, AlertCircle } from 'lucide-react';

export default function Login() {
  const [email, setEmail] = useState('doctor@hospital.com');
  const [password, setPassword] = useState('password123');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const res = await fetch('http://localhost:5000/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.message || 'Login failed');
      
      login(data.token);
      navigate('/dashboard');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-panel">
        <div className="brand-icon-large">
          <BrainCircuit size={48} className="lucide-icon text-indigo" />
        </div>
        <h2 className="login-title">Doctor Portal</h2>
        <p className="login-subtitle">Sign in to access patient records and AI diagnostics</p>

        {error && (
          <div className="error-banner">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="login-form">
          <div className="input-group">
            <label>Email Address</label>
            <div className="input-wrapper">
              <Mail size={18} className="input-icon" />
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="doctor@hospital.com"
                required
              />
            </div>
          </div>

          <div className="input-group">
            <label>Password</label>
            <div className="input-wrapper">
              <Lock size={18} className="input-icon" />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                required
              />
            </div>
          </div>

          <div className="remember-me">
            <input type="checkbox" id="remember" />
            <label htmlFor="remember">Remember me for 30 days</label>
          </div>

          <button type="submit" className="btn-primary login-btn" disabled={loading}>
            {loading ? 'Authenticating...' : 'Secure Sign In'}
          </button>
        </form>
        
        <div className="demo-hint text-center mt-6 text-sm text-gray-400">
          Demo Credentials: doctor@hospital.com / password123
        </div>
      </div>
      
      <div className="login-visual text-center">
         <ActivitySquare size={120} className="visual-icon text-teal opacity-20" />
         <h1 className="hero-title mt-4 text-white">
          BrainScan <span className="grad-text">AI Platform</span>
         </h1>
         <p className="text-gray-300 max-w-md mx-auto mt-4 px-6 text-sm">
           Empowering healthcare professionals with state-of-the-art ensemble deep learning for robust MRI classification.
         </p>
      </div>
    </div>
  );
}
