import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import Login from './pages/Login';
import DashboardLayout from './pages/DashboardLayout';
import Overview from './pages/Overview';
import Patients from './pages/Patients';
import TumorDetection from './pages/TumorDetection';
import Reports from './pages/Reports';
import History from './pages/History';
import './index.css';
import './App.css'; // Adding app specific styles if any

// Component to protect routes
const ProtectedRoute = ({ children }) => {
  const { token } = useAuth();
  if (!token) {
    return <Navigate to="/" replace />;
  }
  return children;
};

export default function App() {
  return (
    <Router>
      <AuthProvider>
        <Routes>
          {/* Public Login Route */}
          <Route path="/" element={<Login />} />

          {/* Protected Dashboard Routes */}
          <Route 
            path="/dashboard" 
            element={
              <ProtectedRoute>
                <DashboardLayout />
              </ProtectedRoute>
            }
          >
            <Route index element={<Overview />} />
            <Route path="patients" element={<Patients />} />
            <Route path="detection" element={<TumorDetection />} />
            <Route path="reports" element={<Reports />} />
            <Route path="history" element={<History />} />
          </Route>

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>
    </Router>
  );
}
