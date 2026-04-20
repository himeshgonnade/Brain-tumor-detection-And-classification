import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { BrainCircuit, LayoutDashboard, Users, FileBarChart, LogOut, ScanLine, History } from 'lucide-react';

export default function DashboardLayout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <div className="dashboard-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <BrainCircuit size={28} className="text-indigo" />
          <span>Brain<span className="grad-text">Scan</span></span>
        </div>

        <nav className="sidebar-nav">
          <NavLink to="/dashboard" end className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <LayoutDashboard size={20} />
            Overview
          </NavLink>
          <NavLink to="/dashboard/patients" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <Users size={20} />
            Patients
          </NavLink>
          <NavLink to="/dashboard/detection" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <ScanLine size={20} />
            AI Detection
          </NavLink>
          <NavLink to="/dashboard/reports" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <FileBarChart size={20} />
            Reports
          </NavLink>
          <NavLink to="/dashboard/history" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
            <History size={20} />
            Activity History
          </NavLink>
        </nav>

        <div className="sidebar-footer">
          <div className="user-profile">
            <div className="avatar">{user?.name?.charAt(0) || 'D'}</div>
            <div className="user-info">
              <div className="user-name">{user?.name || 'Doctor'}</div>
              <div className="user-role">Oncology Dept</div>
            </div>
          </div>
          <button onClick={handleLogout} className="logout-btn">
            <LogOut size={18} />
            Logout
          </button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="dashboard-main">
        <header className="topbar">
          <div className="topbar-search">
             {/* Optional search or breadcrumbs could go here */}
          </div>
          <div className="topbar-actions">
            <span className="system-status">
              <span className="status-dot"></span> System Online
            </span>
          </div>
        </header>
        
        <div className="dashboard-content scroll-custom">
           <Outlet />
        </div>
      </main>
    </div>
  );
}
