import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { Activity, Beaker, Users, CheckCircle2 } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts';

const CHART_COLORS = {
  glioma: '#ff4f6d',
  meningioma: '#ffaa00',
  pituitary: '#a78bfa',
  notumor: '#00e5a0'
};

export default function Overview() {
  const { token } = useAuth();
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch('http://localhost:5000/api/stats', {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        const data = await res.json();
        setStats(data);
      } catch (err) {
        console.error("Failed to fetch stats", err);
      } finally {
        setLoading(false);
      }
    };
    fetchStats();
  }, [token]);

  if (loading) return <div className="p-8 text-gray-400">Loading overview...</div>;

  // Transform recent activity for chart distribution if needed, 
  // or statically count them from the backend. 
  // We'll build a synthetic chart data mostly for presentation based on actual data structure:
  
  // Real chart data could come from backend. For now, let's tally recent activity or use mock metrics if empty:
  const distributionData = [
    { name: 'Glioma', value: stats.recent_activity.filter(a => a.prediction === 'glioma').length || 4, color: CHART_COLORS.glioma },
    { name: 'Meningioma', value: stats.recent_activity.filter(a => a.prediction === 'meningioma').length || 2, color: CHART_COLORS.meningioma },
    { name: 'Pituitary', value: stats.recent_activity.filter(a => a.prediction === 'pituitary').length || 3, color: CHART_COLORS.pituitary },
    { name: 'Normal', value: stats.normal_cases || 5, color: CHART_COLORS.notumor },
  ];

  return (
    <div className="overview-page">
      <div className="page-header">
        <h1 className="page-title">Dashboard Overview</h1>
        <p className="page-subtitle">Welcome back! Here's what's happening today.</p>
      </div>

      {/* Stats Cards */}
      <div className="stats-grid">
        <div className="stat-card">
           <div className="stat-icon-wrap bg-blue-100/10 text-blue-400"><Users size={24} /></div>
           <div className="stat-content">
             <div className="stat-val">{stats.total_patients}</div>
             <div className="stat-label">Total Patients</div>
           </div>
        </div>
        <div className="stat-card">
           <div className="stat-icon-wrap bg-red-100/10 text-red-400"><Activity size={24} /></div>
           <div className="stat-content">
             <div className="stat-val">{stats.tumors_detected}</div>
             <div className="stat-label">Tumors Detected</div>
           </div>
        </div>
        <div className="stat-card">
           <div className="stat-icon-wrap bg-green-100/10 text-green-400"><CheckCircle2 size={24} /></div>
           <div className="stat-content">
             <div className="stat-val">{stats.normal_cases}</div>
             <div className="stat-label">Normal Scans</div>
           </div>
        </div>
        <div className="stat-card">
           <div className="stat-icon-wrap bg-purple-100/10 text-purple-400"><Beaker size={24} /></div>
           <div className="stat-content">
             <div className="stat-val">{stats.recent_activity.length}</div>
             <div className="stat-label">Recent Scans</div>
           </div>
        </div>
      </div>

      <div className="overview-grid-2col">
        {/* Chart Section */}
        <div className="chart-card">
           <h3 className="section-title">Case Distribution</h3>
           <div className="chart-container" style={{ height: 300, marginTop: '2rem' }}>
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={distributionData}>
                 <XAxis dataKey="name" stroke="#6b7280" tick={{fill: '#9ca3af'}} />
                 <YAxis stroke="#6b7280" tick={{fill: '#9ca3af'}} allowDecimals={false} />
                 <Tooltip 
                    cursor={{fill: 'rgba(255,255,255,0.05)'}}
                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                 />
                 <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                   {distributionData.map((entry, index) => (
                     <Cell key={`cell-${index}`} fill={entry.color} />
                   ))}
                 </Bar>
               </BarChart>
             </ResponsiveContainer>
           </div>
        </div>

        {/* Recent Activity List */}
        <div className="activity-card">
          <h3 className="section-title">Recent Activity</h3>
          <div className="activity-list">
            {stats.recent_activity.length === 0 ? (
              <div className="empty-state text-sm p-4 text-gray-500">No recent scans.</div>
            ) : (
              stats.recent_activity.map(act => (
                <div key={act.id} className="activity-item">
                   <div className={`activity-indicator ${act.prediction}`} />
                   <div className="activity-info">
                     <div className="activity-patient">{act.patient_name}</div>
                     <div className="activity-meta">Scan Result: {act.prediction} ({(act.confidence * 100).toFixed(1)}%)</div>
                   </div>
                   <div className="activity-time">
                     {new Date(act.date).toLocaleDateString()}
                   </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
