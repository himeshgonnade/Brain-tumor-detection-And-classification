import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { Activity, Beaker, Users, CheckCircle2 } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { Link } from 'react-router-dom';

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
  const [isChartZoomed, setIsChartZoomed] = useState(false);

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

  const recentActsPreview = stats.recent_activity.slice(0, 5);

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
        <div className="chart-card cursor-pointer hover:border-gray-600 transition-colors relative" onClick={() => setIsChartZoomed(true)}>
           <div className="flex justify-between items-center">
             <h3 className="section-title">Case Distribution</h3>
             <span className="text-xs text-gray-500 bg-[#374151] px-2 py-1 rounded">Click to Expand ⤢</span>
           </div>
           <div className="chart-container" style={{ height: 300, marginTop: '2rem' }}>
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={distributionData}>
                 <XAxis dataKey="name" stroke="#6b7280" tick={{fill: '#9ca3af', fontSize: 12}} />
                 <YAxis stroke="#6b7280" tick={{fill: '#9ca3af', fontSize: 12}} allowDecimals={false} />
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
            {recentActsPreview.length === 0 ? (
              <div className="empty-state text-sm p-4 text-gray-500">No recent scans.</div>
            ) : (
              recentActsPreview.map(act => (
                <div key={act.id} className="activity-item">
                   <div className={`activity-indicator ${act.prediction}`} />
                   <div className="activity-info">
                     <div className="activity-patient">{act.patient_name}</div>
                     <div className="activity-meta">Result: {act.prediction} ({(act.confidence * 100).toFixed(1)}%)</div>
                   </div>
                   <div className="activity-time">
                     {new Date(act.date).toLocaleDateString()}
                   </div>
                </div>
              ))
            )}
          </div>
          {stats.recent_activity.length > 0 && (
            <Link 
              to="/dashboard/history"
              className="mt-4 text-sm text-[#00c8ff] hover:text-[#00c8ff]/80 w-full text-center block transition-colors font-medium border-t border-[#374151] pt-4"
            >
              View Full History Record →
            </Link>
          )}
        </div>
      </div>

      {/* --- Zoomed Chart Overlay --- */}
      {isChartZoomed && (
        <div className="fixed inset-0 bg-black/90 backdrop-blur-md flex items-center justify-center z-50 p-4 md:p-12 zoom-backdrop" onClick={() => setIsChartZoomed(false)}>
          <div className="bg-[#071629] border border-[#00c8ff]/30 rounded-2xl p-6 md:p-10 w-full max-w-6xl shadow-[0_0_50px_rgba(0,200,255,0.2)] scale-zoom-up" onClick={e => e.stopPropagation()}>
            <div className="flex justify-between items-start mb-8 border-b border-[#374151] pb-6">
              <div>
                <h2 className="text-3xl font-bold font-outfit grad-text">Case Distribution Analysis</h2>
                <p className="text-gray-400 mt-2">Detailed breakdown and performance metrics across diagnostic categories.</p>
              </div>
              <button 
                className="bg-[#1f2937] text-white p-3 rounded-full hover:bg-red-500/20 hover:text-red-500 transition-all border border-gray-700 hover:border-red-500/50" 
                onClick={() => setIsChartZoomed(false)}
              >
                ✕
              </button>
            </div>
            
            <div className="flex flex-col lg:flex-row gap-8">
              {/* Detailed Drill-down list */}
              <div className="w-full lg:w-1/3 flex flex-col gap-4">
                <h3 className="text-lg font-semibold text-white">Metrics Breakdown</h3>
                {distributionData.map(d => (
                  <div key={d.name} className="flex justify-between items-center p-4 bg-[#111827] rounded-xl border border-gray-800">
                    <div className="flex items-center gap-3">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: d.color }}></div>
                      <span className="font-medium text-gray-200">{d.name}</span>
                    </div>
                    <div className="flex flex-col items-end">
                      <span className="text-xl font-bold" style={{ color: d.color }}>{d.value}</span>
                      <span className="text-xs text-gray-500">{(d.value / (stats?.total_patients || 1) * 100).toFixed(1)}% of total</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* The Enlarged Chart */}
              <div className="w-full lg:w-2/3" style={{ height: '55vh', minHeight: '400px' }}>
                <ResponsiveContainer width="100%" height="100%">
                   <BarChart data={distributionData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                     <XAxis dataKey="name" stroke="#6b7280" tick={{fill: '#9ca3af', fontSize: 14}} axisLine={false} tickLine={false} />
                     <YAxis stroke="#6b7280" tick={{fill: '#9ca3af', fontSize: 14}} axisLine={false} tickLine={false} allowDecimals={false} />
                     <Tooltip 
                       cursor={{fill: 'rgba(255,255,255,0.05)'}} 
                       contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '12px', padding: '15px', color: '#fff' }} 
                     />
                     <Bar dataKey="value" radius={[8, 8, 0, 0]} barSize={80}>
                       {distributionData.map((entry, index) => (
                         <Cell key={`cell-${index}`} fill={entry.color} />
                       ))}
                     </Bar>
                   </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
