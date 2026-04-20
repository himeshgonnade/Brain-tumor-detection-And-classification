import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { Search, Download } from 'lucide-react';
import { generatePDFReport } from '../utils/pdfGenerator';

export default function History() {
  const { token } = useAuth();
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    const fetchReports = async () => {
      try {
        const res = await fetch('http://localhost:5000/api/reports', {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        const data = await res.json();
        setReports(data);
      } catch (err) {
        console.error("Failed to fetch history", err);
      } finally {
        setLoading(false);
      }
    };
    fetchReports();
  }, [token]);

  const filteredReports = reports.filter(r => 
    r.patient_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    r.prediction.toLowerCase().includes(searchTerm.toLowerCase()) ||
    r.id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleDownload = (report) => {
    generatePDFReport(report, report.patient_name);
  };

  if (loading) return <div className="p-8 text-gray-400">Loading history...</div>;

  return (
    <div className="history-page">
      <div className="page-header">
        <h1 className="page-title">Activity History</h1>
        <p className="page-subtitle">Full record of all MRI scans and AI diagnostics.</p>
      </div>

      <div className="history-filters mt-6">
        <div className="search-box">
          <Search size={18} />
          <input 
            type="text" 
            placeholder="Search by patient name, result, or ID..." 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>

      <div className="history-container mt-6">
        <div className="history-table-wrap">
          <table className="history-table">
            <thead>
              <tr>
                <th>Date & Time</th>
                <th>Patient Name</th>
                <th>Scan Result</th>
                <th>Confidence</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredReports.length === 0 ? (
                <tr>
                  <td colSpan="5" className="text-center py-8 text-gray-500">No matching records found.</td>
                </tr>
              ) : (
                filteredReports.map(report => (
                  <tr key={report.id}>
                    <td>
                      <div className="flex flex-col">
                        <span className="text-white font-medium">{new Date(report.date).toLocaleDateString()}</span>
                        <span className="text-xs text-gray-500">{new Date(report.date).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                      </div>
                    </td>
                    <td>
                      <div className="patient-id-cell">
                        <span className="text-white">{report.patient_name}</span>
                        <span className="text-xs text-gray-500">ID: #{report.id}</span>
                      </div>
                    </td>
                    <td>
                      <span className={`status-badge ${report.prediction}`}>
                        {report.prediction.toUpperCase()}
                      </span>
                    </td>
                    <td>
                      <div className="confidence-cell">
                        <div className="confidence-bar-mini">
                          <div className="fill" style={{ width: `${(report.confidence * 100)}%` }}></div>
                        </div>
                        <span>{(report.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </td>
                    <td>
                      <div className="table-actions">
                        <button 
                          className="action-btn download" 
                          onClick={() => handleDownload(report)}
                          title="Download PDF"
                        >
                          <Download size={16} />
                        </button>
                        {/* More actions like View Detail or Delete could go here */}
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
