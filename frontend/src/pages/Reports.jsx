import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { Download, FileText, Brain } from 'lucide-react';

export default function Reports() {
  const { token } = useAuth();
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchReports = async () => {
      try {
        const res = await fetch('http://localhost:5000/api/reports', {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        const data = await res.json();
        setReports(data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchReports();
  }, [token]);

  const handleDownload = (report) => {
    // Generate a simple PDF or text file for download
    const content = `BrainScan AI - Medical Report
--------------------------------------
Patient Name: ${report.patient_name}
Scan Date: ${new Date(report.date).toLocaleString()}
Report ID: ${report.id}

AI Diagnostic Result
--------------------------------------
Prediction: ${report.prediction.toUpperCase()}
Confidence: ${(report.confidence * 100).toFixed(2)}%

Note: This is an AI generated summary. Please review with an oncologist.
`;
    
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Report_${report.patient_name.replace(' ', '_')}_${report.id}.txt`;
    a.click();
  };

  return (
    <div className="reports-page">
      <div className="page-header">
        <h1 className="page-title">Diagnostic Reports</h1>
        <p className="page-subtitle">History of previous MRI scans and AI analyses.</p>
      </div>

      <div className="reports-grid mt-6">
        {loading ? (
          <div className="text-gray-400 p-8">Loading reports...</div>
        ) : reports.length === 0 ? (
          <div className="text-gray-400 p-8 border border-dashed border-gray-700 rounded-lg">No reports found.</div>
        ) : (
          reports.map(report => (
            <div key={report.id} className="report-card">
              <div className="report-header">
                <div className="flex items-center gap-3">
                  <div className={`report-icon-wrap ${report.prediction}`}>
                    <Brain size={24} />
                  </div>
                  <div>
                    <div className="report-patient">{report.patient_name}</div>
                    <div className="report-date">{new Date(report.date).toLocaleDateString()}</div>
                  </div>
                </div>
              </div>
              <div className="report-body">
                <div className="report-stat">
                  <span className="stat-label">Prediction</span>
                  <span className={`stat-value ${report.prediction}`}>{report.prediction.toUpperCase()}</span>
                </div>
                <div className="report-stat">
                  <span className="stat-label">Confidence</span>
                  <span className="stat-value">{(report.confidence * 100).toFixed(1)}%</span>
                </div>
              </div>
              <div className="report-footer">
                 <button className="btn-ghost flex-center gap-2 w-full justify-center" onClick={() => handleDownload(report)}>
                    <Download size={16} /> Download Report
                 </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
