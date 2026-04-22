import { useState, useCallback, useRef, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { generatePDFReport } from '../utils/pdfGenerator';
import { Download } from 'lucide-react';

const API_URL = 'http://localhost:5000';

const CLASS_INFO = {
  glioma: {
    icon: '🧠', color: '#ff4f6d', tagBg: 'rgba(255,79,109,0.15)', tagColor: '#ff4f6d',
    title: 'Glioma',
    desc: 'A type of tumor that occurs in the brain and spinal cord, originating from glial cells.',
  },
  meningioma: {
    icon: '🩺', color: '#ffaa00', tagBg: 'rgba(255,168,0,0.15)', tagColor: '#ffaa00',
    title: 'Meningioma',
    desc: 'A tumor that arises from the meninges, the membranes covering the brain and spinal cord.',
  },
  pituitary: {
    icon: '💊', color: '#a78bfa', tagBg: 'rgba(124,58,237,0.15)', tagColor: '#a78bfa',
    title: 'Pituitary Tumor',
    desc: 'Abnormal growth in the pituitary gland, often affecting hormonal regulation.',
  },
  notumor: {
    icon: '✅', color: '#00e5a0', tagBg: 'rgba(0,229,160,0.15)', tagColor: '#00e5a0',
    title: 'No Tumor',
    desc: 'The MRI scan appears normal. No evidence of a brain tumor was detected.',
  },
};

const PROB_COLORS = {
  glioma:     'linear-gradient(90deg,#ff4f6d,#ff8c69)',
  meningioma: 'linear-gradient(90deg,#ffaa00,#ffe066)',
  pituitary:  'linear-gradient(90deg,#7c3aed,#a78bfa)',
  notumor:    'linear-gradient(90deg,#00c8a0,#00e5c8)',
};

// ── Confidence Ring ──
function ConfidenceRing({ value }) {
  const r   = 42;
  const circ = 2 * Math.PI * r;
  const pct  = Math.round(value * 100);
  const dash = circ - circ * value;

  return (
    <svg className="confidence-ring-svg" width="110" height="110" viewBox="0 0 100 100">
      <defs>
        <linearGradient id="conf-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%"   stopColor="#00c8ff" />
          <stop offset="100%" stopColor="#0077ff" />
        </linearGradient>
      </defs>
      <circle className="confidence-ring-bg" cx="50" cy="50" r={r} strokeWidth="8" />
      <circle
        className="confidence-ring-fg"
        cx="50" cy="50" r={r}
        strokeWidth="8"
        strokeDasharray={circ}
        strokeDashoffset={dash}
        transform="rotate(-90 50 50)"
      />
      <text className="confidence-center" x="50" y="46">
        <tspan className="confidence-pct" x="50" dy="0">{pct}%</tspan>
        <tspan className="confidence-lbl" x="50" dy="14">Confidence</tspan>
      </text>
    </svg>
  );
}

// ── Upload Zone ──
function UploadZone({ onFile }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef();

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) onFile(file);
  }, [onFile]);

  const handleChange = (e) => {
    const file = e.target.files?.[0];
    if (file) onFile(file);
  };

  return (
    <div
      id="upload-zone"
      className={`upload-zone ${dragging ? 'drag-active' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current.click()}
    >
      <div className="upload-icon">🧬</div>
      <div className="upload-title">Drop MRI Image Here</div>
      <div className="upload-subtitle">or click to browse your files</div>
      <button className="upload-btn" onClick={(e) => { e.stopPropagation(); inputRef.current.click() }}>
        📂 &nbsp;Choose File
      </button>
      <div className="upload-formats">Supported formats: JPG · JPEG · PNG · BMP · WEBP</div>
      <input ref={inputRef} type="file" accept="image/*" onChange={handleChange} hidden id="file-input" />
    </div>
  );
}

// ── Result Card ──
function ResultCard({ result }) {
  const info       = CLASS_INFO[result.prediction] || CLASS_INFO.notumor;
  const probs      = result.probabilities;
  const classNames = Object.keys(probs);

  return (
    <div className="result-section">
      <div className="result-header">
        <div className="result-title">Analysis Complete</div>
        <div className={`result-badge ${result.prediction}`}>
          {info.icon} &nbsp;{result.display_label}
        </div>
      </div>

      <div className="result-grid">
        <div className="sub-card">
          <div className="sub-card-title">Detection Confidence</div>
          <div className="confidence-wrapper">
            <ConfidenceRing value={result.confidence} />
            <div>
              <div className="confidence-val">{Math.round(result.confidence * 100)}%</div>
              <div className="confidence-label">
                Detected: <strong style={{ color: info.color }}>{result.display_label}</strong>
              </div>
              <div className="confidence-label" style={{ marginTop: '0.5rem', fontSize: '0.8rem' }}>
                {info.desc}
              </div>
            </div>
          </div>
        </div>

        <div className="sub-card">
          <div className="sub-card-title">Class Probabilities</div>
          <div className="prob-bars">
            {classNames.sort((a,b) => probs[b]-probs[a]).map((cls) => {
              const pct = (probs[cls] * 100).toFixed(1);
              const label = CLASS_INFO[cls]?.title || cls;
              return (
                <div className="prob-row" key={cls}>
                  <div className="prob-row-header">
                    <span className="prob-name">{label}</span>
                    <span className="prob-value">{pct}%</span>
                  </div>
                  <div className="prob-track">
                    <div
                      className="prob-fill"
                      style={{ width: `${pct}%`, background: PROB_COLORS[cls] || '#00c8ff' }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Heatmap Viewer ── (only shown when a tumor is detected)
function HeatmapViewer({ result }) {
  // Don't render at all for normal / no-tumor results
  if (result.prediction === 'notumor') return null;

  return (
    <div className="main-card mt-6">
      <div className="sub-card-title mb-4 text-sm">
        🔬 &nbsp;GRAD-CAM HEATMAP ANALYSIS
      </div>
      <p className="text-sm text-gray-400 mb-6 leading-relaxed">
        Grad-CAM++ highlights the regions in the MRI that the AI model attended to when predicting
        <strong style={{ color: '#f97316' }}> {result.display_label}</strong>.
        Red / yellow areas indicate the highest activation — likely where the tumor is located.
        Blue areas indicate low attention.
      </p>
      <div className="heatmap-images">
        <div className="heatmap-img-wrap">
          <img src={`data:image/png;base64,${result.original_b64}`} alt="Original MRI" />
          <div className="heatmap-img-label">Original MRI</div>
        </div>
        <div className="heatmap-img-wrap">
          <img src={`data:image/png;base64,${result.overlay_b64}`} alt="Grad-CAM Overlay" />
          <div className="heatmap-img-label">Grad-CAM++ Overlay</div>
        </div>
        <div className="heatmap-img-wrap">
          <img src={`data:image/png;base64,${result.heatmap_b64}`} alt="Raw Heatmap" />
          <div className="heatmap-img-label">Activation Heatmap</div>
        </div>
      </div>
    </div>
  );
}

export default function TumorDetection() {
  const location = useLocation();
  const { token } = useAuth();
  
  // patientId passed from navigation state if available
  const [patientId, setPatientId] = useState(location.state?.patientId || '');
  const [patients, setPatients] = useState([]);
  
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [modelReady, setModelReady] = useState(false);
  const [modelMsg, setModelMsg] = useState('Connecting to AI backend…');
  const [reportSaved, setReportSaved] = useState(false);
  const pollRef = useRef(null);

  // Fetch patients for the dropdown
  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const res = await fetch(`${API_URL}/api/patients`, {
          headers: { 'Authorization': `Bearer ${token}` }
        });
        const data = await res.json();
        setPatients(data);
      } catch (err) { }
    }
    fetchPatients();
  }, [token]);

  // Poll
  useEffect(() => {
    const poll = async () => {
      try {
        const res  = await fetch(`${API_URL}/status`);
        const data = await res.json();
        if (data.ready) {
          setModelReady(true);
          setModelMsg('');
          clearInterval(pollRef.current);
        } else {
          setModelMsg(data.message || 'AI model is warming up…');
        }
      } catch {
        setModelMsg('Waiting for backend to start…');
      }
    };
    poll();
    pollRef.current = setInterval(poll, 3000);
    return () => clearInterval(pollRef.current);
  }, []);

  const handleFile = (f) => {
    setFile(f);
    setResult(null);
    setError(null);
    setReportSaved(false);
    const url = URL.createObjectURL(f);
    setPreview(url);
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setReportSaved(false);
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const fd = new FormData();
      fd.append('file', file);
      const res  = await fetch(`${API_URL}/predict`, { method: 'POST', body: fd });
      const data = await res.json();
      if (res.status === 503) throw new Error(data.error || 'Model is loading.');
      if (!res.ok)            throw new Error(data.error || 'Prediction failed');
      
      setResult(data);
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const saveReport = async () => {
    try {
      const res = await fetch(`${API_URL}/api/reports`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          patient_id: patientId,
          prediction: result.prediction,
          confidence: result.confidence,
          heatmap_b64: result.heatmap_b64,
          overlay_b64: result.overlay_b64,
          original_b64: result.original_b64
        })
      });
      if(res.ok) {
        setReportSaved(true);
      }
    } catch (err) {
      alert("Failed to save report: " + err.message);
    }
  }

  return (
    <div className="detection-page w-full">
       <div className="page-header">
         <h1 className="page-title">AI Tumor Detection</h1>
         <p className="page-subtitle">Upload MRI scan for automated AI analysis</p>
         
         {!modelReady && (
            <span className="model-loading-badge mt-4 inline-flex">
              <span className="model-loading-dot" />
              {modelMsg}
            </span>
          )}
          {modelReady && (
            <span className="model-ready-badge mt-4 inline-flex">✅ Model Ready</span>
          )}
       </div>

       <div className="patient-selector mt-6 mb-6">
         <label className="block text-sm text-gray-400 mb-2">Assign to Patient (Optional)</label>
         <select 
           className="modal-input" 
           value={patientId} 
           onChange={(e) => setPatientId(e.target.value)}
         >
           <option value="">-- Select Patient --</option>
           {patients.map(p => (
             <option key={p.id} value={p.id}>{p.name} (#{p.id})</option>
           ))}
         </select>
       </div>

       <div className="main-card">
          {!file ? (
            <UploadZone onFile={handleFile} />
          ) : (
            <div className="image-preview-wrap">
              <img className="preview-thumbnail" src={preview} alt="preview" />
              <div className="preview-info">
                <div className="preview-filename">{file.name}</div>
                <div className="preview-meta">{(file.size / 1024).toFixed(1)} KB · {file.type}</div>
                <div className="preview-actions">
                  <button
                    className="btn-primary"
                    onClick={handlePredict}
                    disabled={loading || !modelReady}
                    title={!modelReady ? modelMsg : ''}
                  >
                    {loading ? '⏳ Analysing…' : !modelReady ? '⏳ Model Loading…' : '🔍 Analyse MRI'}
                  </button>
                  <button className="btn-ghost" onClick={handleReset}>↩ Reset</button>
                </div>
              </div>
            </div>
          )}

          {loading && (
            <div className="loading-overlay">
              <div className="spinner" />
              <div className="loading-text">Running ensemble inference + Grad-CAM…</div>
            </div>
          )}

          {error && (
            <div className="error-banner">
              ⚠️ <span><strong>Error:</strong> {error}</span>
            </div>
          )}

          {result && !loading && (
            <div className="mt-8">
              <ResultCard result={result} />
              
              {patientId && !reportSaved && (
                 <div className="mt-4 p-4 bg-[#1f2937] border border-[#374151] rounded-lg flex justify-between items-center">
                   <span className="text-gray-300">Save these results to the patient's record.</span>
                   <button onClick={saveReport} className="btn-primary text-sm py-2">Save Report</button>
                 </div>
              )}
              {reportSaved && (
                 <div className="mt-4 p-4 bg-green-900/40 border border-green-500/50 rounded-lg text-green-400">
                    ✅ Report successfully saved to patient record!
                 </div>
              )}
              
              <div className="mt-4 flex justify-end">
                <button 
                  onClick={() => {
                     const selectedPatient = patients.find((p) => p.id === patientId);
                     generatePDFReport(result, selectedPatient ? selectedPatient.name : "Unknown Patient");
                  }} 
                  className="btn-ghost flex items-center gap-2 text-sm"
                >
                  <Download size={16} /> Download PDF Report
                </button>
              </div>
            </div>
          )}
        </div>

        {result && !loading && result.prediction !== 'notumor' && <HeatmapViewer result={result} />}
    </div>
  );
}
