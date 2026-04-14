import { useState, useCallback, useRef, useEffect } from 'react'
import './index.css'

const API_URL = 'http://localhost:5000'

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
}

const PROB_COLORS = {
  glioma:     'linear-gradient(90deg,#ff4f6d,#ff8c69)',
  meningioma: 'linear-gradient(90deg,#ffaa00,#ffe066)',
  pituitary:  'linear-gradient(90deg,#7c3aed,#a78bfa)',
  notumor:    'linear-gradient(90deg,#00c8a0,#00e5c8)',
}

// ── Confidence Ring ────────────────────────────────────────────────────────
function ConfidenceRing({ value }) {
  const r   = 42
  const circ = 2 * Math.PI * r
  const pct  = Math.round(value * 100)
  const dash = circ - circ * value

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
  )
}

// ── Upload Zone ────────────────────────────────────────────────────────────
function UploadZone({ onFile }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef()

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) onFile(file)
  }, [onFile])

  const handleChange = (e) => {
    const file = e.target.files?.[0]
    if (file) onFile(file)
  }

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
  )
}

// ── Result Card ────────────────────────────────────────────────────────────
function ResultCard({ result }) {
  const info       = CLASS_INFO[result.prediction] || CLASS_INFO.notumor
  const probs      = result.probabilities
  const classNames = Object.keys(probs)

  return (
    <div className="result-section">
      <div className="result-header">
        <div className="result-title">Analysis Complete</div>
        <div className={`result-badge ${result.prediction}`}>
          {info.icon} &nbsp;{result.display_label}
        </div>
      </div>

      <div className="result-grid">
        {/* Confidence ring */}
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

        {/* Probability bars */}
        <div className="sub-card">
          <div className="sub-card-title">Class Probabilities</div>
          <div className="prob-bars">
            {classNames.sort((a,b) => probs[b]-probs[a]).map((cls) => {
              const pct = (probs[cls] * 100).toFixed(1)
              const label = CLASS_INFO[cls]?.title || cls
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
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Heatmap Viewer ─────────────────────────────────────────────────────────
function HeatmapViewer({ result }) {
  return (
    <div className="main-card" style={{ marginTop: '1.5rem' }}>
      <div className="sub-card-title" style={{ marginBottom: '1.25rem', fontSize: '0.85rem' }}>
        🔬 &nbsp;GRAD-CAM HEATMAP ANALYSIS
      </div>
      <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.25rem', lineHeight: 1.6 }}>
        Grad-CAM highlights the regions in the MRI that the EfficientNetB0 model focused on when making its prediction.
        Red/yellow areas indicate the highest model attention — likely where the tumor is located.
      </p>
      <div className="heatmap-images">
        <div className="heatmap-img-wrap">
          <img src={`data:image/png;base64,${result.original_b64}`} alt="Original MRI" />
          <div className="heatmap-img-label">Original MRI</div>
        </div>
        <div className="heatmap-img-wrap">
          <img src={`data:image/png;base64,${result.overlay_b64}`} alt="Grad-CAM Overlay" />
          <div className="heatmap-img-label">Grad-CAM Overlay</div>
        </div>
        <div className="heatmap-img-wrap">
          <img src={`data:image/png;base64,${result.heatmap_b64}`} alt="Raw Heatmap" />
          <div className="heatmap-img-label">Activation Heatmap</div>
        </div>
      </div>
    </div>
  )
}

// ── Info Cards ─────────────────────────────────────────────────────────────
function InfoSection() {
  return (
    <section className="info-section app-container">
      <h2 className="section-heading">Tumor Classification Guide</h2>
      <p className="section-sub">Understanding the four categories detected by the model</p>
      <div className="info-grid">
        {Object.entries(CLASS_INFO).map(([key, info]) => (
          <div className="info-card" key={key}>
            <div className="info-card-icon">{info.icon}</div>
            <div className="info-card-title">{info.title}</div>
            <div className="info-card-desc">{info.desc}</div>
            <span
              className="info-card-tag"
              style={{ background: info.tagBg, color: info.tagColor }}
            >
              {key === 'notumor' ? 'Normal' : 'Tumor'}
            </span>
          </div>
        ))}
      </div>
    </section>
  )
}

// ── App Root ───────────────────────────────────────────────────────────────
export default function App() {
  const [file,       setFile]       = useState(null)
  const [preview,    setPreview]    = useState(null)
  const [loading,    setLoading]    = useState(false)
  const [result,     setResult]     = useState(null)
  const [error,      setError]      = useState(null)
  const [modelReady, setModelReady] = useState(false)
  const [modelMsg,   setModelMsg]   = useState('Connecting to AI backend…')
  const pollRef = useRef(null)

  // Poll /status until the model is ready
  useEffect(() => {
    const poll = async () => {
      try {
        const res  = await fetch(`${API_URL}/status`)
        const data = await res.json()
        if (data.ready) {
          setModelReady(true)
          setModelMsg('')
          clearInterval(pollRef.current)
        } else {
          setModelMsg(data.message || 'AI model is warming up…')
        }
      } catch {
        setModelMsg('Waiting for backend to start…')
      }
    }
    poll()                                          // run immediately
    pollRef.current = setInterval(poll, 3000)       // then every 3 s
    return () => clearInterval(pollRef.current)
  }, [])

  const handleFile = (f) => {
    setFile(f)
    setResult(null)
    setError(null)
    const url = URL.createObjectURL(f)
    setPreview(url)
  }

  const handleReset = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  const handlePredict = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const fd = new FormData()
      fd.append('file', file)
      const res  = await fetch(`${API_URL}/predict`, { method: 'POST', body: fd })
      const data = await res.json()
      if (res.status === 503) throw new Error(data.error || 'Model is still loading — please wait and try again.')
      if (!res.ok)            throw new Error(data.error || 'Prediction failed')
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      {/* ── Navbar ── */}
      <nav className="navbar">
        <a href="/" className="navbar-brand">
          <div className="brand-icon">🧠</div>
          <span>Brain<span className="brand-grad">Scan</span> AI</span>
        </a>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          {!modelReady && (
            <span className="model-loading-badge">
              <span className="model-loading-dot" />
              {modelMsg}
            </span>
          )}
          {modelReady && (
            <span className="model-ready-badge">✅ Model Ready</span>
          )}
          <span className="navbar-badge">Beta v1.0</span>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="hero-section app-container">
        <div className="hero-eyebrow">
          <span className="dot"></span>
          AI-Powered Diagnostics
        </div>
        <h1 className="hero-title">
          Detect Brain Tumours<br />
          <span className="grad-text">with Deep Learning</span>
        </h1>
        <p className="hero-subtitle">
          Upload an MRI scan and our ensemble CNN model (MobileNetV2 + EfficientNetB0 + Custom CNN)
          will classify the tumour type and show a Grad-CAM heatmap of the affected region.
        </p>
        <div className="hero-stats">
          {[['4', 'Tumor Classes'], ['3', 'Model Ensemble'], ['Grad-CAM', 'Heatmaps']].map(([v, l]) => (
            <div className="stat-item" key={l}>
              <div className="stat-value">{v}</div>
              <div className="stat-label">{l}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Main Card ── */}
      <section className="app-container" style={{ paddingBottom: '2rem' }}>
        <div className="main-card">

          {/* Upload or Preview */}
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
                    id="btn-analyse"
                    className="btn-primary"
                    onClick={handlePredict}
                    disabled={loading || !modelReady}
                    title={!modelReady ? modelMsg : ''}
                  >
                    {loading      ? '⏳ Analysing…'
                    : !modelReady ? '⏳ Model Loading…'
                    :               '🔍 Analyse MRI'}
                  </button>
                  <button className="btn-ghost" onClick={handleReset}>↩ Reset</button>
                </div>
              </div>
            </div>
          )}

          {/* Loading */}
          {loading && (
            <div className="loading-overlay">
              <div className="spinner" />
              <div className="loading-text">Running ensemble inference + Grad-CAM…</div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="error-banner">
              ⚠️ <span><strong>Error:</strong> {error}</span>
            </div>
          )}

          {/* Result */}
          {result && !loading && (
            <div style={{ marginTop: '2rem' }}>
              <ResultCard result={result} />
            </div>
          )}
        </div>

        {/* Heatmap below */}
        {result && !loading && <HeatmapViewer result={result} />}
      </section>

      {/* ── Info Section ── */}
      <InfoSection />

      {/* ── Footer ── */}
      <footer className="footer">
        BrainScan AI · Ensemble CNN (MobileNetV2 + EfficientNetB0 + Custom CNN) · Grad-CAM Heatmaps
      </footer>
    </>
  )
}
