import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { UserPlus, Search, FileText } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export default function Patients() {
  const { token } = useAuth();
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showAddModal, setShowAddModal] = useState(false);
  const [newPatient, setNewPatient] = useState({ name: '', age: '' });
  const navigate = useNavigate();

  const fetchPatients = async () => {
    try {
      const res = await fetch('http://localhost:5000/api/patients', {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await res.json();
      setPatients(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPatients();
  }, [token]);

  const handleAddPatient = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch('http://localhost:5000/api/patients', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ name: newPatient.name, age: parseInt(newPatient.age) })
      });
      if (res.ok) {
        setNewPatient({ name: '', age: '' });
        setShowAddModal(false);
        fetchPatients();
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleScanPatient = (patientId) => {
    // Navigate to detection page and perhaps pass patient ID. 
    // We can use navigate state to do this
    navigate('/dashboard/detection', { state: { patientId } });
  };

  return (
    <div className="patients-page">
      <div className="page-header flex-between">
        <div>
          <h1 className="page-title">Patient Management</h1>
          <p className="page-subtitle">View and manage patient records and histories.</p>
        </div>
        <button className="btn-primary flex-center gap-2" onClick={() => setShowAddModal(true)}>
          <UserPlus size={18} /> Add Patient
        </button>
      </div>

      <div className="card mt-6">
        <div className="table-toolbar">
           <div className="search-bar">
             <Search size={18} className="search-icon" />
             <input type="text" placeholder="Search patients..." className="search-input" />
           </div>
        </div>
        
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Patient ID</th>
                <th>Name</th>
                <th>Age</th>
                <th>Registered Date</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan="5" className="text-center py-8">Loading...</td></tr>
              ) : patients.length === 0 ? (
                <tr><td colSpan="5" className="text-center py-8 text-gray-500">No patients found. Create one above.</td></tr>
              ) : (
                patients.map(p => (
                  <tr key={p.id}>
                    <td className="font-mono text-sm text-gray-400">#{p.id}</td>
                    <td className="font-medium">{p.name}</td>
                    <td>{p.age}</td>
                    <td>{new Date(p.created_at).toLocaleDateString()}</td>
                    <td>
                      <div className="action-btns">
                        <button 
                          className="btn-ghost icon-btn" 
                          title="New Scan"
                          onClick={() => handleScanPatient(p.id)}
                        >
                          <ScanLine size={18} /> New Scan
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Basic Modal */}
      {showAddModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h3 className="modal-title">Register New Patient</h3>
            <form onSubmit={handleAddPatient}>
              <div className="input-group">
                <label>Full Name</label>
                <input 
                  type="text" 
                  value={newPatient.name} 
                  onChange={e => setNewPatient({...newPatient, name: e.target.value})} 
                  required 
                  className="modal-input"
                />
              </div>
              <div className="input-group mt-4">
                <label>Age</label>
                <input 
                  type="number" 
                  value={newPatient.age} 
                  onChange={e => setNewPatient({...newPatient, age: e.target.value})} 
                  required 
                  className="modal-input"
                  min="0" max="150"
                />
              </div>
              <div className="modal-actions mt-6">
                <button type="button" className="btn-ghost" onClick={() => setShowAddModal(false)}>Cancel</button>
                <button type="submit" className="btn-primary">Save Patient</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

// Ensure ScanLine is imported for the icon button
import { ScanLine } from 'lucide-react';
