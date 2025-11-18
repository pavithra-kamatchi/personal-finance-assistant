// Upload Page
// CSV file upload for transaction data
// Sends to POST /transactions/upload-transactions

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from '../api/axios';
import Navbar from '../components/Navbar';

const Upload: React.FC = () => {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];

      // Validate file type
      if (!selectedFile.name.endsWith('.csv')) {
        setError('Please select a CSV file');
        setFile(null);
        return;
      }

      setFile(selectedFile);
      setError('');
      setSuccess('');
    }
  };
  // Handle file upload submission
  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!file) {
      setError('Please select a file');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');

    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('file', file);

      // Call backend upload endpoint
      const response = await axios.post('/transactions/upload-transactions', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.status === 200) {
        setSuccess(response.data.message || 'Transactions uploaded successfully!');
        setFile(null);

        // Reset file input
        const fileInput = document.getElementById('file-input') as HTMLInputElement;
        if (fileInput) fileInput.value = '';

        // Redirect to dashboard after 2 seconds
        setTimeout(() => {
          navigate('/dashboard');
        }, 2000);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Upload failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Navbar />
      <div className="upload-container">
        <div className="upload-card">
          <h2>Upload Transactions</h2>
          <p className="upload-subtitle">
            Upload a CSV file with your transaction data to analyze your spending
          </p>

          {error && <div className="error-message">{error}</div>}
          {success && <div className="success-message">{success}</div>}

          <form onSubmit={handleUpload}>
            <div className="file-upload-section">
              <label htmlFor="file-input" className="file-label">
                {file ? (
                  <>
                    <span className="file-icon">📄</span>
                    <span>{file.name}</span>
                  </>
                ) : (
                  <>
                    <span className="file-icon">📂</span>
                    <span>Click to select CSV file</span>
                  </>
                )}
              </label>
              <input
                id="file-input"
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                disabled={loading}
                style={{ display: 'none' }}
              />
            </div>

            <div className="upload-info">
              <h4>CSV Format Requirements:</h4>
              <ul>
                <li>File must be in CSV format (.csv)</li>
                <li>Expected columns: date, description, amount</li>
                <li>Date format: YYYY-MM-DD or MM/DD/YYYY</li>
                <li>Amount should be numeric (positive for income, negative for expenses)</li>
              </ul>
            </div>

            <button
              type="submit"
              className="btn btn-primary btn-full"
              disabled={!file || loading}
            >
              {loading ? 'Uploading...' : 'Upload & Process'}
            </button>
          </form>
        </div>
      </div>
    </>
  );
};

export default Upload;
