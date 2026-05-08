// frontend/src/components/DocumentUpload.jsx
import { useState } from "react";
import axios from "axios";

const API      = process.env.REACT_APP_GATEWAY_URL || "http://localhost:9090";
const DOC_TYPES = ["loan_agreement", "policy", "contract", "emi_schedule", "general"];

export default function DocumentUpload() {
  const [file, setFile]       = useState(null);
  const [docType, setDocType] = useState("loan_agreement");
  const [status, setStatus]   = useState(null);
  const [loading, setLoading] = useState(false);

  const upload = async () => {
    if (!file) return;
    setLoading(true);
    setStatus(null);
    const form = new FormData();
    form.append("file",     file);
    form.append("doc_type", docType);
    try {
      const { data } = await axios.post(`${API}/api/ingest`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setStatus({ success: true, data });
      setFile(null);
    } catch (e) {
      setStatus({ success: false, error: e.response?.data?.detail || e.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto mt-8 bg-white rounded-xl shadow p-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-1">Upload BFSI Document</h2>
      <p className="text-sm text-gray-500 mb-6">
        PDF or DOCX · Max 50MB · Processed locally — never uploaded to any external server
      </p>

      {/* Drop zone */}
      <label className="block border-2 border-dashed border-gray-300 rounded-xl p-8 text-center cursor-pointer hover:border-blue-400 transition">
        <input type="file" accept=".pdf,.docx" className="hidden"
          onChange={e => setFile(e.target.files[0])} />
        {file
          ? <p className="text-blue-700 font-medium">{file.name}</p>
          : <><p className="text-gray-500">Drop a file here or click to browse</p>
              <p className="text-xs text-gray-400 mt-1">Loan agreements · Policies · Contracts</p></>
        }
      </label>

      {/* Doc type selector */}
      <div className="mt-4">
        <label className="text-sm text-gray-600 block mb-1">Document type</label>
        <select value={docType} onChange={e => setDocType(e.target.value)}
          className="border rounded-lg px-3 py-2 text-sm w-full focus:outline-none focus:ring-2 focus:ring-blue-500">
          {DOC_TYPES.map(t => <option key={t}>{t}</option>)}
        </select>
      </div>

      <button onClick={upload} disabled={!file || loading}
        className="mt-5 w-full bg-blue-700 text-white py-2.5 rounded-lg font-medium hover:bg-blue-800 disabled:opacity-50 transition">
        {loading ? "Processing locally..." : "Ingest Document"}
      </button>

      {/* Result */}
      {status && (
        <div className={`mt-4 rounded-lg p-4 text-sm ${status.success ? "bg-green-50 text-green-800" : "bg-red-50 text-red-800"}`}>
          {status.success ? (
            <>
              <p className="font-medium">✅ Document ingested successfully</p>
              <p className="mt-1">Chunks stored: <strong>{status.data.chunks}</strong></p>
              <p>Pages parsed: <strong>{status.data.pages}</strong></p>
              <p>Provider: <strong>{status.data.llm_provider}</strong> · Time: <strong>{status.data.elapsed_sec}s</strong></p>
            </>
          ) : (
            <p>❌ {status.error}</p>
          )}
        </div>
      )}
    </div>
  );
}
