// frontend/src/components/AnalyticsDashboard.jsx
import { useEffect, useState } from "react";
import axios from "axios";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from "recharts";

const API    = process.env.REACT_APP_GATEWAY_URL || "http://localhost:9090";
const COLORS = ["#1e40af", "#16a34a", "#dc2626", "#d97706", "#7c3aed"];

export default function AnalyticsDashboard() {
  const [portfolio, setPortfolio] = useState([]);
  const [overdue, setOverdue]     = useState([]);
  const [nlQuery, setNlQuery]     = useState("");
  const [nlResult, setNlResult]   = useState(null);
  const [loading, setLoading]     = useState(false);

  useEffect(() => {
    axios.get(`${API}/api/analytics/portfolio-summary`)
      .then(r => setPortfolio(r.data.results || []));
    axios.get(`${API}/api/analytics/overdue-emi?days=0`)
      .then(r => setOverdue(r.data.results || []));
  }, []);

  const runNlQuery = async () => {
    if (!nlQuery.trim()) return;
    setLoading(true);
    try {
      const { data } = await axios.post(`${API}/api/analytics/query`, { question: nlQuery });
      setNlResult(data);
    } catch (e) {
      setNlResult({ success: false, error: e.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 mt-4">
      <h2 className="text-xl font-semibold text-gray-800">Loan Portfolio Analytics</h2>

      {/* Charts row */}
      <div className="grid grid-cols-2 gap-4">
        {/* Bar chart — disbursed by status */}
        <div className="bg-white rounded-xl shadow p-4">
          <h3 className="text-sm font-medium text-gray-600 mb-3">Total Disbursed by Status</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={portfolio}>
              <XAxis dataKey="status" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip formatter={v => `₹${Number(v).toLocaleString("en-IN")}`} />
              <Bar dataKey="total_disbursed" fill="#1e40af" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Pie chart — loan count */}
        <div className="bg-white rounded-xl shadow p-4">
          <h3 className="text-sm font-medium text-gray-600 mb-3">Loan Count by Status</h3>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={portfolio} dataKey="loan_count" nameKey="status" cx="50%" cy="50%" outerRadius={80} label>
                {portfolio.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Legend />
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Overdue EMI table */}
      {overdue.length > 0 && (
        <div className="bg-white rounded-xl shadow p-4">
          <h3 className="text-sm font-medium text-gray-600 mb-3">Overdue EMIs</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr>{Object.keys(overdue[0]).map(k => (
                  <th key={k} className="text-left px-3 py-2 text-gray-500 font-medium">{k}</th>
                ))}</tr>
              </thead>
              <tbody>
                {overdue.map((row, i) => (
                  <tr key={i} className="border-t hover:bg-gray-50">
                    {Object.values(row).map((v, j) => (
                      <td key={j} className="px-3 py-2 text-gray-700">{String(v)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Natural language query box */}
      <div className="bg-white rounded-xl shadow p-4">
        <h3 className="text-sm font-medium text-gray-600 mb-3">Natural Language Query → SQL</h3>
        <div className="flex gap-2">
          <input value={nlQuery} onChange={e => setNlQuery(e.target.value)}
            onKeyDown={e => e.key === "Enter" && runNlQuery()}
            placeholder="e.g. Which loans have NPA status and amount above 5 lakhs?"
            className="flex-1 border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
          <button onClick={runNlQuery} disabled={loading}
            className="bg-blue-700 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-800 disabled:opacity-50">
            Run
          </button>
        </div>
        {nlResult && (
          <div className="mt-3">
            {nlResult.sql && (
              <pre className="bg-gray-900 text-green-400 text-xs p-3 rounded-lg overflow-x-auto mb-3">
                {nlResult.sql}
              </pre>
            )}
            {nlResult.success && nlResult.results?.length > 0 && (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>{nlResult.columns.map(c => (
                      <th key={c} className="text-left px-3 py-2 text-gray-500 font-medium">{c}</th>
                    ))}</tr>
                  </thead>
                  <tbody>
                    {nlResult.results.map((row, i) => (
                      <tr key={i} className="border-t">
                        {Object.values(row).map((v, j) => (
                          <td key={j} className="px-3 py-2 text-gray-700">{String(v)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {!nlResult.success && <p className="text-red-600 text-sm">{nlResult.error}</p>}
          </div>
        )}
      </div>
    </div>
  );
}
