// frontend/src/components/ChatInterface.jsx
import { useState } from "react";
import axios from "axios";

const API = process.env.REACT_APP_GATEWAY_URL || "http://localhost:9090";

const DOC_TYPES = ["any", "loan_agreement", "policy", "contract", "emi_schedule"];

export default function ChatInterface() {
  const [messages, setMessages]   = useState([]);
  const [input, setInput]         = useState("");
  const [docType, setDocType]     = useState("any");
  const [loading, setLoading]     = useState(false);

  const send = async () => {
    if (!input.trim()) return;
    const q = input.trim();
    setMessages(m => [...m, { role: "user", content: q }]);
    setInput("");
    setLoading(true);

    try {
      const { data } = await axios.post(`${API}/api/query`, {
        question: q,
        doc_type: docType === "any" ? null : docType,
        top_k: 5,
      });
      setMessages(m => [...m, {
        role:     "assistant",
        content:  data.answer,
        sources:  data.sources,
        provider: data.provider,
        latency:  data.latency_ms,
      }]);
    } catch (e) {
      setMessages(m => [...m, { role: "assistant", content: "Error: " + e.message, error: true }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[85vh] bg-white rounded-xl shadow mt-4">
      {/* Toolbar */}
      <div className="flex items-center gap-3 px-4 py-3 border-b bg-gray-50 rounded-t-xl">
        <label className="text-sm text-gray-600">Filter by type:</label>
        <select value={docType} onChange={e => setDocType(e.target.value)}
          className="text-sm border rounded px-2 py-1 bg-white">
          {DOC_TYPES.map(t => <option key={t}>{t}</option>)}
        </select>
        <span className="ml-auto text-xs text-gray-400">
          All queries processed locally — no data leaves the server
        </span>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-20">
            <p className="text-xl mb-2">Ask anything about your BFSI documents</p>
            <p className="text-sm">Upload loan agreements, policies, or contracts first</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-2xl rounded-xl px-4 py-3 ${
              m.role === "user"
                ? "bg-blue-600 text-white"
                : m.error ? "bg-red-50 text-red-700 border border-red-200"
                : "bg-gray-100 text-gray-800"}`}>
              <p className="whitespace-pre-wrap text-sm">{m.content}</p>
              {m.sources?.length > 0 && (
                <div className="mt-2 pt-2 border-t border-gray-200">
                  <p className="text-xs text-gray-500">Sources: {m.sources.join(" · ")}</p>
                  <p className="text-xs text-green-600 mt-0.5">
                    🔒 {m.provider === "ollama" ? "Processed locally via Ollama" : "Azure OpenAI"} · {m.latency}ms
                  </p>
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-xl px-4 py-3 text-sm text-gray-500 animate-pulse">
              Thinking locally...
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-4 border-t flex gap-2">
        <input value={input} onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && !e.shiftKey && send()}
          placeholder="e.g. What is the EMI for loan L-2024-001?"
          className="flex-1 border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
        <button onClick={send} disabled={loading}
          className="bg-blue-700 text-white px-5 py-2 rounded-lg text-sm hover:bg-blue-800 disabled:opacity-50">
          Ask
        </button>
      </div>
    </div>
  );
}
