// frontend/src/App.jsx
import { useState } from "react";
import ChatInterface from "./components/ChatInterface";
import AnalyticsDashboard from "./components/AnalyticsDashboard";
import DocumentUpload from "./components/DocumentUpload";

export default function App() {
  const [tab, setTab] = useState("chat");

  const tabs = [
    { id: "chat",      label: "💬 Ask Documents" },
    { id: "upload",    label: "📄 Upload" },
    { id: "analytics", label: "📊 Analytics" },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-blue-900 text-white px-6 py-3 flex items-center gap-4 shadow-lg">
        <div>
          <span className="font-bold text-lg">🏦 BFSI Document Intelligence</span>
          <span className="ml-3 text-xs bg-blue-700 px-2 py-0.5 rounded">
            Powered by Ollama (local) · AWS Bedrock (cloud)
          </span>
        </div>
        <div className="ml-auto flex gap-2">
          {tabs.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)}
              className={`px-4 py-1.5 rounded text-sm font-medium transition ${
                tab === t.id ? "bg-white text-blue-900" : "hover:bg-blue-800"}`}>
              {t.label}
            </button>
          ))}
        </div>
      </nav>
      <main className="max-w-6xl mx-auto p-4">
        {tab === "chat"      && <ChatInterface />}
        {tab === "upload"    && <DocumentUpload />}
        {tab === "analytics" && <AnalyticsDashboard />}
      </main>
    </div>
  );
}
