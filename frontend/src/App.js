import React, { useState, useRef, useEffect } from "react";
import "./App.css";

const BACKEND = "http://localhost:8001";

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hello! 👋 I'm Nova, your company policy assistant. Ask me anything about leave, travel, IT assets, code of conduct, or onboarding policies!",
    },
  ]);
  const [input, setInput]     = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false); 
  
  const bottomRef    = useRef(null);
  const fileInputRef = useRef(null); 

  
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, uploading]);

  
  const send = async (text) => {
    const question = (text || input).trim();
    if (!question || loading) return;

    const updated = [...messages, { role: "user", content: question }];
    setMessages(updated);
    setInput("");
    setLoading(true);
    try {
      const res  = await fetch(`${BACKEND}/api/query`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ question }),
      });
      const data = await res.json();
      setMessages([...updated, { role: "assistant", content: data.answer }]);
    } catch {
      setMessages([...updated, {
        role:    "assistant",
        content: "⚠️ Cannot reach the server. Make sure the backend is running.",
      }]);
    } finally {
      setLoading(false);
    }
  };

  
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploading(true);
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${BACKEND}/api/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      
      if (data.status === 'success') {
        setMessages(prev => [...prev, { 
          role: "assistant", 
          content: `✅ Document uploaded: **${file.name}** has been securely saved to the cloud storage.` 
        }]);
      } else {
        setMessages(prev => [...prev, { 
          role: "assistant", 
          content: `⚠️ Upload error: ${data.message}` 
        }]);
      }
    } catch (error) {
      console.error("Upload failed:", error);
      setMessages(prev => [...prev, { 
        role: "assistant", 
        content: `⚠️ Failed to reach the upload gateway. Is the backend running?` 
      }]);
    } finally {
      setUploading(false);
      // Reset the file input so the same file can be selected again if needed
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const quickQuestions = [
    "How many casual leaves do I get?",
    "What is my notice period?",
    "What is the travel allowance?",
    "What happens if I lose my laptop?",
    "How do I apply for maternity leave?",
    "What is the onboarding process?",
  ];

  return (
    <div className="app">

      {/* ── Header ── */}
      <div className="header">
        <div className="logo">NT</div>
        <div>
          <div className="title">Nova Technologies</div>
          <div className="subtitle">Company Policy Assistant</div>
        </div>
        <div className="badge">Gemini · LangGraph</div>
      </div>

      {/* ── Quick questions ── */}
      <div className="quick-bar">
        <span className="quick-label">QUICK QUESTIONS:</span>
        {quickQuestions.map((q, i) => (
          <button key={i} className="chip" onClick={() => send(q)}>{q}</button>
        ))}
      </div>

      {/* ── Chat area ── */}
      <div className="chat">
        <div className="messages">
          {messages.map((m, i) => (
            <div key={i} className={`row ${m.role}`}>
              {m.role === "assistant" && <div className="av">NT</div>}
              <div className="bubble">{m.content}</div>
            </div>
          ))}

          {loading && (
            <div className="row assistant">
              <div className="av">NT</div>
              <div className="bubble dots">
                <span/><span/><span/>
              </div>
            </div>
          )}
          
          {uploading && (
            <div className="row assistant">
              <div className="av">NT</div>
              <div className="bubble" style={{ fontStyle: 'italic', color: '#666' }}>
                Uploading document to cloud storage... ⏳
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* ── Input ── */}
        <div className="input-row" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          
          {/* Hidden File Input */}
          <input 
            type="file" 
            accept=".pdf,.zip" 
            ref={fileInputRef}
            onChange={handleFileUpload}
            style={{ display: 'none' }} 
          />
          
          {/* Attachment Button */}
          <button 
            onClick={() => fileInputRef.current?.click()} 
            disabled={loading || uploading}
            title="Upload Policy Document (PDF/ZIP)"
            style={{
              background: 'transparent',
              border: 'none',
              fontSize: '20px',
              cursor: (loading || uploading) ? 'not-allowed' : 'pointer',
              opacity: (loading || uploading) ? 0.5 : 1,
              padding: '0 8px'
            }}
          >
            📎
          </button>

          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }}}
            placeholder="Ask me about company policies..."
            rows={1}
            disabled={loading || uploading}
            style={{ flex: 1 }}
          />
          
          <button onClick={() => send()} disabled={loading || uploading || !input.trim()}>↑</button>
        </div>
        <div className="powered">Powered by Gemini · LangGraph</div>
      </div>
    </div>
  );
}