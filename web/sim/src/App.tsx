import { useEffect, useMemo, useState, useRef } from "react";
import type { Edge, Node } from "@xyflow/react";
import { buildLifeEventGraph, buildSessionKnowledgeGraph } from "./buildGraph";
import type { LifeEvent, Session, SimDataset } from "./dataset";
import { FlowPanel } from "./FlowPanel";
import { mockRetrieve } from "./mockRetrieval";
import ReactMarkdown from 'react-markdown';

// Icons
const MenuIcon = () => <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>;
const PanelRightIcon = () => <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="18" cy="5" r="3"></circle><circle cx="6" cy="12" r="3"></circle><circle cx="18" cy="19" r="3"></circle><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line></svg>;
const SearchIcon = () => <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>;

function loadDataset(): Promise<{ data: SimDataset; events: LifeEvent[] }> {
  return Promise.all([
    fetch(`${import.meta.env.BASE_URL}dataset/conversation_sim_10.json`).then((r) => {
      if (!r.ok) throw new Error(`Dataset HTTP ${r.status}`);
      return r.json() as Promise<SimDataset>;
    }),
    fetch(`${import.meta.env.BASE_URL}dataset/user_events.json`).then((r) => {
      if (!r.ok) return [] as LifeEvent[];
      return r.json() as Promise<LifeEvent[]>;
    }),
  ]).then(([data, events]) => ({ data, events }));
}

export default function App() {
  const [dataset, setDataset] = useState<SimDataset | null>(null);
  const [events, setEvents] = useState<LifeEvent[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [sessionIdx, setSessionIdx] = useState(0);
  const [turnIdx, setTurnIdx] = useState(0);
  const [graphMode, setGraphMode] = useState<"session" | "events">("session");
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<ReturnType<typeof mockRetrieve>>([]);
  
  // Responsive UI States
  const [leftOpen, setLeftOpen] = useState(true);
  const [leftWidth, setLeftWidth] = useState(280);
  const [isDraggingLeft, setIsDraggingLeft] = useState(false);
  
  const [rightOpen, setRightOpen] = useState(true);
  const [rightWidth, setRightWidth] = useState(450);
  const [isDragging, setIsDragging] = useState(false);
  
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Load Data
  useEffect(() => {
    loadDataset()
      .then(({ data, events: lifeEvents }) => {
        setDataset(data);
        setEvents(lifeEvents);
      })
      .catch((e: Error) => setErr(e.message));
  }, []);

  // Handle Responsive Layout
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 850) {
        setLeftOpen(false);
        setRightOpen(false);
      } else if (window.innerWidth < 1150) {
        setRightOpen(false);
        setLeftOpen(true);
      } else {
        setLeftOpen(true);
        setRightOpen(true);
      }
    };
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Handle Resizing Panels
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        // Calculate new width based on mouse position (from right edge of screen)
        const newWidth = window.innerWidth - e.clientX;
        // Constrain width between 300px and 800px (or 60% of screen)
        const maxWidth = Math.min(800, window.innerWidth * 0.6);
        setRightWidth(Math.max(300, Math.min(maxWidth, newWidth)));
      } else if (isDraggingLeft) {
        // Calculate new width based on mouse position (from left edge of screen)
        const newWidth = e.clientX;
        // Constrain width between 200px and 600px (or 40% of screen)
        const maxWidth = Math.min(600, window.innerWidth * 0.4);
        setLeftWidth(Math.max(200, Math.min(maxWidth, newWidth)));
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      setIsDraggingLeft(false);
      document.body.style.cursor = "default";
      document.body.style.userSelect = "auto";
    };

    if (isDragging || isDraggingLeft) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging, isDraggingLeft]);

  const session: Session | null = dataset?.sessions[sessionIdx] ?? null;

  useEffect(() => {
    setTurnIdx(0);
    setHits([]);
    setQ("");
  }, [sessionIdx]);

  // Auto scroll to bottom of chat when turn changes
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turnIdx]);

  const { sessionNodes, sessionEdges, eventNodes, eventEdges } = useMemo(() => {
    if (!dataset || !session) {
      return {
        sessionNodes: [] as Node[],
        sessionEdges: [] as Edge[],
        eventNodes: [] as Node[],
        eventEdges: [] as Edge[],
      };
    }
    const sg = buildSessionKnowledgeGraph(
      session,
      turnIdx,
      dataset.user,
      dataset.secondary_personas || []
    );
    const eg = buildLifeEventGraph(events);
    return {
      sessionNodes: sg.nodes,
      sessionEdges: sg.edges,
      eventNodes: eg.nodes,
      eventEdges: eg.edges,
    };
  }, [dataset, session, turnIdx, events]);

  const maxTurn = session ? Math.max(0, session.turns.length - 1) : 0;

  const runMock = () => {
    if (!session || !q.trim()) return;
    setHits(mockRetrieve(q, session, turnIdx, 5));
  };

  if (err) {
    return (
      <div style={{ padding: 40, display: "flex", justifyContent: "center", height: "100vh", alignItems: "center", background: "var(--bg-app)" }}>
        <div style={{ background: "#fef2f2", color: "var(--error)", padding: 32, borderRadius: "var(--radius-lg)", border: "1px solid #fca5a5", maxWidth: 500, boxShadow: "var(--shadow-lg)" }}>
          <h1 style={{ marginTop: 0, fontSize: 22, display: "flex", alignItems: "center", gap: 12 }}>
            <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
            Gagal memuat dataset
          </h1>
          <p style={{ lineHeight: 1.6 }}>{err}</p>
          <div className="mono" style={{ fontSize: 12, opacity: 0.8, marginTop: 16, background: "rgba(239, 68, 68, 0.1)", padding: 12, borderRadius: 8 }}>
            Jalankan dari root repo:<br/>
            <code>python scripts/export_sim_ui_dataset.py</code><br/>
            Lalu jalankan ulang dev server.
          </div>
        </div>
      </div>
    );
  }

  if (!dataset) {
    return (
      <div style={{ height: "100vh", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)", background: "var(--bg-app)" }}>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
          <div style={{ width: 36, height: 36, border: "3px solid var(--border-base)", borderTopColor: "var(--primary)", borderRadius: "50%", animation: "spin 1s linear infinite" }} />
          <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
          <div style={{ fontWeight: 500 }}>Memuat dataset interaktif...</div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column", background: "var(--bg-app)", overflow: "hidden" }}>
      {/* Header Utama */}
      <header
        style={{
          padding: "12px 20px",
          background: "var(--bg-panel)",
          borderBottom: "1px solid var(--border-base)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 16,
          boxShadow: "var(--shadow-sm)",
          position: "relative",
          zIndex: 10,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 16, minWidth: 0 }}>
          <button 
            onClick={() => setLeftOpen(!leftOpen)}
            className="hover-bg-btn"
            style={{ background: leftOpen ? "var(--primary-bg)" : "transparent", border: "none", color: leftOpen ? "var(--primary)" : "var(--text-muted)", padding: 8, borderRadius: 8, display: "flex", flexShrink: 0 }}
            title="Toggle Sidebar Sesi"
          >
            <MenuIcon />
          </button>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 40,
              height: 40,
              background: "#ffffff",
              borderRadius: 10,
              boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
              border: "1px solid var(--border-light)"
            }}>
              <img 
                src={`${import.meta.env.BASE_URL}assets/logo.png`} 
                alt="Tempograph Logo" 
                style={{ 
                  width: "140%", 
                  height: "140%", 
                  objectFit: "contain",
                }} 
              />
            </div>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontWeight: 700, fontSize: 16, letterSpacing: "-0.02em", color: "var(--text-main)", lineHeight: 1.1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                Tempograph
              </div>
              <div style={{ fontSize: 12, color: "var(--text-lighter)", marginTop: 2, fontWeight: 500, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                Simulasi Interaktif Dataset
              </div>
            </div>
          </div>
        </div>
        
        <div style={{ display: "flex", alignItems: "center", gap: 16, flexShrink: 0 }}>
          <div style={{ fontSize: 11, color: "var(--text-lighter)", display: "none", '@media (minWidth: 600px)': { display: 'block' } } as any}>
            Source: <span className="mono" style={{ color: "var(--text-muted)", fontWeight: 500 }}>conversation_sim_10.json</span>
          </div>
          <button 
            onClick={() => setRightOpen(!rightOpen)}
            className="hover-bg-btn"
            style={{ background: rightOpen ? "var(--primary-bg)" : "transparent", border: "none", color: rightOpen ? "var(--primary)" : "var(--text-muted)", padding: 8, borderRadius: 8, display: "flex", flexShrink: 0 }}
            title="Toggle Panel Graf"
          >
            <PanelRightIcon />
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main style={{ flex: 1, display: "flex", overflow: "hidden", position: "relative" }}>
        
        {/* Left Sidebar: Daftar Sesi */}
        <aside
          style={{
            width: leftOpen ? leftWidth : 0,
            opacity: leftOpen ? 1 : 0,
            transition: isDraggingLeft ? "none" : "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
            borderRight: leftOpen ? "1px solid var(--border-base)" : "none",
            background: "var(--bg-panel)",
            flexShrink: 0,
            zIndex: 5,
            position: "relative",
          }}
        >
          {/* Resize Handle for Left Panel */}
          {leftOpen && (
            <div
              onMouseDown={() => setIsDraggingLeft(true)}
              style={{
                position: "absolute",
                right: -4,
                top: 0,
                bottom: 0,
                width: 8,
                cursor: "col-resize",
                zIndex: 10,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <div style={{ 
                width: 4, 
                height: 32, 
                background: isDraggingLeft ? "var(--primary)" : "transparent", 
                borderRadius: 4,
                transition: "background 0.2s"
              }} className="resize-indicator" />
            </div>
          )}

          <div style={{ width: leftOpen ? leftWidth : 0, height: "100%", display: "flex", flexDirection: "column", padding: "20px 16px", overflowY: "auto", overflowX: "hidden" }}>
            <div style={{ fontSize: 11, fontWeight: 700, textTransform: "uppercase", color: "var(--text-lighter)", marginBottom: 12, paddingLeft: 8, letterSpacing: "0.05em" }}>
              Daftar Sesi ({dataset.sessions.length})
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {dataset.sessions.map((s, i) => {
                const isActive = i === sessionIdx;
                return (
                  <button
                    key={s.session_id}
                    type="button"
                    onClick={() => setSessionIdx(i)}
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      width: "100%",
                      textAlign: "left",
                      padding: "12px 16px",
                      borderRadius: "var(--radius-md)",
                      border: "none",
                      background: isActive ? "var(--primary-bg)" : "transparent",
                      color: isActive ? "var(--primary)" : "var(--text-main)",
                      position: "relative",
                      transition: "background 0.2s",
                    }}
                    onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.background = "var(--bg-app)"; }}
                    onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.background = "transparent"; }}
                  >
                    {isActive && (
                      <div style={{ position: "absolute", left: 0, top: "25%", bottom: "25%", width: 3, background: "var(--primary)", borderRadius: "0 4px 4px 0" }} />
                    )}
                    <div style={{ fontWeight: 600, fontSize: 14 }}>Sesi {s.session_id}</div>
                    <div style={{ fontSize: 12, color: isActive ? "var(--primary)" : "var(--text-muted)", opacity: isActive ? 0.8 : 1, marginTop: 4 }}>
                      {s.date}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </aside>

        {/* Center: Chat & Controls */}
        <section style={{ flex: 1, minWidth: 320, display: "flex", flexDirection: "column", background: "var(--bg-chat)", position: "relative", zIndex: 1 }}>
          {session && (
            <>
              {/* Top Bar: Slider Giliran */}
              <div
                style={{
                  padding: "16px 24px",
                  borderBottom: "1px solid var(--border-base)",
                  display: "flex",
                  alignItems: "center",
                  gap: 16,
                  background: "var(--bg-panel)",
                  boxShadow: "var(--shadow-sm)",
                  zIndex: 2,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 12, flex: 1, background: "var(--bg-app)", padding: "8px 16px", borderRadius: "var(--radius-full)", border: "1px solid var(--border-light)" }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>Giliran</span>
                  <input
                    type="range"
                    min={0}
                    max={maxTurn}
                    value={turnIdx}
                    onChange={(e) => setTurnIdx(Number(e.target.value))}
                    style={{ flex: 1 }}
                  />
                  <span className="mono" style={{ fontSize: 13, fontWeight: 600, color: "var(--primary)", minWidth: 48, textAlign: "right" }}>
                    {turnIdx + 1} <span style={{ color: "var(--text-lighter)", fontWeight: 400 }}>/ {session.turns.length}</span>
                  </span>
                </div>
                
                <div style={{ display: "flex", gap: 8 }}>
                  <button
                    type="button"
                    onClick={() => setTurnIdx((t) => Math.min(maxTurn, t + 1))}
                    style={{
                      padding: "10px 18px",
                      borderRadius: "var(--radius-full)",
                      border: "none",
                      background: "var(--primary)",
                      color: "white",
                      fontWeight: 600,
                      fontSize: 13,
                      boxShadow: "0 2px 8px rgba(59, 130, 246, 0.3)",
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.transform = "translateY(-1px)"}
                    onMouseLeave={(e) => e.currentTarget.style.transform = "translateY(0)"}
                  >
                    Maju +1
                  </button>
                  <button
                    type="button"
                    onClick={() => setTurnIdx(maxTurn)}
                    style={{
                      padding: "10px 16px",
                      borderRadius: "var(--radius-full)",
                      border: "1px solid var(--border-dark)",
                      background: "var(--bg-panel)",
                      color: "var(--text-main)",
                      fontWeight: 500,
                      fontSize: 13,
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = "var(--bg-app)"}
                    onMouseLeave={(e) => e.currentTarget.style.background = "var(--bg-panel)"}
                  >
                    Akhir
                  </button>
                </div>
              </div>

              {/* Transkrip Chat */}
              <div style={{ flex: 1, overflowY: "auto", padding: "24px 32px", scrollBehavior: "smooth" }}>
                <div style={{ textAlign: "center", marginBottom: 32 }}>
                  <span style={{ fontSize: 12, fontWeight: 500, color: "var(--text-muted)", background: "var(--border-light)", padding: "6px 14px", borderRadius: "var(--radius-full)" }}>
                    {session.datetime || session.date}
                  </span>
                </div>
                
                <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                  {session.turns.slice(0, turnIdx + 1).map((t, i) => {
                    const isUser = t.speaker === "user";
                    return (
                      <div
                        key={`${sessionIdx}-${i}`}
                        className="animate-msg"
                        style={{
                          display: "flex",
                          flexDirection: "column",
                          alignItems: isUser ? "flex-end" : "flex-start",
                        }}
                      >
                        <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-lighter)", marginBottom: 6, padding: "0 4px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                          {t.speaker}
                        </div>
                        <div
                          style={{
                            maxWidth: "85%",
                            padding: "14px 18px",
                            borderRadius: "20px",
                            borderBottomRightRadius: isUser ? "4px" : "20px",
                            borderBottomLeftRadius: !isUser ? "4px" : "20px",
                            background: isUser ? "var(--msg-user-bg)" : "var(--msg-bot-bg)",
                            color: isUser ? "var(--msg-user-text)" : "var(--msg-bot-text)",
                            border: isUser ? "none" : "1px solid var(--border-base)",
                            boxShadow: isUser ? "0 4px 12px rgba(59, 130, 246, 0.25)" : "var(--shadow-sm)",
                            fontSize: 15,
                            lineHeight: 1.5,
                          }}
                        >
                          <ReactMarkdown
                            components={{
                              p: ({node, ...props}) => <p style={{ margin: 0, padding: 0 }} {...props} />,
                              em: ({node, ...props}) => <em style={{ fontStyle: "italic", color: isUser ? "rgba(255,255,255,0.9)" : "var(--primary)" }} {...props} />,
                              strong: ({node, ...props}) => <strong style={{ fontWeight: 700 }} {...props} />,
                            }}
                          >
                            {t.text}
                          </ReactMarkdown>
                        </div>
                      </div>
                    );
                  })}
                  <div ref={chatEndRef} style={{ height: 20 }} />
                </div>
              </div>

              {/* Mock Retrieval Bottom Bar */}
              <div style={{ padding: "20px 24px", background: "var(--bg-panel)", borderTop: "1px solid var(--border-base)", boxShadow: "0 -4px 12px rgba(0,0,0,0.02)", zIndex: 2 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <div style={{ flex: 1, position: "relative" }}>
                    <div style={{ position: "absolute", left: 14, top: "50%", transform: "translateY(-50%)", color: "var(--text-lighter)", pointerEvents: "none", display: "flex" }}>
                      <SearchIcon />
                    </div>
                    <input
                      value={q}
                      onChange={(e) => setQ(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && runMock()}
                      placeholder="Cari di transkrip (Mock Jaccard)..."
                      style={{
                        width: "100%",
                        padding: "12px 16px 12px 42px",
                        borderRadius: "var(--radius-lg)",
                        border: "1px solid var(--border-dark)",
                        background: "var(--bg-app)",
                        color: "var(--text-main)",
                        fontSize: 14,
                      }}
                    />
                  </div>
                  <button
                    type="button"
                    onClick={runMock}
                    style={{
                      padding: "12px 24px",
                      borderRadius: "var(--radius-lg)",
                      border: "none",
                      background: "var(--text-main)",
                      color: "white",
                      fontWeight: 600,
                      fontSize: 14,
                      boxShadow: "var(--shadow-md)",
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.transform = "translateY(-1px)"}
                    onMouseLeave={(e) => e.currentTarget.style.transform = "translateY(0)"}
                  >
                    Retrieve
                  </button>
                </div>
                
                {hits.length > 0 && (
                  <div className="animate-msg" style={{ marginTop: 16, background: "var(--bg-app)", borderRadius: "var(--radius-md)", padding: "16px", border: "1px solid var(--border-light)" }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text-muted)", marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.05em" }}>Hasil Pencarian ({hits.length})</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                      {hits.map((h, i) => (
                        <div key={i} style={{ display: "flex", gap: 12, fontSize: 13, alignItems: "flex-start", background: "var(--bg-panel)", padding: "10px 12px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-base)" }}>
                          <span className="mono" style={{ color: "var(--success)", fontWeight: 600, background: "var(--success-bg)", padding: "2px 6px", borderRadius: 4, fontSize: 11, marginTop: 2 }}>
                            {(h.score * 100).toFixed(0)}%
                          </span>
                          <span style={{ color: "var(--text-muted)", fontSize: 11, fontWeight: 600, textTransform: "uppercase", width: 50, marginTop: 2 }}>{h.kind}</span>
                          <span style={{ color: "var(--text-main)", flex: 1, lineHeight: 1.4 }}>{h.preview}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </section>

        {/* Right Sidebar: Graf & Ringkasan */}
        <aside
          style={{
            width: rightOpen ? rightWidth : 0,
            opacity: rightOpen ? 1 : 0,
            transition: isDragging ? "none" : "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
            borderLeft: rightOpen ? "1px solid var(--border-base)" : "none",
            background: "var(--bg-panel)",
            flexShrink: 0,
            zIndex: 5,
            position: "relative",
          }}
        >
          {/* Resize Handle */}
          {rightOpen && (
            <div
              onMouseDown={() => setIsDragging(true)}
              style={{
                position: "absolute",
                left: -4,
                top: 0,
                bottom: 0,
                width: 8,
                cursor: "col-resize",
                zIndex: 10,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <div style={{ 
                width: 4, 
                height: 32, 
                background: isDragging ? "var(--primary)" : "transparent", 
                borderRadius: 4,
                transition: "background 0.2s"
              }} className="resize-indicator" />
            </div>
          )}

          <div style={{ width: rightOpen ? rightWidth : 0, height: "100%", display: "flex", flexDirection: "column", overflow: "hidden" }}>
            {/* Tab Switcher */}
            <div style={{ padding: "20px 24px 0", display: "flex", gap: 8, background: "var(--bg-app)", borderBottom: "1px solid var(--border-base)" }}>
              <button
                type="button"
                onClick={() => setGraphMode("session")}
                style={{
                  flex: 1,
                  padding: "12px 12px",
                  borderRadius: "var(--radius-md) var(--radius-md) 0 0",
                  border: "none",
                  borderBottom: graphMode === "session" ? "2px solid var(--primary)" : "2px solid transparent",
                  background: graphMode === "session" ? "var(--bg-panel)" : "transparent",
                  color: graphMode === "session" ? "var(--primary)" : "var(--text-muted)",
                  fontSize: 13,
                  fontWeight: 600,
                  transition: "none",
                }}
              >
                Knowledge Graph
              </button>
              <button
                type="button"
                onClick={() => setGraphMode("events")}
                style={{
                  flex: 1,
                  padding: "12px 12px",
                  borderRadius: "var(--radius-md) var(--radius-md) 0 0",
                  border: "none",
                  borderBottom: graphMode === "events" ? "2px solid var(--primary)" : "2px solid transparent",
                  background: graphMode === "events" ? "var(--bg-panel)" : "transparent",
                  color: graphMode === "events" ? "var(--primary)" : "var(--text-muted)",
                  fontSize: 13,
                  fontWeight: 600,
                  transition: "none",
                }}
              >
                Life Events
              </button>
            </div>
            
            <div style={{ flex: 1, minHeight: 0, position: "relative" }}>
              {graphMode === "session" ? (
                <FlowPanel
                  nodes={sessionNodes}
                  edges={sessionEdges}
                  title={`Entitas Terekam (hingga giliran ${turnIdx + 1})`}
                />
              ) : (
                <FlowPanel
                  nodes={eventNodes}
                  edges={eventEdges}
                  title="Causal Graph: user_events (40 node pertama)"
                />
              )}
            </div>
            
            {session?.summary && (
              <div style={{ padding: 24, borderTop: "1px solid var(--border-base)", background: "var(--bg-app)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                  <svg width="16" height="16" fill="none" stroke="var(--primary)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                  <strong style={{ color: "var(--text-main)", fontSize: 13 }}>Ringkasan Sesi</strong>
                </div>
                <div style={{ 
                  fontSize: 13, 
                  color: "var(--text-muted)", 
                  lineHeight: 1.6,
                  background: "var(--bg-panel)",
                  padding: 16,
                  borderRadius: "var(--radius-md)",
                  border: "1px solid var(--border-light)",
                  boxShadow: "var(--shadow-sm)",
                  maxHeight: 180,
                  overflowY: "auto"
                }}>
                  <ReactMarkdown
                    components={{
                      p: ({node, ...props}) => <p style={{ margin: "0 0 8px 0" }} {...props} />,
                      em: ({node, ...props}) => <em style={{ fontStyle: "italic", color: "var(--primary)" }} {...props} />,
                      strong: ({node, ...props}) => <strong style={{ fontWeight: 700, color: "var(--text-main)" }} {...props} />,
                    }}
                  >
                    {session.summary}
                  </ReactMarkdown>
                </div>
              </div>
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}
