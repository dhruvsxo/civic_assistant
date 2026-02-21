import { useState, useRef, useEffect } from "react";

// ── Color constants ─────────────────────────────────────────────────────────
const COLORS = {
  bg: "#0B1221",
  surface: "#131F35",
  card: "#192742",
  border: "#1E3158",
  accent: "#00C2A8",
  accentDim: "#00C2A820",
  amber: "#F59E0B",
  red: "#EF4444",
  green: "#22C55E",
  blue: "#3B82F6",
  text: "#E2E8F0",
  muted: "#64748B",
  white: "#FFFFFF",
};

const CATEGORY_ICONS = {
  POTHOLE: "🕳️",
  GARBAGE: "🗑️",
  WATER_LEAKAGE: "💧",
  ELECTRICITY: "⚡",
  STREETLIGHT: "🔦",
  SEWAGE: "🌊",
  TREE_FALL: "🌳",
  ROAD_DAMAGE: "🚧",
  NOISE: "📢",
  OTHER: "📋",
};

const PRIORITY_CONFIG = {
  CRITICAL: { color: "#EF4444", bg: "#EF444420", label: "Critical", dot: "🔴" },
  HIGH:     { color: "#F97316", bg: "#F9731620", label: "High",     dot: "🟠" },
  MEDIUM:   { color: "#F59E0B", bg: "#F59E0B20", label: "Medium",   dot: "🟡" },
  LOW:      { color: "#22C55E", bg: "#22C55E20", label: "Low",      dot: "🟢" },
};

const STATUS_CONFIG = {
  OPEN:        { color: "#3B82F6", label: "Open" },
  IN_PROGRESS: { color: "#F59E0B", label: "In Progress" },
  ESCALATED:   { color: "#EF4444", label: "Escalated" },
  RESOLVED:    { color: "#22C55E", label: "Resolved" },
  DUPLICATE:   { color: "#64748B", label: "Duplicate" },
};

// ── Sample complaint database (pre-seeded) ──────────────────────────────────
const SEED_COMPLAINTS = [
  { id: "A1B2C3D4", category: "POTHOLE", priority: "HIGH", status: "IN_PROGRESS", zone: "Koramangala", description: "Large pothole near 5th block signal causing accidents", created_at: "2026-02-16T09:30:00Z", assigned_to: "PWD", similar_count: 3 },
  { id: "E5F6G7H8", category: "GARBAGE", priority: "MEDIUM", status: "OPEN", zone: "Whitefield", description: "Garbage not collected for 3 days near ITPL main gate", created_at: "2026-02-17T08:00:00Z", assigned_to: "BBMP Sanitation", similar_count: 1 },
  { id: "I9J0K1L2", category: "ELECTRICITY", priority: "CRITICAL", status: "ESCALATED", zone: "Jayanagar", description: "Transformer sparking near 4th T Block, fire risk", created_at: "2026-02-17T11:00:00Z", assigned_to: "BESCOM", similar_count: 0 },
  { id: "M3N4O5P6", category: "WATER_LEAKAGE", priority: "HIGH", status: "IN_PROGRESS", zone: "HSR Layout", description: "Burst water main on Sector 2, road flooded", created_at: "2026-02-18T06:15:00Z", assigned_to: "BWSSB", similar_count: 2 },
  { id: "Q7R8S9T0", category: "STREETLIGHT", priority: "MEDIUM", status: "OPEN", zone: "Marathahalli", description: "7 streetlights out on ORR stretch", created_at: "2026-02-18T07:45:00Z", assigned_to: "BESCOM", similar_count: 0 },
  { id: "U1V2W3X4", category: "SEWAGE", priority: "HIGH", status: "OPEN", zone: "Koramangala", description: "Sewage overflow near 3rd block, stench unbearable", created_at: "2026-02-18T10:00:00Z", assigned_to: "BWSSB", similar_count: 4 },
  { id: "Y5Z6A7B8", category: "POTHOLE", priority: "MEDIUM", status: "RESOLVED", zone: "Indiranagar", description: "Pothole on 100 Feet Road filled", created_at: "2026-02-15T14:00:00Z", assigned_to: "PWD", similar_count: 0 },
  { id: "C9D0E1F2", category: "TREE_FALL", priority: "CRITICAL", status: "RESOLVED", zone: "Whitefield", description: "Tree fallen blocking Hope Farm junction", created_at: "2026-02-14T08:30:00Z", assigned_to: "Forest Dept", similar_count: 0 },
];

// ── LangGraph-style conversation state machine ─────────────────────────────
const GRAPH_STATES = {
  IDLE: "idle",
  INTAKE: "intake",
  CATEGORIZE: "categorize",
  DUPLICATE_CHECK: "duplicate_check",
  CONFIRM: "confirm",
  DONE: "done",
};

// ── Civic knowledge base for RAG simulation ─────────────────────────────────
const CIVIC_KB = [
  "BBMP SLA: Critical=4h, High=24h, Medium=72h, Low=7d",
  "Potholes >6 inches = HIGH priority. Highway potholes contact NHAI",
  "Burning garbage = CRITICAL, contact pollution control board",
  "Burst water mains affecting traffic = CRITICAL. BWSSB helpline: 1916",
  "Sparking electrical wires/transformer = CRITICAL. BESCOM helpline: 1912",
  "Fallen trees blocking roads = CRITICAL. Forest dept responds in 2-4 hours",
  "Duplicate complaints increase the priority of the original complaint",
  "Citizens can escalate if SLA is breached by 150%",
];

// ── LLM Provider Config ──────────────────────────────────────────────────────
// Toggle between Ollama (local) and Groq (cloud) in the UI
// Groq free key: https://console.groq.com

const LLM_PROVIDERS = {
  ollama: {
    name: "Ollama",
    model: "llama3.2:3b",
    label: "🦙 Llama 3.2 3B",
    sublabel: "Local • Free • Offline",
    color: "#00C2A8",
  },
  groq: {
    name: "Groq",
    model: "llama-3.3-70b-versatile",
    label: "⚡ Llama 3.3 70B",
    sublabel: "Groq Cloud • Free Tier • Fast",
    color: "#F97316",
  },
};

async function callLLM(provider, groqKey, systemPrompt, userMessage, history = []) {
  if (provider === "groq") {
    // ── Groq API ─────────────────────────────────────────────
    if (!groqKey) throw new Error("Groq API key not set. Get one free at https://console.groq.com");
    const messages = [
      { role: "system", content: systemPrompt },
      ...history.map(m => ({ role: m.role, content: m.content })),
      { role: "user", content: userMessage },
    ];
    const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${groqKey}`,
      },
      body: JSON.stringify({
        model: LLM_PROVIDERS.groq.model,
        messages,
        temperature: 0.3,
        max_tokens: 1000,
      }),
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error?.message || "Groq API error");
    }
    const data = await response.json();
    return data.choices?.[0]?.message?.content || "No response from Groq.";

  } else {
    // ── Ollama API ────────────────────────────────────────────
    const messages = [
      { role: "system", content: systemPrompt },
      ...history.map(m => ({ role: m.role, content: m.content })),
      { role: "user", content: userMessage },
    ];
    const response = await fetch("http://localhost:11434/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: LLM_PROVIDERS.ollama.model,
        messages,
        stream: false,
        options: { temperature: 0.3 },
      }),
    });
    if (!response.ok) throw new Error("Ollama not reachable. Is `ollama serve` running?");
    const data = await response.json();
    return data.message?.content || "No response from Ollama.";
  }
}

// ── Civic Bot System Prompt ──────────────────────────────────────────────────
function buildSystemPrompt(graphState, complaints, ragContext) {
  const openComplaints = complaints.filter(c => c.status !== "RESOLVED" && c.status !== "DUPLICATE");
  const zones = [...new Set(openComplaints.map(c => c.zone))];

  return `You are CivicBot, an AI assistant for Bengaluru Municipal Corporation (BBMP) Smart City Initiative.

Current graph state: ${graphState}
Active complaints in system: ${openComplaints.length}
Affected zones: ${zones.join(", ")}

Relevant civic policies (RAG context):
${ragContext}

Your role based on state:
- INTAKE: Greet user, understand if they want to REPORT an issue, TRACK a complaint, or get INFO. Ask for details.
- CATEGORIZE: Extract category (POTHOLE/GARBAGE/WATER_LEAKAGE/ELECTRICITY/STREETLIGHT/SEWAGE/TREE_FALL/OTHER), location/zone, description.
- DUPLICATE_CHECK: If you detect a likely duplicate based on zone+category, warn the user.
- CONFIRM: Summarize what you've understood and confirm with the user.
- DONE: Provide the complaint ID and next steps.

Rules:
- Be empathetic, professional, and concise (2-4 sentences max per response)
- Always try to extract: category, zone/location, description severity
- If reporting: guide through a natural conversation to get all details
- For tracking requests: ask for their complaint ID
- ALWAYS respond in JSON format:
{
  "reply": "your response to citizen",
  "next_state": "intake|categorize|duplicate_check|confirm|done",
  "extracted": {
    "category": "POTHOLE|GARBAGE|WATER_LEAKAGE|ELECTRICITY|STREETLIGHT|SEWAGE|TREE_FALL|OTHER|null",
    "zone": "area name or null",
    "description": "cleaned description or null",
    "priority": "CRITICAL|HIGH|MEDIUM|LOW|null",
    "intent": "report|track|info|null"
  },
  "suggestions": ["Quick reply 1", "Quick reply 2"]
}`;
}

// ── Priority computation ─────────────────────────────────────────────────────
function computePriority(category, description) {
  const desc = description?.toLowerCase() || "";
  const rules = {
    ELECTRICITY: { default: "HIGH", kw: { fire: "CRITICAL", spark: "CRITICAL", shock: "CRITICAL" } },
    WATER_LEAKAGE: { default: "HIGH", kw: { flood: "CRITICAL", burst: "CRITICAL" } },
    SEWAGE: { default: "HIGH", kw: { overflow: "CRITICAL" } },
    TREE_FALL: { default: "CRITICAL", kw: {} },
    POTHOLE: { default: "MEDIUM", kw: { deep: "HIGH", accident: "CRITICAL", highway: "HIGH" } },
    GARBAGE: { default: "LOW", kw: { burning: "CRITICAL", hospital: "HIGH", school: "HIGH" } },
    STREETLIGHT: { default: "MEDIUM", kw: { dark: "HIGH", highway: "HIGH" } },
  };
  const r = rules[category];
  if (!r) return "MEDIUM";
  for (const [kw, p] of Object.entries(r.kw)) {
    if (desc.includes(kw)) return p;
  }
  return r.default;
}

function generateId() {
  return Math.random().toString(36).substr(2, 8).toUpperCase();
}

function slaText(priority) {
  return { CRITICAL: "4 hours", HIGH: "24 hours", MEDIUM: "3 days", LOW: "7 days" }[priority] || "72 hours";
}

function deptFor(category) {
  return {
    POTHOLE: "Public Works Department",
    GARBAGE: "BBMP Sanitation",
    WATER_LEAKAGE: "BWSSB Water Supply",
    ELECTRICITY: "BESCOM Electrical",
    STREETLIGHT: "BESCOM Lighting",
    SEWAGE: "BWSSB Drainage",
    TREE_FALL: "Forest & Horticulture",
    OTHER: "General Services",
  }[category] || "General Services";
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab] = useState("citizen");
  const [complaints, setComplaints] = useState();
  const [llmProvider, setLlmProvider] = useState("ollama");
  const [groqKey, setGroqKey] = useState("");
  const [showGroqInput, setShowGroqInput] = useState(false);

  // Citizen chat state
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [graphState, setGraphState] = useState(GRAPH_STATES.IDLE);
  const [extracted, setExtracted] = useState({});
  const [suggestions, setSuggestions] = useState(["🕳️ Report a pothole", "🗑️ Garbage issue", "Track my complaint"]);
  const [trackId, setTrackId] = useState("");
  const chatRef = useRef(null);

  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages]);

  // Boot greeting
  useEffect(() => {
    setMessages([{
      role: "assistant",
      content: "👋 Welcome to **CivicBot** — Bengaluru's Smart City Assistant!\n\nI can help you:\n• 📝 Report civic issues (potholes, garbage, water leaks, electricity)\n• 🔍 Track your complaint status\n• ℹ️ Get information about services\n\nWhat can I help you with today?",
    }]);
  }, []);

  const sendMessage = async (text) => {
    const msg = text || input.trim();
    if (!msg || loading) return;
    setInput("");

    const userMsg = { role: "user", content: msg };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setSuggestions([]);
    setLoading(true);

    try {
      const ragContext = CIVIC_KB.slice(0, 4).join("\n");
      const currentState = graphState === GRAPH_STATES.IDLE ? GRAPH_STATES.INTAKE : graphState;
      const systemPrompt = buildSystemPrompt(currentState, complaints, ragContext);
      
      const history = messages.slice(-8);
      const raw = await callLLM(llmProvider, groqKey, systemPrompt, msg, history);

      let parsed;
      try {
        const clean = raw.replace(/```json\n?|\n?```/g, "").trim();
        parsed = JSON.parse(clean);
      } catch {
        parsed = { reply: raw, next_state: currentState, extracted: {}, suggestions: [] };
      }

      const nextState = parsed.next_state || currentState;
      const newExtracted = { ...extracted, ...(parsed.extracted || {}) };
      setExtracted(newExtracted);
      setGraphState(nextState);
      setSuggestions(parsed.suggestions || []);

      let reply = parsed.reply || raw;

      // If state is "done" and we have enough data, create complaint
      if (nextState === "done" && newExtracted.category && newExtracted.description) {
        const category = newExtracted.category;
        const zone = newExtracted.zone || "Bengaluru";
        const description = newExtracted.description;
        const priority = computePriority(category, description);
        const id = generateId();
        const dept = deptFor(category);
        const sla = slaText(priority);

        // Duplicate check
        const potentialDup = complaints.find(c =>
          c.category === category &&
          c.zone === zone &&
          c.status !== "RESOLVED" &&
          c.status !== "DUPLICATE"
        );

        if (potentialDup) {
          const updated = complaints.map(c =>
            c.id === potentialDup.id ? { ...c, similar_count: c.similar_count + 1 } : c
          );
          setComplaints(updated);
          reply = `🔍 **Duplicate Detected!**\n\nWe found an existing complaint (#${potentialDup.id}) for a similar **${category.replace("_", " ")}** issue in **${zone}**.\n\nYour report has been linked — this increases its priority! Track it with ID: \`${potentialDup.id}\`\n\nCurrent status: **${potentialDup.status}**`;
          setGraphState(GRAPH_STATES.IDLE);
          setExtracted({});
          setSuggestions([`Track #${potentialDup.id}`, "Report different issue"]);
        } else {
          const newComplaint = {
            id,
            category,
            priority,
            status: "OPEN",
            zone,
            description,
            created_at: new Date().toISOString(),
            assigned_to: dept,
            similar_count: 0,
          };
          setComplaints(prev => [newComplaint, ...prev]);
          reply = `✅ **Complaint Registered!**\n\n📋 ID: \`${id}\`\n🏷️ Category: ${CATEGORY_ICONS[category]} ${category.replace("_"," ")}\n${PRIORITY_CONFIG[priority].dot} Priority: **${priority}**\n🏢 Assigned to: ${dept}\n⏱️ Estimated resolution: **${sla}**\n\nSave your ID **${id}** to track status anytime!`;
          setGraphState(GRAPH_STATES.IDLE);
          setExtracted({});
          setSuggestions(["Report another issue", "Go to Officer Dashboard"]);
        }
      }

      setMessages([...newMessages, { role: "assistant", content: reply }]);
    } catch (err) {
      setMessages([...newMessages, {
        role: "assistant",
        content: "⚠️ Sorry, I encountered an error. Please try again.",
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleTrack = () => {
    const c = complaints.find(x => x.id.toUpperCase() === trackId.toUpperCase());
    if (c) {
      const cfg = PRIORITY_CONFIG[c.priority];
      const sCfg = STATUS_CONFIG[c.status] || {};
      const msg = `📋 **Complaint #${c.id}**\n\n${CATEGORY_ICONS[c.category]} **${c.category.replace("_"," ")}** — ${c.zone}\n${cfg.dot} Priority: **${c.priority}**\n🔵 Status: **${c.status}**\n🏢 Assigned: ${c.assigned_to}\n📅 Reported: ${new Date(c.created_at).toLocaleDateString()}\n\n${c.description}`;
      setMessages(prev => [...prev, { role: "user", content: `Track #${trackId}` }, { role: "assistant", content: msg }]);
      setTrackId("");
    } else {
      setMessages(prev => [...prev, { role: "user", content: `Track #${trackId}` }, { role: "assistant", content: `❌ Complaint ID **${trackId}** not found. Please check the ID and try again.` }]);
      setTrackId("");
    }
  };

  // Dashboard stats
  const stats = {
    total: complaints.length,
    open: complaints.filter(c => c.status === "OPEN").length,
    inProgress: complaints.filter(c => c.status === "IN_PROGRESS").length,
    escalated: complaints.filter(c => c.status === "ESCALATED").length,
    resolved: complaints.filter(c => c.status === "RESOLVED").length,
  };

  const hotZones = Object.entries(
    complaints.filter(c => c.status !== "RESOLVED").reduce((acc, c) => {
      acc[c.zone] = (acc[c.zone] || 0) + 1;
      return acc;
    }, {})
  ).sort((a, b) => b[1] - a[1]).slice(0, 5);

  const sorted = [...complaints].sort((a, b) => {
    const p = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
    return p[a.priority] - p[b.priority];
  });

  const updateStatus = (id, newStatus) => {
    setComplaints(prev => prev.map(c =>
      c.id === id ? { ...c, status: newStatus } : c
    ));
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={{
      fontFamily: "'DM Sans', 'Segoe UI', sans-serif",
      background: COLORS.bg,
      minHeight: "100vh",
      color: COLORS.text,
      display: "flex",
      flexDirection: "column",
    }}>
      {/* Header */}
      <header style={{
        background: `linear-gradient(135deg, ${COLORS.surface} 0%, #0D1B31 100%)`,
        borderBottom: `1px solid ${COLORS.border}`,
        padding: "0 24px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        height: 64,
        flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 40, height: 40, borderRadius: 10,
            background: `linear-gradient(135deg, ${COLORS.accent}, #0091FF)`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 20, fontWeight: 900,
          }}>🏙️</div>
          <div>
            <div style={{ fontWeight: 800, fontSize: 16, letterSpacing: "-0.3px" }}>
              Smart City <span style={{ color: COLORS.accent }}>CivicBot</span>
            </div>
            <div style={{ fontSize: 11, color: COLORS.muted, marginTop: 1 }}>
              Bengaluru Municipal Corporation • Powered by Claude + LangGraph + RAG
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {/* LLM Provider Toggle */}
          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 4 }}>
            <div style={{ display: "flex", gap: 6, background: COLORS.card, borderRadius: 10, padding: 4, border: `1px solid ${COLORS.border}` }}>
              {Object.entries(LLM_PROVIDERS).map(([key, cfg]) => (
                <button key={key} onClick={() => { setLlmProvider(key); setShowGroqInput(key === "groq"); }} style={{
                  padding: "5px 14px", borderRadius: 7, border: "none", cursor: "pointer",
                  fontWeight: 600, fontSize: 12, transition: "all 0.2s",
                  background: llmProvider === key ? cfg.color : "transparent",
                  color: llmProvider === key ? "#000" : COLORS.muted,
                }}>
                  {cfg.label}
                </button>
              ))}
            </div>
            <div style={{ fontSize: 10, color: COLORS.muted }}>
              {LLM_PROVIDERS[llmProvider].sublabel}
            </div>
          </div>

          {/* Groq API Key input */}
          {showGroqInput && (
            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <input
                type="password"
                value={groqKey}
                onChange={e => setGroqKey(e.target.value)}
                placeholder="gsk_... (Groq API key)"
                style={{
                  background: COLORS.card, border: `1px solid ${groqKey ? COLORS.accent : COLORS.amber}`,
                  borderRadius: 8, color: COLORS.text, padding: "6px 12px",
                  fontSize: 12, outline: "none", width: 200,
                }}
              />
              {!groqKey && (
                <a href="https://console.groq.com" target="_blank" rel="noreferrer"
                  style={{ fontSize: 11, color: COLORS.amber, textDecoration: "none" }}>
                  Get free key →
                </a>
              )}
              {groqKey && <span style={{ color: COLORS.green, fontSize: 16 }}>✓</span>}
            </div>
          )}

          <div style={{ display: "flex", gap: 6, background: COLORS.card, borderRadius: 10, padding: 4, border: `1px solid ${COLORS.border}` }}>
            {["citizen", "officer"].map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)} style={{
                padding: "7px 18px", borderRadius: 7, border: "none", cursor: "pointer",
                fontWeight: 600, fontSize: 13, transition: "all 0.2s",
                background: activeTab === tab ? COLORS.accent : "transparent",
                color: activeTab === tab ? "#000" : COLORS.muted,
              }}>
                {tab === "citizen" ? "👤 Citizen" : "🏛️ Officer Dashboard"}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div style={{ flex: 1, overflow: "hidden", display: "flex" }}>

        {/* ── CITIZEN VIEW ── */}
        {activeTab === "citizen" && (
          <div style={{ flex: 1, display: "flex", gap: 0 }}>

            {/* Chat */}
            <div style={{ flex: 1, display: "flex", flexDirection: "column", borderRight: `1px solid ${COLORS.border}` }}>
              {/* Graph state indicator */}
              <div style={{
                padding: "10px 20px",
                background: COLORS.surface,
                borderBottom: `1px solid ${COLORS.border}`,
                display: "flex", alignItems: "center", gap: 10, fontSize: 12,
              }}>
                <span style={{ color: COLORS.muted }}>Workflow:</span>
                {["intake","categorize","duplicate_check","confirm","done"].map((s, i) => (
                  <span key={s} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    {i > 0 && <span style={{ color: COLORS.border }}>→</span>}
                    <span style={{
                      padding: "2px 10px", borderRadius: 20,
                      background: graphState === s ? COLORS.accentDim : "transparent",
                      color: graphState === s ? COLORS.accent : COLORS.muted,
                      fontWeight: graphState === s ? 700 : 400,
                      border: `1px solid ${graphState === s ? COLORS.accent : "transparent"}`,
                      textTransform: "capitalize",
                    }}>{s.replace("_"," ")}</span>
                  </span>
                ))}
                {Object.values(extracted).some(Boolean) && (
                  <span style={{ marginLeft: "auto", color: COLORS.accent, fontSize: 11 }}>
                    {extracted.category && `📌 ${extracted.category} `}
                    {extracted.zone && `📍 ${extracted.zone}`}
                  </span>
                )}
              </div>

              {/* Messages */}
              <div ref={chatRef} style={{
                flex: 1, overflowY: "auto", padding: "20px",
                display: "flex", flexDirection: "column", gap: 14,
              }}>
                {messages.map((m, i) => (
                  <div key={i} style={{
                    display: "flex",
                    justifyContent: m.role === "user" ? "flex-end" : "flex-start",
                    alignItems: "flex-end", gap: 8,
                  }}>
                    {m.role === "assistant" && (
                      <div style={{
                        width: 30, height: 30, borderRadius: 8, flexShrink: 0,
                        background: `linear-gradient(135deg, ${COLORS.accent}, #0091FF)`,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 14,
                      }}>🤖</div>
                    )}
                    <div style={{
                      maxWidth: "78%",
                      padding: "12px 16px",
                      borderRadius: m.role === "user" ? "16px 4px 16px 16px" : "4px 16px 16px 16px",
                      background: m.role === "user"
                        ? `linear-gradient(135deg, ${COLORS.accent}22, ${COLORS.blue}22)`
                        : COLORS.card,
                      border: `1px solid ${m.role === "user" ? COLORS.accent + "44" : COLORS.border}`,
                      fontSize: 14, lineHeight: 1.6,
                      whiteSpace: "pre-line",
                    }}>
                      {m.content.split(/\*\*(.+?)\*\*/g).map((part, pi) =>
                        pi % 2 === 1
                          ? <strong key={pi} style={{ color: COLORS.accent }}>{part}</strong>
                          : part.split(/`(.+?)`/g).map((p2, pj) =>
                              pj % 2 === 1
                                ? <code key={pj} style={{ background: COLORS.surface, padding: "2px 6px", borderRadius: 4, fontSize: 12, color: COLORS.amber }}>{p2}</code>
                                : p2
                            )
                      )}
                    </div>
                  </div>
                ))}
                {loading && (
                  <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
                    <div style={{
                      width: 30, height: 30, borderRadius: 8,
                      background: `linear-gradient(135deg, ${COLORS.accent}, #0091FF)`,
                      display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14,
                    }}>🤖</div>
                    <div style={{
                      padding: "12px 16px", borderRadius: "4px 16px 16px 16px",
                      background: COLORS.card, border: `1px solid ${COLORS.border}`,
                      display: "flex", gap: 5, alignItems: "center",
                    }}>
                      {[0, 0.2, 0.4].map((d, i) => (
                        <div key={i} style={{
                          width: 7, height: 7, borderRadius: "50%", background: COLORS.accent,
                          animation: "pulse 1.2s infinite",
                          animationDelay: `${d}s`, opacity: 0.7,
                        }} />
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Suggestions */}
              {suggestions.length > 0 && !loading && (
                <div style={{ padding: "0 16px 12px", display: "flex", flexWrap: "wrap", gap: 8 }}>
                  {suggestions.map((s, i) => (
                    <button key={i} onClick={() => sendMessage(s)} style={{
                      padding: "7px 14px", borderRadius: 20, border: `1px solid ${COLORS.border}`,
                      background: COLORS.card, color: COLORS.text, cursor: "pointer",
                      fontSize: 13, fontWeight: 500, transition: "all 0.2s",
                    }}
                    onMouseEnter={e => e.target.style.borderColor = COLORS.accent}
                    onMouseLeave={e => e.target.style.borderColor = COLORS.border}>
                      {s}
                    </button>
                  ))}
                </div>
              )}

              {/* Input */}
              <div style={{
                padding: "14px 16px",
                borderTop: `1px solid ${COLORS.border}`,
                background: COLORS.surface,
                display: "flex", gap: 10, alignItems: "flex-end",
              }}>
                <textarea
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }}}
                  placeholder="Describe your civic issue (e.g. 'Large pothole on MG Road near Garuda Mall')"
                  rows={2}
                  style={{
                    flex: 1, background: COLORS.card, border: `1px solid ${COLORS.border}`,
                    borderRadius: 10, color: COLORS.text, padding: "10px 14px",
                    fontSize: 14, resize: "none", outline: "none", fontFamily: "inherit",
                    lineHeight: 1.5,
                  }}
                />
                <button onClick={() => sendMessage()} disabled={loading || !input.trim()} style={{
                  width: 46, height: 46, borderRadius: 10, border: "none",
                  background: input.trim() && !loading ? COLORS.accent : COLORS.border,
                  color: input.trim() && !loading ? "#000" : COLORS.muted,
                  cursor: input.trim() && !loading ? "pointer" : "default",
                  fontSize: 20, transition: "all 0.2s", flexShrink: 0,
                }}>↑</button>
              </div>
            </div>

            {/* Sidebar — Track + Recent */}
            <div style={{ width: 300, display: "flex", flexDirection: "column", background: COLORS.surface }}>
              <div style={{ padding: 16, borderBottom: `1px solid ${COLORS.border}` }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 10, color: COLORS.muted }}>TRACK COMPLAINT</div>
                <div style={{ display: "flex", gap: 8 }}>
                  <input
                    value={trackId}
                    onChange={e => setTrackId(e.target.value.toUpperCase())}
                    onKeyDown={e => e.key === "Enter" && handleTrack()}
                    placeholder="Enter ID e.g. A1B2C3D4"
                    style={{
                      flex: 1, background: COLORS.card, border: `1px solid ${COLORS.border}`,
                      borderRadius: 8, color: COLORS.text, padding: "8px 12px",
                      fontSize: 13, outline: "none", fontFamily: "monospace",
                    }}
                  />
                  <button onClick={handleTrack} style={{
                    padding: "8px 12px", background: COLORS.accent, border: "none",
                    borderRadius: 8, cursor: "pointer", fontWeight: 700, fontSize: 13, color: "#000",
                  }}>→</button>
                </div>
              </div>
              <div style={{ flex: 1, overflowY: "auto", padding: 12 }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 10, color: COLORS.muted, padding: "4px 4px" }}>RECENT COMPLAINTS</div>
                {complaints.slice(0, 8).map(c => {
                  const pc = PRIORITY_CONFIG[c.priority];
                  const sc = STATUS_CONFIG[c.status] || {};
                  return (
                    <div key={c.id} style={{
                      background: COLORS.card, borderRadius: 10, padding: "10px 12px",
                      marginBottom: 8, border: `1px solid ${COLORS.border}`, cursor: "pointer",
                    }}
                    onClick={() => {
                      const msg = `Track #${c.id}`;
                      setMessages(prev => [...prev,
                        { role: "user", content: msg },
                        { role: "assistant", content: `📋 **#${c.id}** — ${CATEGORY_ICONS[c.category]} ${c.category.replace("_"," ")} in ${c.zone}\n${pc.dot} ${c.priority} | 🔵 ${c.status}\n🏢 ${c.assigned_to}\n${c.description}` }
                      ]);
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 6 }}>
                        <span style={{ fontSize: 13, fontWeight: 700 }}>
                          {CATEGORY_ICONS[c.category]} {c.category.replace("_"," ")}
                        </span>
                        <span style={{
                          fontSize: 10, padding: "2px 8px", borderRadius: 20,
                          background: pc.bg, color: pc.color, fontWeight: 700,
                        }}>{c.priority}</span>
                      </div>
                      <div style={{ fontSize: 11, color: COLORS.muted, marginBottom: 4 }}>📍 {c.zone}</div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span style={{ fontSize: 10, fontFamily: "monospace", color: COLORS.amber }}>#{c.id}</span>
                        <span style={{ fontSize: 10, color: sc.color }}>{c.status.replace("_"," ")}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* ── OFFICER DASHBOARD ── */}
        {activeTab === "officer" && (
          <div style={{ flex: 1, overflowY: "auto", padding: 24, display: "flex", flexDirection: "column", gap: 20 }}>

            {/* Stats Row */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 14 }}>
              {[
                { label: "Total", val: stats.total, color: COLORS.blue, icon: "📋" },
                { label: "Open", val: stats.open, color: COLORS.blue, icon: "🔓" },
                { label: "In Progress", val: stats.inProgress, color: COLORS.amber, icon: "⚙️" },
                { label: "Escalated", val: stats.escalated, color: COLORS.red, icon: "🚨" },
                { label: "Resolved", val: stats.resolved, color: COLORS.green, icon: "✅" },
              ].map(s => (
                <div key={s.label} style={{
                  background: COLORS.card, borderRadius: 14, padding: "16px 18px",
                  border: `1px solid ${COLORS.border}`,
                }}>
                  <div style={{ fontSize: 22, marginBottom: 6 }}>{s.icon}</div>
                  <div style={{ fontSize: 28, fontWeight: 800, color: s.color }}>{s.val}</div>
                  <div style={{ fontSize: 12, color: COLORS.muted, marginTop: 2 }}>{s.label}</div>
                </div>
              ))}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: 20 }}>

              {/* Priority Queue */}
              <div style={{ background: COLORS.card, borderRadius: 14, border: `1px solid ${COLORS.border}`, overflow: "hidden" }}>
                <div style={{
                  padding: "16px 20px", borderBottom: `1px solid ${COLORS.border}`,
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                }}>
                  <span style={{ fontWeight: 700, fontSize: 15 }}>🚨 Priority Queue</span>
                  <span style={{ fontSize: 12, color: COLORS.muted }}>{sorted.filter(c => c.status !== "RESOLVED").length} active</span>
                </div>
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                    <thead>
                      <tr style={{ background: COLORS.surface }}>
                        {["ID", "Category", "Zone", "Priority", "Status", "Dept", "Reports", "Action"].map(h => (
                          <th key={h} style={{ padding: "10px 14px", textAlign: "left", color: COLORS.muted, fontWeight: 600, fontSize: 11, whiteSpace: "nowrap" }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {sorted.filter(c => c.status !== "RESOLVED").map(c => {
                        const pc = PRIORITY_CONFIG[c.priority];
                        const sc = STATUS_CONFIG[c.status] || {};
                        return (
                          <tr key={c.id} style={{ borderTop: `1px solid ${COLORS.border}` }}>
                            <td style={{ padding: "10px 14px", fontFamily: "monospace", color: COLORS.amber, fontSize: 11 }}>{c.id}</td>
                            <td style={{ padding: "10px 14px", whiteSpace: "nowrap" }}>{CATEGORY_ICONS[c.category]} {c.category.replace("_"," ")}</td>
                            <td style={{ padding: "10px 14px", color: COLORS.muted, fontSize: 12 }}>{c.zone}</td>
                            <td style={{ padding: "10px 14px" }}>
                              <span style={{
                                padding: "3px 10px", borderRadius: 20, fontSize: 11,
                                background: pc.bg, color: pc.color, fontWeight: 700,
                              }}>{pc.dot} {c.priority}</span>
                            </td>
                            <td style={{ padding: "10px 14px" }}>
                              <span style={{ color: sc.color, fontSize: 12, fontWeight: 600 }}>{c.status.replace("_"," ")}</span>
                            </td>
                            <td style={{ padding: "10px 14px", color: COLORS.muted, fontSize: 11 }}>{c.assigned_to}</td>
                            <td style={{ padding: "10px 14px", textAlign: "center" }}>
                              {c.similar_count > 0 && (
                                <span style={{ background: "#EF444420", color: "#EF4444", padding: "2px 8px", borderRadius: 20, fontSize: 11, fontWeight: 700 }}>
                                  +{c.similar_count}
                                </span>
                              )}
                            </td>
                            <td style={{ padding: "10px 14px" }}>
                              <select
                                value={c.status}
                                onChange={e => updateStatus(c.id, e.target.value)}
                                style={{
                                  background: COLORS.surface, border: `1px solid ${COLORS.border}`,
                                  color: COLORS.text, borderRadius: 6, padding: "4px 8px",
                                  fontSize: 11, cursor: "pointer",
                                }}>
                                {Object.keys(STATUS_CONFIG).map(s => <option key={s} value={s}>{s.replace("_"," ")}</option>)}
                              </select>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Hot Zones + Category Breakdown */}
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                {/* Hot Zones */}
                <div style={{ background: COLORS.card, borderRadius: 14, border: `1px solid ${COLORS.border}`, overflow: "hidden" }}>
                  <div style={{ padding: "14px 18px", borderBottom: `1px solid ${COLORS.border}`, fontWeight: 700, fontSize: 14 }}>
                    🔥 Hot Zones
                  </div>
                  <div style={{ padding: "12px 18px", display: "flex", flexDirection: "column", gap: 10 }}>
                    {hotZones.map(([zone, count]) => {
                      const max = hotZones[0]?.[1] || 1;
                      const pct = (count / max) * 100;
                      return (
                        <div key={zone}>
                          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 4 }}>
                            <span>📍 {zone}</span>
                            <span style={{ color: count >= 3 ? COLORS.red : COLORS.amber, fontWeight: 700 }}>{count} issue{count !== 1 ? "s" : ""}</span>
                          </div>
                          <div style={{ height: 6, background: COLORS.border, borderRadius: 3, overflow: "hidden" }}>
                            <div style={{
                              height: "100%", width: `${pct}%`, borderRadius: 3,
                              background: count >= 3
                                ? `linear-gradient(90deg, ${COLORS.red}, ${COLORS.amber})`
                                : `linear-gradient(90deg, ${COLORS.accent}, ${COLORS.blue})`,
                            }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Category Breakdown */}
                <div style={{ background: COLORS.card, borderRadius: 14, border: `1px solid ${COLORS.border}`, overflow: "hidden" }}>
                  <div style={{ padding: "14px 18px", borderBottom: `1px solid ${COLORS.border}`, fontWeight: 700, fontSize: 14 }}>
                    📊 Category Breakdown
                  </div>
                  <div style={{ padding: "12px 18px", display: "flex", flexDirection: "column", gap: 8 }}>
                    {Object.entries(
                      complaints.reduce((acc, c) => { acc[c.category] = (acc[c.category] || 0) + 1; return acc; }, {})
                    ).sort((a,b) => b[1]-a[1]).map(([cat, count]) => (
                      <div key={cat} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span style={{ fontSize: 13 }}>{CATEGORY_ICONS[cat]} {cat.replace("_"," ")}</span>
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <div style={{ width: 60, height: 4, background: COLORS.border, borderRadius: 2, overflow: "hidden" }}>
                            <div style={{ height: "100%", width: `${(count/complaints.length)*100}%`, background: COLORS.accent, borderRadius: 2 }} />
                          </div>
                          <span style={{ fontSize: 12, color: COLORS.muted, width: 16, textAlign: "right" }}>{count}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* RAG Context Card */}
                <div style={{ background: COLORS.card, borderRadius: 14, border: `1px solid ${COLORS.border}`, overflow: "hidden" }}>
                  <div style={{ padding: "14px 18px", borderBottom: `1px solid ${COLORS.border}`, fontWeight: 700, fontSize: 14 }}>
                    🧠 RAG Intelligence
                  </div>
                  <div style={{ padding: "12px 18px" }}>
                    <div style={{ fontSize: 12, color: COLORS.muted, lineHeight: 1.7 }}>
                      <div style={{ marginBottom: 6, color: COLORS.accent, fontWeight: 600 }}>Active Knowledge Sources:</div>
                      <div>📚 BBMP Policy Guidelines</div>
                      <div>⚖️ SLA Enforcement Rules</div>
                      <div>🗺️ Zone-based Priority Logic</div>
                      <div>🔍 Semantic Duplicate Detection</div>
                      <div style={{ marginTop: 8, padding: "6px 10px", background: COLORS.surface, borderRadius: 6 }}>
                        <span style={{ color: COLORS.green }}>✓</span> LangGraph Workflow Active
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Architecture Banner */}
      <div style={{
        background: COLORS.surface,
        borderTop: `1px solid ${COLORS.border}`,
        padding: "8px 24px",
        display: "flex", gap: 24, alignItems: "center",
        fontSize: 11, color: COLORS.muted,
      }}>
        <span style={{ color: COLORS.accent, fontWeight: 700 }}>Architecture:</span>
        {[
          ["🦙", "Ollama (Llama 3.2 3B)", llmProvider === "ollama" ? COLORS.accent : COLORS.muted],
          ["⚡", "Groq (Llama 3.3 70B)", llmProvider === "groq" ? "#F97316" : COLORS.muted],
          ["🔗", "LangChain", "#8B5CF6"],
          ["🔄", "LangGraph State Machine", "#3B82F6"],
          ["🧠", "RAG + FAISS", "#F59E0B"],
          ["⚡", "FastAPI", "#22C55E"],
        ].map(([icon, label, color]) => (
          <span key={label} style={{ display: "flex", gap: 5, alignItems: "center" }}>
            <span>{icon}</span>
            <span style={{ color }}>{label}</span>
          </span>
        ))}
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.3; transform: scale(0.8); }
          50% { opacity: 1; transform: scale(1.2); }
        }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #1E3158; border-radius: 2px; }
        textarea:focus { border-color: #00C2A8 !important; }
        input:focus { border-color: #00C2A8 !important; outline: none; }
      `}</style>
    </div>
  );
}
