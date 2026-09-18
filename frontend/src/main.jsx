import { useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

function App() {
  const [document, setDocument] = useState("");
  const [question, setQuestion] = useState("");
  const [summary, setSummary] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function call(path, body, setter) {
    setBusy(true); setError("");
    try {
      const response = await fetch(`http://localhost:8000${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Request failed");
      setter(payload); if (payload.sources) setSources(payload.sources);
    } catch (requestError) { setError(requestError.message); } finally { setBusy(false); }
  }

  return <main><header><p className="eyebrow">LEGAL-GPT2 / RESEARCH WORKBENCH</p><h1>Read the record.<br /><em>Find the signal.</em></h1><p className="lede">Grounded summarization and document questions for legal research workflows.</p></header>
    <section className="workspace"><div className="input-panel"><label>Source document<textarea value={document} onChange={(event) => setDocument(event.target.value)} placeholder="Paste a legal document or extracted passage..." /></label><div className="actions"><button disabled={!document || busy} onClick={() => call("/summarize", { document }, (payload) => setSummary(payload.summary))}>Summarize</button><label className="question"><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about the record" /><button disabled={!document || !question || busy} onClick={() => call("/question", { document, question }, (payload) => setAnswer(payload.answer))}>Ask</button></label></div></div><div className="results"><article><p className="eyebrow">SUMMARY</p><p>{summary || "Your generated summary will appear here."}</p></article><article><p className="eyebrow">ANSWER</p><p>{answer || "Answers stay tied to the supplied document."}</p>{sources.map((source) => <blockquote key={`${source.document}-${source.chunk_id}`}>“{source.text}”<small>{source.document} · page {source.page} · chunk {source.chunk_id}</small></blockquote>)}</article></div></section>{busy && <p className="status">Working...</p>}{error && <p className="error">{error}</p>}<footer>Research and education only. Verify important claims against authoritative sources.</footer></main>;
}

createRoot(document.getElementById("root")).render(<App />);