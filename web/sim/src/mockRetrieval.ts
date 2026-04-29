import type { Session, Turn } from "./dataset";

const tokenize = (s: string) =>
  new Set(
    s
      .toLowerCase()
      .split(/\W+/)
      .filter((w) => w.length > 2)
  );

function jaccard(a: Set<string>, b: Set<string>): number {
  if (!a.size || !b.size) return 0;
  let inter = 0;
  for (const x of a) if (b.has(x)) inter++;
  const union = a.size + b.size - inter;
  return union ? inter / union : 0;
}

export interface RetrievalHit {
  kind: "turn" | "summary";
  score: number;
  preview: string;
  turnIndex?: number;
}

/** Client-side mock: token overlap against transcript so far (proxy for dense + keyword recall). */
export function mockRetrieve(query: string, session: Session, maxTurnInclusive: number, topK = 5): RetrievalHit[] {
  const q = tokenize(query);
  if (!q.size) return [];
  const hits: RetrievalHit[] = [];

  const turns = session.turns || [];
  for (let i = 0; i <= maxTurnInclusive && i < turns.length; i++) {
    const t: Turn = turns[i];
    const text = `${t.speaker}: ${t.text}`;
    const score = jaccard(q, tokenize(text));
    if (score > 0) hits.push({ kind: "turn", score, preview: text.slice(0, 220), turnIndex: i });
  }

  if (session.summary) {
    const s = jaccard(q, tokenize(session.summary));
    if (s > 0) hits.push({ kind: "summary", score: s * 0.85, preview: session.summary.slice(0, 280) });
  }

  hits.sort((a, b) => b.score - a.score);
  return hits.slice(0, topK);
}
