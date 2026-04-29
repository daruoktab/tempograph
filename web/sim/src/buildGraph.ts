import type { Edge, Node } from "@xyflow/react";
import type { LifeEvent, SecondaryPersona, Session, UserBlock } from "./dataset";

const slug = (s: string) =>
  s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 48) || "x";

/** Entities from ground_truths up to and including turn index (matches turn_id). */
export function collectEntitiesUpToTurn(
  session: Session,
  maxTurnId: number
): Map<string, { type: string; mentions: number }> {
  const m = new Map<string, { type: string; mentions: number }>();
  const gts = session.ground_truths || [];
  for (const gt of gts) {
    if (gt.turn_id > maxTurnId) break;
    for (const e of gt.entities_mentioned || []) {
      const name = (e.name || "").trim();
      if (name.length < 2) continue;
      const prev = m.get(name);
      if (prev) prev.mentions += 1;
      else m.set(name, { type: e.type || "entity", mentions: 1 });
    }
  }
  return m;
}

export function buildSessionKnowledgeGraph(
  session: Session,
  maxTurnId: number,
  user: UserBlock,
  secondaries: SecondaryPersona[]
): { nodes: Node[]; edges: Edge[] } {
  const cx = 420;
  const cy = 340;
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  nodes.push({
    id: "user",
    position: { x: cx - 70, y: cy - 24 },
    data: { label: user.name, sub: "User" },
    type: "pill",
  });

  const secs = secondaries.slice(0, 8);
  secs.forEach((sp, i) => {
    const angle = -Math.PI / 2 + (i / Math.max(secs.length, 1)) * Math.PI * 2 * 0.85;
    const r = 160;
    const id = `sec-${slug(sp.name)}`;
    nodes.push({
      id,
      position: { x: cx + Math.cos(angle) * r - 60, y: cy + Math.sin(angle) * r - 20 },
      data: { label: sp.name, sub: sp.relationship },
      type: "pill",
    });
    edges.push({
      id: `e-user-${id}`,
      source: "user",
      target: id,
      label: "persona",
      animated: true,
      style: { stroke: "#8b5cf6", strokeWidth: 2 },
    });
  });

  const ents = [...collectEntitiesUpToTurn(session, maxTurnId).entries()]
    .sort((a, b) => b[1].mentions - a[1].mentions)
    .slice(0, 24);

  ents.forEach(([name, meta], i) => {
    const angle = -Math.PI / 2 + (i / Math.max(ents.length, 1)) * Math.PI * 2;
    const r = 280;
    const id = `ent-${slug(name)}`;
    nodes.push({
      id,
      position: { x: cx + Math.cos(angle) * r - 70, y: cy + Math.sin(angle) * r - 22 },
      data: { label: name, sub: `${meta.type} · ×${meta.mentions}` },
      type: "pill",
    });
    edges.push({
      id: `e-user-ent-${id}`,
      source: "user",
      target: id,
      label: "mentioned",
      animated: true,
      style: { stroke: "#0ea5e9", strokeWidth: 1.5, opacity: 0.6 },
    });
  });

  const entIds = new Map(ents.map(([n]) => [n, `ent-${slug(n)}`]));
  const pairSeen = new Set<string>();
  const gts = session.ground_truths || [];
  for (const gt of gts) {
    if (gt.turn_id > maxTurnId) break;
    const names = (gt.entities_mentioned || []).map((e) => e.name.trim()).filter((n) => n.length > 1);
    for (let i = 0; i < names.length; i++) {
      for (let j = i + 1; j < names.length; j++) {
        const a = entIds.get(names[i]);
        const b = entIds.get(names[j]);
        if (!a || !b || a === b) continue;
        const key = [a, b].sort().join("--");
        if (pairSeen.has(key)) continue;
        pairSeen.add(key);
        edges.push({
          id: `co-${key}`,
          source: a,
          target: b,
          animated: false,
          style: { stroke: "#cbd5e1", strokeWidth: 1, strokeDasharray: "4 4" },
        });
      }
    }
  }

  return { nodes, edges };
}

export function buildLifeEventGraph(events: LifeEvent[] | undefined): { nodes: Node[]; edges: Edge[] } {
  const cx = 400;
  const cy = 300;
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  const slice = (events ?? []).slice(0, 40);
  slice.forEach((ev, i) => {
    const angle = (i / Math.max(slice.length, 1)) * Math.PI * 2;
    const r = 220 + (i % 3) * 28;
    nodes.push({
      id: ev.id,
      position: { x: cx + Math.cos(angle) * r - 55, y: cy + Math.sin(angle) * r - 18 },
      data: { label: ev.id, sub: ev.date },
      type: "pill",
    });
    for (const c of ev.caused_by || []) {
      edges.push({
        id: `${c}->${ev.id}`,
        source: c,
        target: ev.id,
        label: "caused",
        animated: true,
        style: { stroke: "#f59e0b", strokeWidth: 2 },
      });
    }
  });
  return { nodes, edges };
}
