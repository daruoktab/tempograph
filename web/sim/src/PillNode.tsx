import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";

export type PillNodeData = { label: string; sub: string };

export const PillNode = memo(function PillNode({ data }: NodeProps) {
  const d = data as PillNodeData;
  return (
    <div
      style={{
        minWidth: 140,
        maxWidth: 220,
        padding: "12px 16px",
        borderRadius: "999px", // Fully rounded pill shape
        background: "#ffffff",
        border: "1px solid var(--border-base)",
        boxShadow: "var(--shadow-md)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        transition: "transform 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.2s ease",
        cursor: "pointer",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = "translateY(-3px) scale(1.02)";
        e.currentTarget.style.boxShadow = "var(--shadow-lg)";
        e.currentTarget.style.borderColor = "var(--border-dark)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = "translateY(0) scale(1)";
        e.currentTarget.style.boxShadow = "var(--shadow-md)";
        e.currentTarget.style.borderColor = "var(--border-base)";
      }}
    >
      <Handle 
        type="target" 
        position={Position.Top} 
        style={{ background: "var(--primary)", width: 8, height: 8, border: "2px solid #fff", top: -4 }} 
      />
      <div style={{ fontWeight: 600, fontSize: 13, color: "var(--text-main)", lineHeight: 1.2 }}>{d.label}</div>
      {d.sub && (
        <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2, fontWeight: 500 }}>{d.sub}</div>
      )}
      <Handle 
        type="source" 
        position={Position.Bottom} 
        style={{ background: "var(--primary)", width: 8, height: 8, border: "2px solid #fff", bottom: -4 }} 
      />
    </div>
  );
});
