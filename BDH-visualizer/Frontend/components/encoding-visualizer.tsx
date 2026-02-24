"use client";

import { useMemo } from "react";

// --- Helper Functions ---

const seededRandom = (seed: string) => {
  let h = 0xdeadbeef;
  for (let i = 0; i < seed.length; i++) {
    h = Math.imul(h ^ seed.charCodeAt(i), 2654435761);
  }
  return ((h ^ (h >>> 13)) >>> 0) / 4294967296;
};

// Heatmap colour: slate → indigo → violet
const heatColor = (v: number): string => {
  const r = Math.round(148 + (109 - 148) * v);
  const g = Math.round(163 + (40 - 163) * v);
  const b = Math.round(184 + (217 - 184) * v);
  return `rgb(${r},${g},${b})`;
};

const EmbeddingBar = ({ values }: { values: number[] }) => (
  <div className="flex rounded overflow-hidden" style={{ width: "100%", maxWidth: 80, height: 16, gap: 1 }}>
    {values.map((v, i) => (
      <div key={i} className="flex-1" style={{ backgroundColor: heatColor(v), opacity: 0.8 + v * 0.2 }} />
    ))}
  </div>
);

// --- Main Visualizer ---

const EncodingVisualizerV2 = ({
  inputText,
  tokenIds,
  tokensData,
}: {
  inputText: string;
  tokenIds?: number[];
  tokensData?: any[];
}) => {
  const data = useMemo(() => {
    if (tokensData && tokensData.length > 0) {
      return tokensData.map((t) => {
        const raw: number[] = t.original_embedding;
        const min = Math.min(...raw);
        const max = Math.max(...raw);
        const range = max - min || 1;
        const embedding = raw.map((v) => (v - min) / range);
        return { token: t.token_text.trim(), id: t.token_id, embedding };
      });
    }
    if (!inputText.trim()) return [];
    return inputText
      .trim()
      .split(/\s+/)
      .map((token, index) => ({
        token,
        id: tokenIds?.[index] ?? Math.floor(seededRandom(token) * 50000),
        embedding: Array.from({ length: 32 }, (_, i) => seededRandom(`${token}-${i}`)),
      }));
  }, [inputText, tokenIds, tokensData]);

  if (!data.length) return null;

  return (
    <div className="relative group w-full p-0">

      {/* Animated dashed SVG border on hover */}
      <svg
        className="absolute inset-0 w-full h-full pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-300"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
        style={{ overflow: "visible" }}
      >
        <rect
          x="1" y="1"
          width="calc(100% - 2px)"
          height="calc(100% - 2px)"
          rx="10" ry="10"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.2"
          strokeDasharray="10 8"
          strokeLinecap="round"
          className="text-neutral-300 dark:text-neutral-600"
        />
      </svg>

      <div className="px-4 py-3">
        {/* Section label */}
        <div className="mb-4">
          <div className="flex items-center gap-2 mb-0.5">
            <span className="text-[12px] font-mono text-neutral-400 dark:text-neutral-600 tracking-widest tabular-nums">01</span>
            <p className="text-lg font-bold uppercase tracking-[0.12em] text-neutral-600 dark:text-neutral-300">
              Token Embeddings
            </p>
          </div>
          <p className="text-[10px] text-neutral-400 dark:text-neutral-600 pl-6">
            Tokenization &amp; embedding lookup
          </p>
        </div>

        {/* Table layout for consistent column alignment */}
        <table className="w-full border-collapse" style={{ tableLayout: "fixed" }}>
          <colgroup>
            <col style={{ width: "30%" }} />
            <col style={{ width: "20px" }} />
            <col />
            <col style={{ width: "52px" }} />
          </colgroup>
          <thead>
            <tr>
              <th className="text-left text-[9px] font-medium uppercase tracking-widest text-neutral-400 dark:text-neutral-600 pb-2 pl-1">
                Token
              </th>
              <th />
              <th className="text-left text-[9px] font-medium uppercase tracking-widest text-neutral-400 dark:text-neutral-600 pb-2">
                Vector <span className="normal-case tracking-normal opacity-60">(256d)</span>
              </th>
              <th className="text-right text-[9px] font-medium uppercase tracking-widest text-neutral-400 dark:text-neutral-600 pb-2 pr-1">
                ID
              </th>
            </tr>
          </thead>
          <tbody>
            {data.map((item, i) => (
              <tr
                key={`${item.token}-${i}`}
                className="group/row hover:bg-neutral-100/70 dark:hover:bg-neutral-800/50 transition-colors"
              >
                {/* Token pill */}
                <td className="py-2 pl-1">
                  <span className="inline-block text-sm font-semibold font-mono bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 rounded-md px-1 py-0.5 border border-neutral-200 dark:border-neutral-700 truncate max-w-full">
                    {item.token || <span className="opacity-40 italic">spc</span>}
                  </span>
                </td>

                {/* Arrow */}
                <td className="text-center py-2">
                  <svg width="18" height="10" viewBox="0 0 22 10" fill="none" className="inline-block text-neutral-300 dark:text-neutral-600">
                    <line x1="0" y1="5" x2="17" y2="5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
                    <polyline points="12,1.5 17,5 12,8.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </td>

                {/* Heatmap */}
                <td className="py-2">
                  <EmbeddingBar values={item.embedding} />
                </td>

                {/* Token ID */}
                <td className="text-right text-sm font-mono text-neutral-400 dark:text-neutral-500 tabular-nums py-2 pr-1">
                  {item.id}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// --- Main Export ---

export function EncodingVisualizer({
  inputText,
  tokenIds,
  tokensData,
}: {
  inputText: string;
  tokenIds?: number[];
  tokensData?: any[];
}) {
  if (!inputText.trim()) return null;

  return (
    <div className="w-full animate-in fade-in slide-in-from-bottom-4 duration-500">
      <EncodingVisualizerV2
        inputText={inputText}
        tokenIds={tokenIds}
        tokensData={tokensData}
      />
    </div>
  );
}
