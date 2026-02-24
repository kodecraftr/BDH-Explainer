"use client";

import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertCircle, X } from "lucide-react";

// --- Helper Functions ---

// Formatting helper
const formatNumber = (num: number, decimals = 2) => {
  if (num === -Infinity) return "-∞";
  return num.toFixed(decimals);
};

// --- Types ---

interface TokenData {
  token_id: number;
  word: string; // mapped from 'token'
  logit: number;
  scaledLogit: number;
  isTopK: boolean;
  topKLogit: number;
  softmax: number;
}

interface LogitItem {
  token_id: number;
  token: string;
  logit: number;
}

// --- Components ---

// Version 1: Top-K list styled to match the EncodingVisualizer card
const ProbabilityVisualizerV1 = ({
  data,
  topK,
  onClick,
}: {
  data: TokenData[];
  topK: number;
  onClick: () => void;
}) => {
  const topTokens = [...data].sort((a, b) => b.softmax - a.softmax).slice(0, topK);

  return (
    <div
      className="relative group w-full cursor-zoom-in animate-in fade-in slide-in-from-bottom-4 duration-500"
      onClick={onClick}
    >
      {/* Animated dashed border on hover — matches EncodingVisualizer */}
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
          <div className="flex items-center justify-between mb-0.5">
              <div className="flex items-center gap-2">
              <span className="text-[12px] font-mono text-neutral-400 dark:text-neutral-600 tracking-widest tabular-nums">03</span>
              <p className="text-lg font-bold uppercase tracking-[0.12em] text-neutral-600 dark:text-neutral-300">
                Next Token
              </p>
            </div>
            <span className="text-[10px] text-neutral-400 bg-neutral-100 dark:bg-neutral-800 px-2 py-0.5 rounded-full font-normal">
              Click for details
            </span>
          </div>
          <p className="text-[10px] text-neutral-400 dark:text-neutral-600 pl-6">
            Top-K sampling &amp; softmax distribution
          </p>
        </div>

        {/* Table-style layout for perfect column alignment */}
        <table className="w-full border-collapse" style={{ tableLayout: "fixed" }}>
          <colgroup>
            <col style={{ width: "22%" }} />
            <col style={{ width: "14%" }} />
            <col style={{ width: "16%" }} />
            <col />
            <col style={{ width: "15%" }} />
          </colgroup>
          <thead>
            <tr>
              <th className="text-left text-[9px] font-medium uppercase tracking-widest text-neutral-400 dark:text-neutral-600 pb-2 pl-1">
                Token
              </th>
              <th className="text-right text-[9px] font-medium uppercase tracking-widest text-neutral-400 dark:text-neutral-600 pb-2 pr-2">
                Logit
              </th>
              <th className="text-right text-[9px] font-medium uppercase tracking-widest text-neutral-400 dark:text-neutral-600 pb-2 pr-2">
                ÷ Temp
              </th>
              <th className="text-left text-[9px] font-medium uppercase tracking-widest text-neutral-400 dark:text-neutral-600 pb-2 pl-2">
                Softmax <span className="normal-case tracking-normal opacity-60">(top-{topK})</span>
              </th>
              <th className="text-right text-[9px] font-medium uppercase tracking-widest text-neutral-400 dark:text-neutral-600 pb-2 pr-1">
                %
              </th>
            </tr>
          </thead>
          <tbody>
            {topTokens.map((item, idx) => (
              <tr
                key={`${item.word}-${idx}`}
                className="group/row hover:bg-neutral-100/70 dark:hover:bg-neutral-800/50 transition-colors rounded-lg"
              >
                {/* Token pill */}
                <td className="py-2 pl-1">
                  <span className="inline-block text-xs font-semibold font-mono bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 rounded-md px-1.5 py-0.5 border border-neutral-200 dark:border-neutral-700 truncate max-w-full">
                    {item.word}
                  </span>
                </td>

                {/* Raw logit */}
                <td className="text-right text-xs font-mono text-neutral-400 dark:text-neutral-500 tabular-nums py-2 pr-2">
                  {formatNumber(item.logit, 1)}
                </td>

                {/* Scaled logit (÷ temperature) */}
                <td className="text-right text-xs font-mono text-neutral-500 dark:text-neutral-400 tabular-nums py-2 pr-2">
                  {formatNumber(item.scaledLogit, 1)}
                </td>

                {/* Softmax probability bar */}
                <td className="py-2 px-2">
                  <div className="w-full rounded overflow-hidden bg-neutral-100 dark:bg-neutral-800/50" style={{ height: 14 }}>
                    <div
                      className="h-full rounded bg-indigo-400 dark:bg-indigo-500 transition-all duration-500"
                      style={{ width: `${Math.max(2, item.softmax * 100)}%`, opacity: 0.8 + item.softmax * 0.2 }}
                    />
                  </div>
                </td>

                {/* Softmax % */}
                <td className="text-right text-xs font-mono text-neutral-400 dark:text-neutral-500 tabular-nums py-2 pr-1">
                  {(item.softmax * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// Version 2: Detailed Table (Modal Content)
const ProbabilityVisualizerV2 = ({
  data,
  temperature,
  topK,
}: {
  data: TokenData[];
  temperature: number;
  topK: number;
}) => {
  // Sort by Softmax (descending) to show the "winners" at the top.
  const sortedData = useMemo(() => {
    return [...data].sort((a, b) => b.softmax - a.softmax);
  }, [data]);

  return (
    <Card className="border-none shadow-none bg-transparent">
      <div className="overflow-x-auto">
        <div className="min-w-[800px] text-sm">
          {/* Header */}
          <div className="grid grid-cols-12 gap-4 border-b border-neutral-200 dark:border-neutral-800 pb-3 mb-3 font-medium text-neutral-500 text-xs uppercase tracking-wider">
            <div className="col-span-2">Token</div>
            <div className="col-span-2 text-right">Logits</div>
            <div className="col-span-2 text-right">Scaled (/{temperature})</div>
            <div className="col-span-2 text-right text-indigo-600 dark:text-indigo-400">
              Top-K ({topK})
            </div>
            <div className="col-span-4 pl-4">Softmax</div>
          </div>

          {/* Rows - Using content-visibility for performance with large lists */}
          <div
            className="space-y-2 max-h-[60vh] overflow-y-auto pr-2"
            style={{ contentVisibility: "auto" }}
          >
            {sortedData.map((item, index) => (
              <div
                key={`${item.word}-${index}`} // Use index fallback if tokens are not unique
                className={`grid grid-cols-12 gap-4 items-center p-2 rounded-md transition-colors ${item.isTopK ? "bg-indigo-50/50 dark:bg-indigo-900/10" : "hover:bg-neutral-50 dark:hover:bg-neutral-900/50 opacity-60"}`}
              >
                {/* Token */}
                <div
                  className="col-span-2 font-mono font-medium text-neutral-700 dark:text-neutral-300 truncate"
                  title={item.word}
                >
                  {item.word}
                </div>

                {/* Logits */}
                <div className="col-span-2 text-right font-mono text-neutral-500">
                  {formatNumber(item.logit)}
                </div>

                {/* Scaled Logits */}
                <div className="col-span-2 text-right font-mono text-neutral-500">
                  {formatNumber(item.scaledLogit)}
                </div>

                {/* Top-K Logits */}
                <div
                  className={`col-span-2 text-right font-mono font-bold ${item.topKLogit === -Infinity ? "text-neutral-300 dark:text-neutral-700" : "text-indigo-600 dark:text-indigo-400"}`}
                >
                  {formatNumber(item.topKLogit)}
                </div>

                {/* Softmax Bar */}
                <div className="col-span-4 pl-4 flex items-center gap-3">
                  <div className="flex-1 h-2 bg-neutral-100 dark:bg-neutral-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-indigo-500 transition-all"
                      style={{ width: `${Math.max(0, item.softmax * 100)}%` }}
                    />
                  </div>
                  <span className="w-12 text-right font-mono text-xs text-neutral-600 dark:text-neutral-400">
                    {(item.softmax * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            ))}
            {sortedData.length === 0 && (
              <div className="text-center py-8 text-neutral-500">
                No logits data available.
              </div>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
};

// --- Main Component ---

export function ProbabilityVisualizer({
  inputText,
  temperature = 1,
  topK = 5,
  logits = [],
  error = null,
}: {
  inputText: string;
  temperature?: number;
  topK?: number;
  logits?: LogitItem[];
  error?: string | null;
}) {
  const [isDetailOpen, setIsDetailOpen] = useState(false);

  const data = useMemo(() => {
    if (!logits || logits.length === 0) return [];

    // 1. Process Raw Data
    const processed = logits.map((item) => ({
      token_id: item.token_id,
      word: item.token,
      logit: item.logit,
      // 2. Scaled Logits
      scaledLogit: item.logit / temperature,
    }));

    // 3. Identification of Top-K
    // We need to identify which items are effectively in the "Top K" based on their scaled logit
    const sortedByScaled = [...processed].sort(
      (a, b) => b.scaledLogit - a.scaledLogit,
    );

    // Create a set of the top K items (we use their original index or object reference, but here we rebuild)
    const topKItems = sortedByScaled.slice(0, topK);
    const topKSet = new Set(topKItems); // Set of references

    // 4. Softmax Calculation (Top-K Restricted)
    // We only perform softmax on the Top-K set to visualize the sampling distribution.
    // Non-Top-K items are effectively masked (prob = 0).

    // Find max for numerical stability within the Top-K set
    const maxScaledTopK = Math.max(...topKItems.map((p) => p.scaledLogit));

    // Calculate sum of exps ONLY for Top-K items
    const sumExpsTopK = topKItems.reduce((acc, p) => {
      const exp = Math.exp(p.scaledLogit - maxScaledTopK);
      return acc + exp; // Math.exp(-Infinity) === 0
    }, 0);

    // Final mapping
    return processed.map((p) => {
      const isTopK = topKSet.has(p); // Check reference inclusion

      let softmax = 0;
      if (isTopK && sumExpsTopK > 0) {
        const exp = Math.exp(p.scaledLogit - maxScaledTopK);
        softmax = exp / sumExpsTopK;
      }

      return {
        ...p,
        isTopK,
        topKLogit: isTopK ? p.scaledLogit : -Infinity,
        softmax,
      };
    });
  }, [logits, temperature, topK]);

  if (error) {
    return (
      <Card className="border-red-200 dark:border-red-900/50 bg-red-50/50 dark:bg-red-900/20">
        <CardContent className="flex items-center gap-3 p-4 text-red-600 dark:text-red-400">
          <AlertCircle className="w-5 h-5" />
          <p className="text-sm font-medium">{error}</p>
        </CardContent>
      </Card>
    );
  }

  if (!data.length && !inputText) return null;
  if (!data.length && inputText) {
    // Loading state or empty state if generated but no logits yet
    return (
      <Card className="border-neutral-200 dark:border-neutral-800 shadow-sm opacity-50">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg text-neutral-400">
            Probability
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-neutral-400 italic">
            Waiting for model data...
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <>
      <ProbabilityVisualizerV1
        data={data}
        topK={topK}
        onClick={() => setIsDetailOpen(true)}
      />

      {/* Modal Overlay */}
      {isDetailOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6"
          style={{ animation: "fade-in 0.2s ease-out forwards" }}
        >
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm transition-opacity"
            onClick={() => setIsDetailOpen(false)}
          />

          {/* Modal Content */}
          <div
            className="relative z-10 w-full max-w-5xl max-h-[90vh] overflow-hidden shadow-2xl animate-in zoom-in-95 duration-200 bg-white dark:bg-neutral-950 rounded-xl border border-neutral-200 dark:border-neutral-800 flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="p-4 border-b border-neutral-100 dark:border-neutral-800 flex justify-between items-center bg-neutral-50/50 dark:bg-neutral-900/50">
              <div>
                <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 text-lg">
                  Next Token Probability Distribution
                </h3>
                <p className="text-sm text-neutral-500">
                  Visualizing the Softmax selection process
                </p>
              </div>
              <button
                onClick={() => setIsDetailOpen(false)}
                className="text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300 p-2 rounded-md hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
              >
                <span className="sr-only">Close</span>
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Scrollable Body */}
            <div className="p-6 overflow-hidden flex-1 flex flex-col">
              <ProbabilityVisualizerV2
                data={data}
                temperature={temperature}
                topK={topK}
              />
            </div>

            {/* Footer */}
            <div className="p-3 bg-neutral-50 dark:bg-neutral-900 border-t border-neutral-100 dark:border-neutral-800 text-center text-xs text-neutral-500 font-mono">
              Logits → Scaled Logits (/Temp) → Top-K Filter → Softmax →
              Probabilities
            </div>
          </div>
        </div>
      )}
    </>
  );
}
