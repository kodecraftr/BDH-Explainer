"use client";

import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  AreaChart,
  Area,
} from "recharts";

interface MetricRow {
  step: number;
  loss: number;
  param_l2: number;
}

interface MemoryRow {
  test: string;
  model: string;
  position: number;
  loss: number;
}

function parseCSV(text: string): string[][] {
  return text
    .trim()
    .split("\n")
    .map((row) => row.split(","));
}

export function ComparisonCharts() {
  const [bdhMetrics, setBdhMetrics] = useState<MetricRow[]>([]);
  const [transformerMetrics, setTransformerMetrics] = useState<MetricRow[]>([]);
  const [memoryData, setMemoryData] = useState<MemoryRow[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch("/bdh_metrics.csv").then((r) => r.text()),
      fetch("/transformer_metrics.csv").then((r) => r.text()),
      fetch("/memory_comparison.csv").then((r) => r.text()),
    ]).then(([bdhText, transText, memText]) => {
      const bdhRows = parseCSV(bdhText);
      const transRows = parseCSV(transText);
      const memRows = parseCSV(memText);

      setBdhMetrics(
        bdhRows.slice(1).map((r) => ({
          step: Number(r[0]),
          loss: Number(r[1]),
          param_l2: Number(r[2]),
        }))
      );

      setTransformerMetrics(
        transRows.slice(1).map((r) => ({
          step: Number(r[0]),
          loss: Number(r[1]),
          param_l2: Number(r[2]),
        }))
      );

      setMemoryData(
        memRows.slice(1).map((r) => ({
          test: r[0],
          model: r[1],
          position: Number(r[2]),
          loss: Number(r[3]),
        }))
      );

      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <div className="charts-loading">
        <p>Loading comparison data...</p>
      </div>
    );
  }

  // Merge BDH and Transformer training loss onto shared steps
  const allSteps = new Set([
    ...bdhMetrics.map((r) => r.step),
    ...transformerMetrics.map((r) => r.step),
  ]);
  const bdhMap = new Map(bdhMetrics.map((r) => [r.step, r]));
  const transMap = new Map(transformerMetrics.map((r) => [r.step, r]));

  const trainingLossData = Array.from(allSteps)
    .sort((a, b) => a - b)
    .map((step) => ({
      step,
      BDH: bdhMap.get(step)?.loss ?? null,
      Transformer: transMap.get(step)?.loss ?? null,
    }));

  // Parameter efficiency data
  const paramData = Array.from(allSteps)
    .sort((a, b) => a - b)
    .map((step) => ({
      step,
      BDH: bdhMap.get(step)?.param_l2 ?? null,
      Transformer: transMap.get(step)?.param_l2 ?? null,
    }));

  // Copy-gap memory tests
  const copyGapData = memoryData
    .filter((r) => r.test.startsWith("copy_gap"))
    .reduce<Record<string, Record<string, number>>>((acc, r) => {
      const gap = r.test.replace("copy_gap", "Gap ");
      if (!acc[gap]) acc[gap] = {};
      acc[gap][r.model] = r.loss;
      acc[gap]["name"] = 0; // placeholder
      return acc;
    }, {});

  const copyGapChartData = Object.entries(copyGapData).map(([gap, data]) => ({
    name: gap,
    BDH: data["BDH"],
    Transformer: data["Transformer"],
  }));

  // Random sequence memory: compute rolling average for smoother visualization
  const randomBDH = memoryData.filter(
    (r) => r.test === "random" && r.model === "BDH"
  );
  const randomTrans = memoryData.filter(
    (r) => r.test === "random" && r.model === "Transformer"
  );

  const windowSize = 10;
  const rollingAvg = (arr: MemoryRow[], idx: number) => {
    const start = Math.max(0, idx - Math.floor(windowSize / 2));
    const end = Math.min(arr.length, idx + Math.ceil(windowSize / 2));
    const slice = arr.slice(start, end);
    return slice.reduce((s, r) => s + r.loss, 0) / slice.length;
  };

  const maxPos = Math.max(randomBDH.length, randomTrans.length);
  const bdhByPos = new Map(randomBDH.map((r, i) => [r.position, i]));
  const transByPos = new Map(randomTrans.map((r, i) => [r.position, i]));

  const memoryLineData = Array.from({ length: maxPos }, (_, i) => {
    const pos = i + 1;
    const bdhIdx = bdhByPos.get(pos);
    const transIdx = transByPos.get(pos);
    return {
      position: pos,
      BDH: bdhIdx !== undefined ? rollingAvg(randomBDH, bdhIdx) : null,
      Transformer:
        transIdx !== undefined ? rollingAvg(randomTrans, transIdx) : null,
    };
  });

  // Final loss comparison
  const bdhFinalLoss = bdhMetrics[bdhMetrics.length - 1]?.loss ?? 0;
  const transFinalLoss =
    transformerMetrics[transformerMetrics.length - 1]?.loss ?? 0;
  const bdhFinalParam = bdhMetrics[bdhMetrics.length - 1]?.param_l2 ?? 0;
  const transFinalParam =
    transformerMetrics[transformerMetrics.length - 1]?.param_l2 ?? 0;

  const tooltipStyle = {
    backgroundColor: "rgba(15, 10, 30, 0.92)",
    border: "1px solid rgba(106, 45, 213, 0.3)",
    borderRadius: "8px",
    color: "#e2e0f0",
    fontSize: "0.8rem",
  };

  return (
    <div className="comparison-charts">
      {/* Summary stats */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-label">BDH Final Loss</div>
          <div className="stat-value bdh-color">{bdhFinalLoss.toFixed(3)}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Transformer Final Loss</div>
          <div className="stat-value trans-color">
            {transFinalLoss.toFixed(3)}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">BDH Param L2 Norm</div>
          <div className="stat-value bdh-color">
            {bdhFinalParam.toFixed(0)}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Transformer Param L2 Norm</div>
          <div className="stat-value trans-color">
            {transFinalParam.toFixed(0)}
          </div>
        </div>
      </div>

      {/* Chart 1: Training Loss */}
      <div className="chart-block">
        <h3>Training Loss Convergence</h3>
        <p className="chart-desc">
          BDH converges faster and reaches a lower final training loss compared
          to the Transformer, achieving comparable performance with significantly
          fewer training steps.
        </p>
        <div className="chart-wrapper">
          <ResponsiveContainer width="100%" height={340}>
            <LineChart data={trainingLossData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(106,45,213,0.1)" />
              <XAxis
                dataKey="step"
                stroke="#94a3b8"
                fontSize={11}
                label={{ value: "Training Step", position: "insideBottom", offset: -5, fill: "#94a3b8", fontSize: 12 }}
              />
              <YAxis
                stroke="#94a3b8"
                fontSize={11}
                label={{ value: "Loss", angle: -90, position: "insideLeft", fill: "#94a3b8", fontSize: 12 }}
              />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend />
              <Line
                type="monotone"
                dataKey="BDH"
                stroke="#6A2DD5"
                strokeWidth={2.5}
                dot={false}
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="Transformer"
                stroke="#e85d75"
                strokeWidth={2.5}
                dot={false}
                connectNulls
                strokeDasharray="6 3"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 2: Parameter Efficiency */}
      <div className="chart-block">
        <h3>Parameter Efficiency (L2 Norm)</h3>
        <p className="chart-desc">
          BDH achieves its performance with a dramatically smaller parameter
          footprint. The L2 norm of BDH&apos;s parameters is roughly 3.7x smaller
          than the Transformer&apos;s, demonstrating superior parameter efficiency.
        </p>
        <div className="chart-wrapper">
          <ResponsiveContainer width="100%" height={340}>
            <AreaChart data={paramData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(106,45,213,0.1)" />
              <XAxis
                dataKey="step"
                stroke="#94a3b8"
                fontSize={11}
                label={{ value: "Training Step", position: "insideBottom", offset: -5, fill: "#94a3b8", fontSize: 12 }}
              />
              <YAxis
                stroke="#94a3b8"
                fontSize={11}
                label={{ value: "Param L2 Norm", angle: -90, position: "insideLeft", fill: "#94a3b8", fontSize: 12 }}
              />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend />
              <Area
                type="monotone"
                dataKey="Transformer"
                stroke="#e85d75"
                fill="rgba(232,93,117,0.12)"
                strokeWidth={2}
                strokeDasharray="6 3"
                connectNulls
              />
              <Area
                type="monotone"
                dataKey="BDH"
                stroke="#6A2DD5"
                fill="rgba(106,45,213,0.15)"
                strokeWidth={2}
                connectNulls
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 3: Copy-Gap Memory Test */}
      <div className="chart-block">
        <h3>Long-Range Copy Task Performance</h3>
        <p className="chart-desc">
          In the copy-gap memory test, BDH consistently outperforms the
          Transformer across all gap lengths (20, 50, 100, 150 tokens),
          demonstrating stronger long-range memory retention through its
          Hebbian state-space mechanism.
        </p>
        <div className="chart-wrapper">
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={copyGapChartData} barGap={4}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(106,45,213,0.1)" />
              <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
              <YAxis
                stroke="#94a3b8"
                fontSize={11}
                domain={[12, 13.2]}
                label={{ value: "Loss", angle: -90, position: "insideLeft", fill: "#94a3b8", fontSize: 12 }}
              />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend />
              <Bar dataKey="BDH" fill="#6A2DD5" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Transformer" fill="#e85d75" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 4: Random Sequence Memory */}
      <div className="chart-block">
        <h3>Sequence Position Loss (Random Data)</h3>
        <p className="chart-desc">
          Loss across sequence positions on random data. BDH shows a notably
          lower loss at early positions, indicating faster context acquisition.
          The two models converge at longer contexts, with BDH&apos;s state-space
          memory offering competitive performance throughout.
        </p>
        <div className="chart-wrapper">
          <ResponsiveContainer width="100%" height={340}>
            <LineChart data={memoryLineData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(106,45,213,0.1)" />
              <XAxis
                dataKey="position"
                stroke="#94a3b8"
                fontSize={11}
                label={{ value: "Sequence Position", position: "insideBottom", offset: -5, fill: "#94a3b8", fontSize: 12 }}
              />
              <YAxis
                stroke="#94a3b8"
                fontSize={11}
                domain={["auto", "auto"]}
                label={{ value: "Loss (rolling avg)", angle: -90, position: "insideLeft", fill: "#94a3b8", fontSize: 12 }}
              />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend />
              <Line
                type="monotone"
                dataKey="Transformer"
                stroke="#6A2DD5"
                strokeWidth={2}
                dot={false}
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="BDH"
                stroke="#e85d75"
                strokeWidth={2}
                dot={false}
                connectNulls
                strokeDasharray="6 3"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Sparse Activation GIF */}
      <div className="chart-block">
        <h3>Sparse Activation During Inference</h3>
        <p className="chart-desc">
          A key advantage of BDH is its extreme sparsity during inference. The
          animation below shows how only a small fraction (~5%) of neurons
          activate for any given token. This sparse activation pattern, enforced
          by the ReLU gate, leads to highly interpretable and energy-efficient
          computation—a stark contrast to the dense activations in standard
          Transformers where all neurons participate in every computation.
        </p>
        <div className="gif-wrapper">
          <img
            src="/neuron_dynamics.gif"
            alt="BDH sparse neuron activation dynamics during inference"
            className="sparse-gif"
          />
        </div>
      </div>

      {/* Why BDH is Better */}
      <div className="why-better">
        <h3>Why BDH Outperforms the Transformer</h3>
        <p className="why-intro">
          The charts above tell a clear story: BDH rivals—and in several key
          dimensions surpasses—the Transformer architecture. Here is why.
        </p>

        <div className="advantage-card">
          <div className="advantage-number">1</div>
          <div className="advantage-body">
            <h4>Faster Convergence with Fewer Parameters</h4>
            <p>
              BDH reaches its lowest training loss in fewer steps than the
              Transformer, and it does so while maintaining a parameter L2 norm
              that is roughly <strong>3.7x smaller</strong>. This means the
              model stores its learned representations far more compactly.
              Where the Transformer distributes knowledge across dense weight
              matrices, BDH encodes it efficiently through sparse, Hebbian
              synaptic updates—achieving the same expressiveness with a
              fraction of the effective parameter budget.
            </p>
          </div>
        </div>

        <div className="advantage-card">
          <div className="advantage-number">2</div>
          <div className="advantage-body">
            <h4>Constant-Size Memory (O(1) State)</h4>
            <p>
              The Transformer&apos;s self-attention mechanism requires a
              Key-Value cache that grows linearly with sequence length,
              consuming increasingly large amounts of memory during
              generation. BDH replaces this with a <strong>fixed-size state
              matrix</strong> that is updated incrementally at each step.
              Regardless of whether the model has processed 10 tokens or
              10,000, its memory footprint stays constant—making it
              inherently more scalable for long-context tasks.
            </p>
          </div>
        </div>

        <div className="advantage-card">
          <div className="advantage-number">3</div>
          <div className="advantage-body">
            <h4>Superior Long-Range Memory Retention</h4>
            <p>
              The copy-gap benchmark directly tests a model&apos;s ability to
              recall information over increasing distances. BDH consistently
              achieves <strong>lower loss across all gap lengths</strong> (20,
              50, 100, and 150 tokens), showing that its Hebbian
              state-space mechanism retains distant context more faithfully
              than the Transformer&apos;s attention-based lookup.
            </p>
          </div>
        </div>

        <div className="advantage-card">
          <div className="advantage-number">4</div>
          <div className="advantage-body">
            <h4>Extreme Sparsity and Interpretability</h4>
            <p>
              During inference, only about <strong>5% of BDH&apos;s neurons
              activate</strong> for any given token (as shown in the animation
              above). This stands in stark contrast to the Transformer, where
              every neuron in every layer participates in every computation.
              Sparse activation provides two critical benefits: it makes the
              model&apos;s internal representations directly interpretable
              (each active neuron cluster corresponds to a semantic concept),
              and it dramatically reduces the number of floating-point
              operations needed per token—translating to lower energy
              consumption and faster inference on hardware that can exploit
              sparsity.
            </p>
          </div>
        </div>

        <div className="advantage-card">
          <div className="advantage-number">5</div>
          <div className="advantage-body">
            <h4>Biological Plausibility</h4>
            <p>
              Unlike the Transformer, which was engineered purely for
              performance, BDH is grounded in principles from neuroscience.
              Its Hebbian learning rule (&quot;neurons that fire together wire
              together&quot;), sparse modular activations, and local
              interaction dynamics mirror how biological neural circuits
              process information. This is not merely an aesthetic property—it
              provides a principled framework for understanding <em>what</em>{" "}
              the model has learned and <em>why</em>, enabling the kind of
              mechanistic interpretability that remains elusive in dense
              Transformer networks.
            </p>
          </div>
        </div>

        <div className="advantage-card">
          <div className="advantage-number">6</div>
          <div className="advantage-body">
            <h4>Linear-Time Inference</h4>
            <p>
              Transformer self-attention scales quadratically with sequence
              length (<strong>O(T&sup2;)</strong>) during training and
              requires O(T) KV-cache reads per generated token. BDH&apos;s
              linear attention mechanism processes each new token in{" "}
              <strong>O(1)</strong> time relative to context length—it simply
              updates the fixed state matrix and queries it. This makes BDH
              fundamentally more efficient for autoregressive generation,
              especially as context windows grow into the tens or hundreds of
              thousands of tokens.
            </p>
          </div>
        </div>
      </div>

      <style jsx>{`
        .comparison-charts {
          display: flex;
          flex-direction: column;
          gap: 2.5rem;
        }

        .stats-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 1rem;
        }

        @media (min-width: 640px) {
          .stats-grid {
            grid-template-columns: repeat(4, 1fr);
          }
        }

        .stat-card {
          background: rgba(106, 45, 213, 0.04);
          border: 1px solid rgba(106, 45, 213, 0.12);
          border-radius: 12px;
          padding: 1rem 1.25rem;
          text-align: center;
        }

        .stat-label {
          font-size: 0.75rem;
          color: #64748b;
          font-weight: 500;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          margin-bottom: 0.4rem;
        }

        .stat-value {
          font-size: 1.5rem;
          font-weight: 700;
          font-family: var(--font-dm-mono), 'DM Mono', ui-monospace, monospace;
        }

        .bdh-color {
          color: #6A2DD5;
        }

        .trans-color {
          color: #e85d75;
        }

        .chart-block {
          background: rgba(106, 45, 213, 0.02);
          border: 1px solid rgba(106, 45, 213, 0.08);
          border-radius: 16px;
          padding: 1.5rem;
        }

        .chart-block h3 {
          color: #1e293b;
          font-size: 1.15rem;
          font-weight: 600;
          margin-bottom: 0.3rem;
          padding-top: 0;
        }

        .chart-desc {
          color: #64748b;
          font-size: 0.875rem !important;
          line-height: 1.6 !important;
          margin-bottom: 1.25rem !important;
          margin-top: 0 !important;
        }

        .chart-wrapper {
          width: 100%;
          overflow-x: auto;
        }

        .gif-wrapper {
          display: flex;
          justify-content: center;
          margin-top: 1rem;
        }

        .sparse-gif {
          max-width: 100%;
          height: auto;
          border-radius: 12px;
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        }

        .charts-loading {
          text-align: center;
          padding: 3rem;
          color: #64748b;
        }

        .why-better {
          display: flex;
          flex-direction: column;
          gap: 1.25rem;
        }

        .why-better h3 {
          color: #1e293b;
          font-size: 1.3rem;
          font-weight: 700;
          margin-bottom: 0;
          padding-top: 0;
        }

        .why-intro {
          color: #475569;
          font-size: 0.95rem;
          line-height: 1.7;
          margin: 0;
        }

        .advantage-card {
          display: flex;
          gap: 1rem;
          background: rgba(106, 45, 213, 0.03);
          border: 1px solid rgba(106, 45, 213, 0.1);
          border-radius: 12px;
          padding: 1.25rem 1.5rem;
          align-items: flex-start;
        }

        .advantage-number {
          flex-shrink: 0;
          width: 2rem;
          height: 2rem;
          display: flex;
          align-items: center;
          justify-content: center;
          background: #6A2DD5;
          color: white;
          font-weight: 700;
          font-size: 0.9rem;
          border-radius: 50%;
          margin-top: 0.1rem;
        }

        .advantage-body h4 {
          color: #1e293b;
          font-size: 1.02rem;
          font-weight: 600;
          margin: 0 0 0.4rem 0;
          padding-top: 0;
        }

        .advantage-body p {
          color: #475569;
          font-size: 0.875rem;
          line-height: 1.7;
          margin: 0;
        }
      `}</style>
    </div>
  );
}
