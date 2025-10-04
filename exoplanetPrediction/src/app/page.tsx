"use client";

import Image from "next/image";
import { useMemo, useState } from "react";

type DatasetOption = {
  id: string;
  label: string;
  description: string;
};

type ModelOption = {
  id: string;
  label: string;
  description: string;
};

type Candidate = {
  id: string;
  score: number;
  period: string;
  depth: string;
  duration: string;
  snr: string;
};

const datasetOptions: DatasetOption[] = [
  {
    id: "tess-sample",
    label: "TESS Sample",
    description: "Curated light curves from TESS sectors with annotated transits.",
  },
  {
    id: "kepler-sample",
    label: "Kepler Sample",
    description: "Kepler confirmed planets and threshold crossing events for benchmarking.",
  },
  {
    id: "upload",
    label: "Upload dataset",
    description: "Bring your own CSV/JSON formatted as { time, flux, meta }.",
  },
];

const modelOptions: ModelOption[] = [
  {
    id: "cnn",
    label: "CNN",
    description: "Convolutional network tuned for transit-like features in detrended curves.",
  },
  {
    id: "lstm",
    label: "LSTM",
    description: "Sequence model that captures temporal dependencies in the light curve.",
  },
  {
    id: "rf",
    label: "Random Forest",
    description: "Tree ensemble using engineered statistics from the flux series.",
  },
  {
    id: "xgb",
    label: "XGBoost",
    description: "Boosted gradient trees with built-in handling of class imbalance.",
  },
  {
    id: "custom",
    label: "Custom",
    description: "Plug in weights or an endpoint that serves your bespoke model.",
  },
];

const mockCandidates: Candidate[] = [
  { id: "TIC-299250015", score: 0.94, period: "3.21 d", depth: "980 ppm", duration: "2.7 hr", snr: "11.2" },
  { id: "TIC-1723701", score: 0.88, period: "7.84 d", depth: "620 ppm", duration: "3.1 hr", snr: "9.5" },
  { id: "KIC-8120608", score: 0.86, period: "11.02 d", depth: "450 ppm", duration: "4.0 hr", snr: "8.9" },
];

export default function Home() {
  const [dataset, setDataset] = useState<string>(datasetOptions[0]?.id ?? "");
  const [targetMode, setTargetMode] = useState<"single" | "bulk">("single");
  const [targetId, setTargetId] = useState("");
  const [bulkTargets, setBulkTargets] = useState("");
  const [detrend, setDetrend] = useState(true);
  const [normalize, setNormalize] = useState(true);
  const [gapFill, setGapFill] = useState(false);
  const [windowLength, setWindowLength] = useState(2);
  const [model, setModel] = useState<string>(modelOptions[0]?.id ?? "cnn");
  const [threshold, setThreshold] = useState(0.5);
  const [epochs, setEpochs] = useState(25);
  const [learningRate, setLearningRate] = useState(0.001);
  const [batchSize, setBatchSize] = useState(64);
  const [validationSplit, setValidationSplit] = useState(0.2);
  const [seed, setSeed] = useState(42);
  const [minPeriod, setMinPeriod] = useState(0.5);
  const [maxPeriod, setMaxPeriod] = useState(40);
  const [peakCount, setPeakCount] = useState(5);

  const selectedDataset = useMemo(
    () => datasetOptions.find((option) => option.id === dataset),
    [dataset],
  );

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-white/5 bg-slate-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-col gap-6 px-6 py-12 sm:py-16 lg:px-8">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <Image
                src="/winnhacks-logo.png"
                alt="WinnHacks logo"
                width={72}
                height={72}
                className="h-16 w-16 rounded-xl bg-slate-900/70 p-3 ring-1 ring-cyan-400/40"
                priority
              />
              <div>
                <p className="text-xs uppercase tracking-[0.3em] text-cyan-400">WinnHacks NASA Space Apps 2025</p>
                <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">
                  A World Away: Hunting for Exoplanets with AI
                </h1>
              </div>
            </div>
            <div className="rounded-full border border-cyan-500/40 px-4 py-2 text-sm text-cyan-300">Live prototype</div>
          </div>
          <p className="max-w-3xl text-sm text-slate-300 sm:text-base">
            Configure datasets, preprocessing, and AI models to detect exoplanet candidates from TESS and Kepler light curves. Monitor diagnostics, interpretability insights, and download comprehensive reports for your team.
          </p>
          <div className="flex flex-wrap gap-3 text-xs text-slate-400">
            <span className="rounded-full border border-white/10 px-3 py-1">Light curve ingestion</span>
            <span className="rounded-full border border-white/10 px-3 py-1">BLS period search</span>
            <span className="rounded-full border border-white/10 px-3 py-1">Machine learning inference</span>
            <span className="rounded-full border border-white/10 px-3 py-1">Explainability</span>
            <span className="rounded-full border border-white/10 px-3 py-1">Reporting</span>
          </div>
        </div>
      </header>

      <main className="mx-auto grid min-h-[calc(100vh-200px)] max-w-6xl gap-10 px-6 py-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] lg:px-8">
        <section className="space-y-8">
          <ConfigurationCard title="1. Dataset" description="Choose a built-in sample or upload your own light curves.">
            <div className="space-y-4">
              <div className="grid gap-3 sm:grid-cols-3">
                {datasetOptions.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    onClick={() => setDataset(option.id)}
                    className={`rounded-xl border p-4 text-left transition hover:border-cyan-400/60 hover:bg-cyan-400/5 ${
                      dataset === option.id ? "border-cyan-400 bg-cyan-400/10" : "border-white/10"
                    }`}
                  >
                    <p className="text-sm font-semibold text-white">{option.label}</p>
                    <p className="mt-2 text-xs text-slate-300">{option.description}</p>
                  </button>
                ))}
              </div>
              {selectedDataset?.id === "upload" && (
                <div className="rounded-lg border border-dashed border-white/20 bg-white/5 p-4 text-xs text-slate-200">
                  <p className="font-semibold text-white">Upload CSV or JSON</p>
                  <p className="mt-1 text-slate-300">Expected keys: time[], flux[], meta. Drag and drop files or paste JSON.</p>
                </div>
              )}
            </div>
          </ConfigurationCard>

          <ConfigurationCard title="2. Targets" description="Select a single TIC/KIC identifier or analyze a bulk list.">
            <div className="flex gap-3 text-xs">
              <button
                type="button"
                onClick={() => setTargetMode("single")}
                className={`rounded-full px-4 py-1 transition ${
                  targetMode === "single" ? "bg-cyan-500 text-slate-950" : "bg-white/10"
                }`}
              >
                Single target
              </button>
              <button
                type="button"
                onClick={() => setTargetMode("bulk")}
                className={`rounded-full px-4 py-1 transition ${
                  targetMode === "bulk" ? "bg-cyan-500 text-slate-950" : "bg-white/10"
                }`}
              >
                Bulk list
              </button>
            </div>
            {targetMode === "single" ? (
              <div className="space-y-2 text-sm">
                <label className="block text-slate-200">TIC/KIC identifier</label>
                <input
                  value={targetId}
                  onChange={(event) => setTargetId(event.target.value)}
                  placeholder="e.g. TIC 299250015"
                  className="w-full rounded-md border border-white/10 bg-slate-900 px-4 py-2 text-sm focus:border-cyan-400 focus:outline-none"
                />
              </div>
            ) : (
              <div className="space-y-2 text-sm">
                <label className="block text-slate-200">Paste a list of TIC/KIC identifiers</label>
                <textarea
                  value={bulkTargets}
                  onChange={(event) => setBulkTargets(event.target.value)}
                  placeholder="One identifier per line"
                  rows={4}
                  className="w-full rounded-md border border-white/10 bg-slate-900 px-4 py-2 text-sm focus:border-cyan-400 focus:outline-none"
                />
              </div>
            )}
          </ConfigurationCard>

          <ConfigurationCard title="3. Preprocessing" description="Prepare the light curves before model ingestion.">
            <div className="grid gap-4 sm:grid-cols-2">
              <ToggleRow label="Detrend" value={detrend} onChange={setDetrend} helper="Removes long-term trends via spline fitting." />
              <ToggleRow label="Normalize" value={normalize} onChange={setNormalize} helper="Scale flux to unit median to stabilize training." />
              <ToggleRow label="Gap fill" value={gapFill} onChange={setGapFill} helper="Interpolate missing cadences with inpainting." />
              <div className="space-y-2 text-sm">
                <label className="block text-slate-200">Window length (days)</label>
                <input
                  type="number"
                  min={0.5}
                  step={0.5}
                  value={windowLength}
                  onChange={(event) => setWindowLength(Number(event.target.value))}
                  className="w-full rounded-md border border-white/10 bg-slate-900 px-4 py-2 text-sm focus:border-cyan-400 focus:outline-none"
                />
              </div>
            </div>
          </ConfigurationCard>

          <ConfigurationCard title="4. Hyperparameters" description="Tune the inference/training pipeline when needed.">
            <div className="space-y-6">
              <div className="grid gap-4 sm:grid-cols-2">
                {modelOptions.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    onClick={() => setModel(option.id)}
                    className={`rounded-xl border p-4 text-left transition hover:border-cyan-400/60 hover:bg-cyan-400/5 ${
                      model === option.id ? "border-cyan-400 bg-cyan-400/10" : "border-white/10"
                    }`}
                  >
                    <p className="text-sm font-semibold text-white">{option.label}</p>
                    <p className="mt-2 text-xs text-slate-300">{option.description}</p>
                  </button>
                ))}
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <SliderField
                  label="Decision threshold"
                  value={threshold}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={setThreshold}
                  helper={`Score cutoff at ${threshold.toFixed(2)} for positive detections.`}
                />
                <NumberField label="Epochs" value={epochs} onChange={setEpochs} min={1} max={200} helper="Training iterations per run." />
                <NumberField
                  label="Learning rate"
                  value={learningRate}
                  onChange={setLearningRate}
                  step={0.0001}
                  min={0.0001}
                  max={0.01}
                  helper="Optimizer step size (Adam by default)."
                />
                <NumberField label="Batch size" value={batchSize} onChange={setBatchSize} min={16} max={512} step={16} helper="Samples per gradient update." />
                <NumberField
                  label="Validation split"
                  value={validationSplit}
                  onChange={setValidationSplit}
                  step={0.05}
                  min={0.05}
                  max={0.5}
                  helper="Fraction of examples held out for validation."
                />
                <NumberField label="Random seed" value={seed} onChange={setSeed} min={0} max={9999} helper="Controls reproducibility for stochastic steps." />
              </div>

              <div className="grid gap-4 sm:grid-cols-3">
                <NumberField
                  label="BLS min period (days)"
                  value={minPeriod}
                  onChange={setMinPeriod}
                  step={0.1}
                  min={0.1}
                  max={100}
                  helper="Shortest period to search."
                />
                <NumberField
                  label="BLS max period (days)"
                  value={maxPeriod}
                  onChange={setMaxPeriod}
                  step={0.5}
                  min={1}
                  max={200}
                  helper="Longest period considered."
                />
                <NumberField
                  label="Number of peaks"
                  value={peakCount}
                  onChange={setPeakCount}
                  min={1}
                  max={20}
                  helper="Top peaks returned from periodogram."
                />
              </div>
            </div>
          </ConfigurationCard>

          <ReviewCard
            dataset={selectedDataset?.label ?? ""}
            targetSummary={targetMode === "single" && targetId ? targetId : `${bulkTargets.split("\n").filter(Boolean).length} targets`}
            model={modelOptions.find((option) => option.id === model)?.label ?? ""}
            threshold={threshold}
          />
        </section>

        <section className="space-y-8">
          <ResultsPanel />
          <DiagnosticsPanel />
          <CandidateGallery candidates={mockCandidates} />
          <DownloadsPanel />
        </section>
      </main>
    </div>
  );
}

function ConfigurationCard({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-white/10 bg-white/5 p-6 shadow-lg shadow-black/40">
      <div className="flex flex-col gap-2 border-b border-white/10 pb-4">
        <h2 className="text-lg font-semibold text-white">{title}</h2>
        <p className="text-sm text-slate-300">{description}</p>
      </div>
      <div className="pt-4 text-sm text-slate-200">{children}</div>
    </section>
  );
}

function ToggleRow({
  label,
  helper,
  value,
  onChange,
}: {
  label: string;
  helper: string;
  value: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <div className="flex items-start justify-between gap-4 rounded-lg border border-white/10 bg-slate-900/60 p-4">
      <div>
        <p className="text-sm font-semibold text-white">{label}</p>
        <p className="mt-1 text-xs text-slate-300">{helper}</p>
      </div>
      <button
        type="button"
        onClick={() => onChange(!value)}
        className={`relative h-6 w-11 rounded-full transition ${
          value ? "bg-cyan-500" : "bg-white/20"
        }`}
        aria-pressed={value}
      >
        <span
          className={`absolute top-1/2 h-5 w-5 -translate-y-1/2 rounded-full bg-white transition ${
            value ? "translate-x-[22px]" : "translate-x-[2px]"
          }`}
        />
      </button>
    </div>
  );
}

function NumberField({
  label,
  value,
  onChange,
  helper,
  min,
  max,
  step,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  helper?: string;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <div className="space-y-2 text-sm">
      <label className="block text-slate-200">{label}</label>
      <input
        type="number"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        min={min}
        max={max}
        step={step}
        className="w-full rounded-md border border-white/10 bg-slate-900 px-4 py-2 text-sm focus:border-cyan-400 focus:outline-none"
      />
      {helper ? <p className="text-xs text-slate-400">{helper}</p> : null}
    </div>
  );
}

function SliderField({
  label,
  value,
  onChange,
  helper,
  min,
  max,
  step,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  helper?: string;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <div className="space-y-2 text-sm">
      <label className="block text-slate-200">{label}</label>
      <input
        type="range"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        min={min}
        max={max}
        step={step}
        className="w-full"
      />
      {helper ? <p className="text-xs text-slate-400">{helper}</p> : null}
    </div>
  );
}

function ReviewCard({
  dataset,
  targetSummary,
  model,
  threshold,
}: {
  dataset: string;
  targetSummary: string;
  model: string;
  threshold: number;
}) {
  return (
    <div className="rounded-2xl border border-cyan-500/40 bg-cyan-500/10 p-6 text-sm text-slate-200 shadow-lg shadow-cyan-900/40">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-white">Ready to analyze?</h3>
          <p className="mt-1 text-xs text-cyan-100/80">Review selections and launch the pipeline.</p>
        </div>
        <button className="rounded-full bg-cyan-400 px-5 py-2 text-sm font-semibold text-slate-950 shadow shadow-cyan-900/40 hover:bg-cyan-300">
          Run analysis
        </button>
      </div>
      <dl className="mt-4 grid gap-3 sm:grid-cols-2">
        <div>
          <dt className="text-xs uppercase tracking-wide text-cyan-200/80">Dataset</dt>
          <dd className="text-sm text-white">{dataset || "Select a dataset"}</dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-cyan-200/80">Targets</dt>
          <dd className="text-sm text-white">{targetSummary || "No targets specified"}</dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-cyan-200/80">Model</dt>
          <dd className="text-sm text-white">{model}</dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-cyan-200/80">Threshold</dt>
          <dd className="text-sm text-white">{threshold.toFixed(2)}</dd>
        </div>
      </dl>
    </div>
  );
}

function ResultsPanel() {
  const metrics = [
    { label: "Precision", value: "0.91" },
    { label: "Recall", value: "0.87" },
    { label: "ROC AUC", value: "0.95" },
    { label: "PR AUC", value: "0.92" },
  ];

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
      <header className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-white">Run status</h2>
          <p className="text-xs text-slate-300">Monitor aggregate performance across all targets.</p>
        </div>
        <span className="rounded-full border border-emerald-400/40 bg-emerald-500/10 px-3 py-1 text-xs text-emerald-200">Completed</span>
      </header>
      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        {metrics.map((metric) => (
          <div key={metric.label} className="rounded-xl border border-white/10 bg-slate-900/60 p-4">
            <p className="text-xs uppercase tracking-wide text-slate-400">{metric.label}</p>
            <p className="mt-2 text-2xl font-semibold text-white">{metric.value}</p>
            <div className="mt-3 h-2 rounded-full bg-slate-800">
              <div className="h-full rounded-full bg-cyan-400" style={{ width: `${Number(metric.value) * 100}%` }} />
            </div>
          </div>
        ))}
      </div>
      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <ChartPlaceholder title="Confusion matrix" subtitle="TP/TN vs FP/FN" />
        <ChartPlaceholder title="Calibration curve" subtitle="Reliability of predicted probabilities" />
        <ChartPlaceholder title="ROC & PR curves" subtitle="Classifier diagnostics" className="lg:col-span-2" />
      </div>
    </div>
  );
}

function DiagnosticsPanel() {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
      <header className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-white">Target diagnostics</h2>
          <p className="text-xs text-slate-300">Inspect light curves, BLS periodograms, and phase-folded fits.</p>
        </div>
        <button className="rounded-full border border-white/15 px-4 py-1 text-xs text-slate-200 hover:border-cyan-400/40 hover:text-cyan-200">
          Switch target
        </button>
      </header>
      <div className="mt-6 space-y-4">
        <ChartPlaceholder title="Light curve" subtitle="Raw & detrended flux vs time" />
        <ChartPlaceholder title="BLS periodogram" subtitle="Power spectrum of candidate periods" />
        <ChartPlaceholder title="Phase-folded transit" subtitle="Transit model fit at detected period" />
        <ChartPlaceholder title="Learning curves" subtitle="Loss/metric vs epoch" />
        <ChartPlaceholder title="Feature importance" subtitle="SHAP values for model interpretation" />
      </div>
    </div>
  );
}

function CandidateGallery({ candidates }: { candidates: Candidate[] }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
      <header className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-white">Top candidates</h2>
          <p className="text-xs text-slate-300">Ranked by detection score with key transit parameters.</p>
        </div>
        <button className="rounded-full border border-white/15 px-4 py-1 text-xs text-slate-200 hover:border-cyan-400/40 hover:text-cyan-200">
          View all
        </button>
      </header>
      <div className="mt-6 grid gap-4 sm:grid-cols-2">
        {candidates.map((candidate) => (
          <div key={candidate.id} className="rounded-xl border border-white/10 bg-slate-900/60 p-4">
            <p className="text-sm font-semibold text-white">{candidate.id}</p>
            <p className="mt-2 text-xs text-slate-300">Score {candidate.score.toFixed(2)} | Period {candidate.period}</p>
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-slate-200">
              <MetricBadge label="Depth" value={candidate.depth} />
              <MetricBadge label="Duration" value={candidate.duration} />
              <MetricBadge label="SNR" value={candidate.snr} />
              <MetricBadge label="Threshold" value={candidate.score >= 0.5 ? "Pass" : "Check"} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function MetricBadge({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-white/10 bg-slate-800/80 px-3 py-2 text-left">
      <p className="text-[11px] uppercase tracking-wide text-slate-400">{label}</p>
      <p className="mt-1 text-sm font-medium text-white">{value}</p>
    </div>
  );
}

function DownloadsPanel() {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
      <h2 className="text-lg font-semibold text-white">Export deliverables</h2>
      <p className="mt-1 text-xs text-slate-300">Download artifacts for science review or submission packages.</p>
      <div className="mt-6 grid gap-4 sm:grid-cols-3">
        <DownloadCard title="Predictions CSV" description="targetId, y_pred, probability, period, depth, duration" status="Ready" />
        <DownloadCard title="Report PDF" description="Transit plots, metrics, and operator notes" status="Queued" />
        <DownloadCard title="Model card JSON" description="Configuration, data provenance, hyperparameters" status="Ready" />
      </div>
    </div>
  );
}

function DownloadCard({
  title,
  description,
  status,
}: {
  title: string;
  description: string;
  status: "Ready" | "Queued" | "Processing";
}) {
  return (
    <div className="flex h-full flex-col justify-between rounded-xl border border-white/10 bg-slate-900/60 p-4 text-sm">
      <div>
        <p className="text-sm font-semibold text-white">{title}</p>
        <p className="mt-2 text-xs text-slate-300">{description}</p>
      </div>
      <button
        className={`mt-4 rounded-full px-4 py-2 text-xs font-semibold transition ${
          status === "Ready"
            ? "bg-cyan-400 text-slate-950 hover:bg-cyan-300"
            : "border border-white/15 text-slate-300"
        }`}
        disabled={status !== "Ready"}
      >
        {status === "Ready" ? "Download" : status}
      </button>
    </div>
  );
}

function ChartPlaceholder({
  title,
  subtitle,
  className,
}: {
  title: string;
  subtitle: string;
  className?: string;
}) {
  return (
    <div
      className={`flex h-40 flex-col justify-between rounded-xl border border-white/10 bg-gradient-to-br from-slate-900/80 via-slate-900/40 to-slate-900/80 p-4 text-sm text-slate-200 ${className ?? ""}`}
    >
      <div>
        <p className="text-sm font-semibold text-white">{title}</p>
        <p className="text-xs text-slate-300">{subtitle}</p>
      </div>
      <div className="flex items-center justify-center text-xs text-slate-500">
        <span className="rounded border border-dashed border-white/15 px-3 py-1">Chart renders here</span>
      </div>
    </div>
  );
}


