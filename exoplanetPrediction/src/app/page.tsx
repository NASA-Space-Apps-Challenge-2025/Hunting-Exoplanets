"use client";

import Image from "next/image";
import { ChangeEvent, useEffect, useState } from "react";

type InputMode = "manual" | "upload";
type ModelType = "pretrained" | "user";

type ManualFieldKey =
  | "koi_period"
  | "koi_depth"
  | "koi_duration"
  | "koi_prad"
  | "koi_impact"
  | "koi_model_snr"
  | "koi_max_mult_ev"
  | "koi_num_transits"
  | "koi_steff"
  | "koi_srad"
  | "koi_kepmag"
  | "koi_insol"
  | "koi_teq";

type ManualField = {
  key: ManualFieldKey;
  label: string;
  helper?: string;
  placeholder?: string;
  min?: number;
  max?: number;
  step?: number;
};

type ManualFormState = Record<ManualFieldKey, string>;
type ManualRow = Record<ManualFieldKey, number>;

type ConfusionMatrix = [
  [number, number],
  [number, number]
];

type ModelMetrics = {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  roc_auc?: number;
  confusion_matrix?: ConfusionMatrix;
  [key: string]: unknown;
};

type ModelHyperparameters = {
  learning_rate?: number;
  max_depth?: number;
  num_trees?: number;
  [key: string]: unknown;
};

type ModelTrainingInfo = {
  training_samples?: number;
  features?: string[];
  trained_at?: string;
  [key: string]: unknown;
};

type ModelInfo = {
  model_type?: string;
  metrics?: ModelMetrics;
  hyperparameters?: ModelHyperparameters;
  training_info?: ModelTrainingInfo;
  timestamp?: string;
};

const METRIC_KEYS = ["accuracy", "precision", "recall", "f1_score", "roc_auc"] as const;

const METRIC_LABELS: Record<(typeof METRIC_KEYS)[number], string> = {
  accuracy: "Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1_score: "F1 Score",
  roc_auc: "ROC AUC",
};

const HYPERPARAM_LABELS: Record<keyof ModelHyperparameters, string> = {
  learning_rate: "Learning Rate",
  max_depth: "Max Depth",
  num_trees: "Number of Trees",
};


type UploadCardProps = {
  fields: ManualField[];
  error: string | null;
  info: string | null;
  onFileChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onTemplateDownload: () => void;
};

const manualFieldDefinitions: ManualField[] = [
  { key: "koi_period", label: "Orbital Period (days)", helper: "koi_period", placeholder: "10.512", min: 0, step: 0.0001 },
  { key: "koi_depth", label: "Transit Depth (ppm)", helper: "koi_depth", placeholder: "850", min: 0, step: 0.01 },
  { key: "koi_duration", label: "Transit Duration (hrs)", helper: "koi_duration", placeholder: "3.5", min: 0, step: 0.0001 },
  { key: "koi_prad", label: "Planet Radius (Earth radii)", helper: "koi_prad", placeholder: "2.1", min: 0, step: 0.0001 },
  { key: "koi_impact", label: "Impact Parameter", helper: "koi_impact", placeholder: "0.4", min: 0, max: 1, step: 0.0001 },
  { key: "koi_model_snr", label: "Model SNR", helper: "koi_model_snr", placeholder: "12.8", min: 0, step: 0.0001 },
  { key: "koi_max_mult_ev", label: "Max Multiple Events", helper: "koi_max_mult_ev", placeholder: "1", min: 0, step: 1 },
  { key: "koi_num_transits", label: "Number of Transits", helper: "koi_num_transits", placeholder: "5", min: 0, step: 1 },
  { key: "koi_steff", label: "Stellar Effective Temp (K)", helper: "koi_steff", placeholder: "5778", min: 0, step: 0.1 },
  { key: "koi_srad", label: "Stellar Radius (Solar radii)", helper: "koi_srad", placeholder: "1.0", min: 0, step: 0.0001 },
  { key: "koi_kepmag", label: "Kepler Magnitude", helper: "koi_kepmag", placeholder: "13.6", min: 0, step: 0.0001 },
  { key: "koi_insol", label: "Insolation (Earth flux)", helper: "koi_insol", placeholder: "1.2", min: 0, step: 0.0001 },
  { key: "koi_teq", label: "Equilibrium Temp (K)", helper: "koi_teq", placeholder: "980", min: 0, step: 0.1 },
];

const requiredHeaders = manualFieldDefinitions.map((field) => field.key);

function downloadCsv(content: string, filename: string) {
  const blob = new Blob([content], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

async function extractErrorMessage(response: Response): Promise<string> {
  try {
    const data = await response.json();
    if (data && typeof data.detail === 'string') {
      return data.detail;
    }
    return JSON.stringify(data, null, 2);
  } catch {
    const fallback = await response.text();
    return fallback || `Request failed with status ${response.status}`;
  }
}

function parseManualCsv(text: string): ManualRow[] {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (lines.length === 0) {
    throw new Error("CSV is empty. Include a header row and at least one data row.");
  }

  const headersRaw = lines[0].split(",").map((cell) => sanitizeCell(cell));
  const headersLower = headersRaw.map((header) => header.toLowerCase());

  const suggestions: string[] = [];

  headersRaw.forEach((header) => {
    const matchingField = manualFieldDefinitions.find((field) => field.key.toLowerCase() === header.toLowerCase());
    if (!matchingField) {
      suggestions.push(`Remove or rename "${header}" to one of: ${requiredHeaders.join(", ")}`);
    } else if (matchingField.key !== header) {
      suggestions.push(`Rename "${header}" to "${matchingField.key}"`);
    }
  });

  requiredHeaders.forEach((required) => {
    if (!headersLower.includes(required.toLowerCase())) {
      suggestions.push(`Add column "${required}" to the header row`);
    }
  });

  if (suggestions.length > 0) {
    throw new Error(suggestions.join("\n"));
  }

  const indexByKey = new Map<ManualFieldKey, number>();
  requiredHeaders.forEach((key) => {
    const idx = headersRaw.findIndex((header) => header.toLowerCase() === key.toLowerCase());
    indexByKey.set(key as ManualFieldKey, idx);
  });

  const rows: ManualRow[] = [];
  for (let lineIndex = 1; lineIndex < lines.length; lineIndex += 1) {
    const line = lines[lineIndex];
    if (line.trim().length === 0) {
      continue;
    }
    const cells = line.split(",").map((cell) => sanitizeCell(cell));
    if (cells.length < headersRaw.length) {
      throw new Error(`Row ${lineIndex + 1} has ${cells.length} values but ${headersRaw.length} columns are expected.`);
    }
    const row = {} as ManualRow;
    for (const field of manualFieldDefinitions) {
      const columnIndex = indexByKey.get(field.key);
      if (columnIndex === undefined) {
        continue;
      }
      const numericValue = Number.parseFloat(cells[columnIndex]);
      if (Number.isNaN(numericValue)) {
        throw new Error(`All values must be numeric. Check column "${field.key}" in row ${lineIndex + 1}.`);
      }
      row[field.key] = numericValue;
    }
    rows.push(row);
  }

  if (rows.length === 0) {
    throw new Error("No data rows found in the CSV.");
  }

  return rows;
}

function sanitizeCell(cell: string) {
  return cell.trim().replace(/^\"|\"$/g, "");
}

function createEmptyManualForm(): ManualFormState {
  return manualFieldDefinitions.reduce<ManualFormState>((acc, field) => {
    acc[field.key] = "";
    return acc;
  }, {} as ManualFormState);
}

function formatMetricValue(value: unknown) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "—";
  }
  if (value >= 0 && value <= 1) {
    return `${(value * 100).toFixed(2)}%`;
  }
  return Number.isInteger(value) ? value.toString() : value.toFixed(2);
}

function formatHyperparameterValue(value: unknown) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "—";
  }
  return Number.isInteger(value) ? value.toString() : value.toFixed(2);
}

function formatTimestamp(value?: string) {
  if (!value) {
    return "—";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

export default function Home() {
  const [inputMode, setInputMode] = useState<InputMode>("manual");
  const [modelType, setModelType] = useState<ModelType>("pretrained");
  const [isManualModalOpen, setManualModalOpen] = useState(false);
  const [manualForm, setManualForm] = useState<ManualFormState>(() => createEmptyManualForm());
  const [manualRows, setManualRows] = useState<ManualRow[]>([]);
  const [manualError, setManualError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadInfo, setUploadInfo] = useState<string | null>(null);
  const [uploadRowCount, setUploadRowCount] = useState<number>(0);

  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [infoLoading, setInfoLoading] = useState(false);
  const [infoError, setInfoError] = useState<string | null>(null);

  const [pretrainedFile, setPretrainedFile] = useState<File | null>(null);
  const [pretrainedSession, setPretrainedSession] = useState("");
  const [pretrainedResult, setPretrainedResult] = useState<Record<string, unknown> | null>(null);
  const [pretrainedError, setPretrainedError] = useState<string | null>(null);
  const [pretrainedLoading, setPretrainedLoading] = useState(false);
  const [userTrainFile, setUserTrainFile] = useState<File | null>(null);
  const [userHyperparams, setUserHyperparams] = useState({
    learningRate: "0.03",
    maxDepth: "8",
    numTrees: "700",
  });
  const [userTrainLoading, setUserTrainLoading] = useState(false);
  const [userTrainError, setUserTrainError] = useState<string | null>(null);
  const [userTrainResult, setUserTrainResult] = useState<Record<string, unknown> | null>(null);

  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

  const manualRowCount = manualRows.length;
  const confusionMatrix = modelInfo?.metrics?.confusion_matrix;
  const trainingFeatures = modelInfo?.training_info?.features ?? [];
  const hasTrainingFeatures = trainingFeatures.length > 0;
  const modelUpdatedAt = modelInfo?.timestamp ?? modelInfo?.training_info?.trained_at;

  useEffect(() => {
    let cancelled = false;

    async function fetchModelInfo() {
      setInfoLoading(true);
      setInfoError(null);
      try {
        const response = await fetch(new URL('/info', backendUrl).toString());
        if (!response.ok) {
          const message = await extractErrorMessage(response);
          throw new Error(message);
        }
        const data = (await response.json()) as ModelInfo;
        if (!cancelled) {
          setModelInfo(data);
        }
      } catch (error) {
        if (!cancelled) {
          setInfoError(error instanceof Error ? error.message : 'Failed to load model info.');
        }
      } finally {
        if (!cancelled) {
          setInfoLoading(false);
        }
      }
    }

    fetchModelInfo();

    return () => {
      cancelled = true;
    };
  }, [backendUrl]);
  function handleManualButtonClick() {
    setInputMode("manual");
    setManualModalOpen(true);
  }

  function handleManualClose() {
    setManualModalOpen(false);
    setManualError(null);
  }

  function handleManualFieldChange(key: ManualFieldKey, value: string) {
    setManualForm((prev) => ({ ...prev, [key]: value }));
  }

  function handleManualAddRow() {
    const emptyField = manualFieldDefinitions.find((field) => manualForm[field.key].trim() === "");
    if (emptyField) {
      setManualError(`Please provide a value for ${emptyField.helper ?? emptyField.label}.`);
      return;
    }

    const parsedRow = {} as ManualRow;
    for (const field of manualFieldDefinitions) {
      const numericValue = Number.parseFloat(manualForm[field.key]);
      if (Number.isNaN(numericValue)) {
        setManualError(`Use numeric values only for ${field.helper ?? field.label}.`);
        return;
      }
      parsedRow[field.key] = numericValue;
    }

    setManualRows((prev) => [...prev, parsedRow]);
    setManualForm(createEmptyManualForm());
    setManualError(null);
  }

  function handleManualDownloadCsv() {
    if (manualRows.length === 0) {
      setManualError("Add at least one row before downloading.");
      return;
    }

    const header = manualFieldDefinitions.map((field) => field.key).join(",");
    const body = manualRows
      .map((row) => manualFieldDefinitions.map((field) => row[field.key].toString()).join(","))
      .join("\n");
    const csv = body.length > 0 ? `${header}\n${body}` : header;

    downloadCsv(csv, "winnhacks_manual_entries.csv");
  }

  function handleUploadButtonClick() {
    setInputMode("upload");
    setUploadError(null);
  }

  function handleUploadTemplateDownload() {
    const header = manualFieldDefinitions.map((field) => field.key).join(",");
    const sampleRow = manualFieldDefinitions.map((field) => field.placeholder ?? "").join(",");
    const csv = `${header}
${sampleRow}`;
    downloadCsv(csv, "winnhacks_template.csv");
  }

  function handlePretrainedFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    setPretrainedFile(file);
    setPretrainedResult(null);
    setPretrainedError(null);
  }

  async function handlePretrainedSubmit() {
    if (!pretrainedFile) {
      setPretrainedError("Choose a CSV file before running the pretrained model.");
      return;
    }

    const formData = new FormData();
    formData.append("file", pretrainedFile);

    const inferenceUrl = new URL('/inference', backendUrl);
    if (pretrainedSession.trim().length > 0) {
      inferenceUrl.searchParams.set('session', pretrainedSession.trim());
    }

    setPretrainedLoading(true);
    setPretrainedError(null);
    setPretrainedResult(null);

    try {
      const response = await fetch(inferenceUrl.toString(), {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const message = await extractErrorMessage(response);
        throw new Error(message);
      }
      const data = (await response.json()) as Record<string, unknown>;
      setPretrainedResult(data);
    } catch (error) {
      setPretrainedError(error instanceof Error ? error.message : 'Failed to run pretrained model.');
    } finally {
      setPretrainedLoading(false);
    }
  }

  function handleUploadFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    if (!file.name.toLowerCase().endsWith(".csv")) {
      setUploadError("Please upload a CSV file with extension .csv.");
      setUploadInfo(null);
      setUploadRowCount(0);
      setPretrainedFile(null);
      event.target.value = "";
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const textContent = String(reader.result ?? "");
        const rows = parseManualCsv(textContent);
        setUploadRowCount(rows.length);
        setUploadInfo(`Validated ${rows.length} row${rows.length === 1 ? "" : "s"} from ${file.name}.`);
        setUploadError(null);
        setPretrainedFile(file);
      } catch (error) {
        setUploadInfo(null);
        setUploadRowCount(0);
        setUploadError(error instanceof Error ? error.message : "Unable to parse the provided CSV file.");
        setPretrainedFile(null);
      }
    };
    reader.onerror = () => {
      setUploadInfo(null);
      setUploadRowCount(0);
      setUploadError("Unable to read the selected file. Please try again.");
      setPretrainedFile(null);
    };
    reader.readAsText(file);

    event.target.value = "";
  }
  function handleUserTrainFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    setUserTrainFile(file);
    setUserTrainResult(null);
    setUserTrainError(null);
  }

  function handleUserHyperparamChange(key: keyof typeof userHyperparams, value: string) {
    setUserHyperparams((prev) => ({ ...prev, [key]: value }));
  }

  async function handleUserTrainSubmit() {
    if (!userTrainFile) {
      setUserTrainError("Select a CSV file before submitting retraining.");
      return;
    }

    const learningRate = Number.parseFloat(userHyperparams.learningRate);
    const maxDepth = Number.parseInt(userHyperparams.maxDepth, 10);
    const numTrees = Number.parseInt(userHyperparams.numTrees, 10);

    if (Number.isNaN(learningRate) || Number.isNaN(maxDepth) || Number.isNaN(numTrees)) {
      setUserTrainError("Provide numeric values for learning rate, max depth, and number of trees.");
      return;
    }

    const formData = new FormData();
    formData.append("file", userTrainFile);
    formData.append("learning_rate", learningRate.toString());
    formData.append("max_depth", maxDepth.toString());
    formData.append("num_trees", numTrees.toString());

    setUserTrainLoading(true);
    setUserTrainError(null);
    setUserTrainResult(null);

    try {
      const retrainUrl = new URL('/retrain', backendUrl).toString();
      const response = await fetch(retrainUrl, { method: 'POST', body: formData });
      if (!response.ok) {
        const message = await extractErrorMessage(response);
        throw new Error(message);
      }
      const data = (await response.json()) as Record<string, unknown>;
      setUserTrainResult(data);
    } catch (error) {
      setUserTrainError(error instanceof Error ? error.message : 'Retraining request failed.');
    } finally {
      setUserTrainLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-950 to-slate-900 text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-6xl flex-col gap-12 px-6 py-12 lg:px-10">
        <header className="rounded-3xl border border-cyan-400/20 bg-slate-900/60 p-8 shadow-[0_0_60px_rgba(34,211,238,0.15)]">
          <div className="flex flex-wrap items-center gap-6">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-cyan-400/50 bg-slate-950/80">
              <Image
                src="/winnhacks-logo.png"
                alt="WinnHacks logo"
                width={64}
                height={64}
                className="h-14 w-14 object-contain"
                priority
              />
            </div>
            <div className="space-y-2">
              <h1 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">
                WinnHacks - Exoplanet Prediction Model
              </h1>
              <p className="text-sm uppercase tracking-[0.4em] text-cyan-400">Competition Project</p>
            </div>
            <div className="ml-auto flex flex-wrap gap-2 text-xs uppercase tracking-[0.4em]">
              {manualRowCount > 0 ? (
                <span className="rounded-full border border-cyan-400/30 bg-cyan-400/10 px-4 py-1 text-cyan-200">
                  {manualRowCount} manual row{manualRowCount === 1 ? "" : "s"}
                </span>
              ) : null}
              {uploadRowCount > 0 ? (
                <span className="rounded-full border border-emerald-400/30 bg-emerald-400/10 px-4 py-1 text-emerald-200">
                  {uploadRowCount} uploaded row{uploadRowCount === 1 ? "" : "s"}
                </span>
              ) : null}
            </div>
          </div>
          <div className="mt-8 grid gap-4 lg:grid-cols-[2fr_3fr]">
            <div className="rounded-2xl border border-cyan-400/20 bg-slate-950/70 p-6 text-sm text-slate-300">
              <h2 className="text-base font-semibold text-cyan-300">Scope of the Website</h2>
              <p className="mt-3 leading-relaxed text-slate-300">
                Configure, visualize, and compare exoplanet detection models for the WinnHacks hackathon. Tune hyperparameters,
                upload custom experiment logs, and benchmark against our pretrained baseline inside a responsive dark interface.
              </p>
            </div>
            <div className="rounded-2xl border border-cyan-400/20 bg-slate-950/40 p-6 text-sm text-slate-300">
              <p className="leading-relaxed">
                This cockpit ties together data ingestion, model selection, and performance analytics. Use it to orchestrate rapid
                experimentation, capture winning configurations, and craft a presentation-ready story for the judges.
              </p>
            </div>
          </div>
        </header>

        <main className="space-y-12">
          <section className="rounded-3xl border border-cyan-400/20 bg-slate-900/70 p-8 shadow-[0_0_50px_rgba(34,211,238,0.1)]">
            <div className="flex flex-col items-center gap-6 text-center">
              <div>
                <p className="text-sm uppercase tracking-[0.6em] text-cyan-500">Input Pipeline</p>
                <h2 className="mt-2 text-xl font-semibold text-white">Hyperparameters of the ML model we have</h2>
              </div>

              <div className="mt-6 flex flex-col items-center justify-center gap-4 text-sm sm:flex-row">
                <ActionButton label="Enter Manually" active={inputMode === "manual"} onClick={handleManualButtonClick} />
                <span className="text-xs uppercase tracking-[0.6em] text-slate-400">or</span>
                <ActionButton label="Upload CSV" active={inputMode === "upload"} onClick={handleUploadButtonClick} />
              </div>

              <div className="flex flex-wrap items-center justify-center gap-3 text-sm">
                <ToggleButton
                  label="Pretrained"
                  active={modelType === "pretrained"}
                  onClick={() => setModelType("pretrained")}
                />
                <ToggleButton
                  label="User Trained"
                  active={modelType === "user"}
                  onClick={() => setModelType("user")}
                />
              </div>
            </div>

            {inputMode === "upload" ? (
              <div className="mt-10">
                <UploadCard
                  fields={manualFieldDefinitions}
                  error={uploadError}
                  info={uploadInfo}
                  onFileChange={handleUploadFileChange}
                  onTemplateDownload={handleUploadTemplateDownload}
                />
              </div>
            ) : null}
            <div className="mt-12 flex justify-center">
              {modelType === "pretrained" ? (
                <div className="flex w-full max-w-3xl flex-col items-center space-y-6 text-center rounded-2xl border border-cyan-400/15 bg-slate-950/40 p-8 shadow-[0_0_35px_rgba(34,211,238,0.1)]">
                  <header className="flex items-center justify-center gap-3 text-sm font-semibold text-cyan-200">
                    <span className="rounded-full border border-cyan-400/40 bg-cyan-400/10 px-3 py-1 text-xs uppercase tracking-widest">
                      Pretrained Flow
                    </span>
                    <span className="text-slate-400">Baseline evaluation</span>
                  </header>
                  <div className="w-full space-y-4 text-left">
                    <p className="text-xs uppercase tracking-[0.4em] text-cyan-400">Model snapshot</p>
                    <div className="rounded-2xl border border-cyan-400/20 bg-slate-900/70 p-6 text-left text-slate-200">
                      {infoLoading ? (
                        <div className="flex items-center gap-3 text-sm text-slate-300">
                          <span className="h-2 w-2 animate-pulse rounded-full bg-cyan-300" />
                          <span>Loading model metadata...</span>
                        </div>
                      ) : infoError ? (
                        <div className="rounded-xl border border-rose-400/40 bg-rose-500/10 p-4 text-sm text-rose-200">
                          {infoError}
                        </div>
                      ) : modelInfo ? (
                        <div className="space-y-6">
                          <div className="flex flex-wrap items-start justify-between gap-4">
                            <div>
                              <p className="text-[11px] uppercase tracking-[0.4em] text-cyan-400">Model type</p>
                              <p className="mt-2 text-xl font-semibold text-white">
                                {modelInfo.model_type ?? "Unknown model"}
                              </p>
                            </div>
                            <div className="text-right text-xs text-slate-400">
                              <p className="text-[10px] uppercase tracking-[0.4em] text-cyan-400">Last updated</p>
                              <p className="mt-2 text-sm text-slate-200">{formatTimestamp(modelUpdatedAt)}</p>
                            </div>
                          </div>
                          <div className="space-y-3">
                            <p className="text-[11px] uppercase tracking-[0.4em] text-cyan-400">Core metrics</p>
                            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                              {METRIC_KEYS.map((metricKey) => {
                                const value = modelInfo.metrics?.[metricKey];
                                const formatted = formatMetricValue(value);
                                const valueClass = typeof value === "number" && !Number.isNaN(value)
                                  ? "text-white"
                                  : "text-slate-500";
                                return (
                                  <div
                                    key={metricKey}
                                    className="rounded-xl border border-cyan-400/15 bg-slate-950/60 p-4"
                                  >
                                    <p className="text-[11px] uppercase tracking-[0.3em] text-cyan-300">
                                      {METRIC_LABELS[metricKey]}
                                    </p>
                                    <p className={`mt-2 text-2xl font-semibold ${valueClass}`}>{formatted}</p>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                          {confusionMatrix ? (
                            <div className="space-y-3">
                              <p className="text-[11px] uppercase tracking-[0.4em] text-cyan-400">Confusion matrix</p>
                              <div className="overflow-hidden rounded-xl border border-cyan-400/15 bg-slate-950/60">
                                <table className="w-full text-sm text-slate-200">
                                  <thead className="bg-slate-900/70 text-xs uppercase tracking-[0.3em] text-slate-400">
                                    <tr>
                                      <th className="px-3 py-2 text-left" />
                                      <th className="px-3 py-2 text-center">Predicted Negative</th>
                                      <th className="px-3 py-2 text-center">Predicted Positive</th>
                                    </tr>
                                  </thead>
                                  <tbody className="divide-y divide-slate-800/70">
                                    {confusionMatrix.map((row, rowIndex) => (
                                      <tr key={`confusion-row-${rowIndex}`} className="text-center">
                                        <th className="px-3 py-2 text-left text-xs uppercase tracking-[0.3em] text-cyan-300">
                                          {rowIndex === 0 ? "Actual Negative" : "Actual Positive"}
                                        </th>
                                        {row.map((cell, columnIndex) => (
                                          <td
                                            key={`confusion-cell-${rowIndex}-${columnIndex}`}
                                            className="px-3 py-2 font-mono text-base text-white"
                                          >
                                            {cell}
                                          </td>
                                        ))}
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            </div>
                          ) : null}
                          <div className="space-y-3">
                            <p className="text-[11px] uppercase tracking-[0.4em] text-cyan-400">Hyperparameters</p>
                            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                              {(Object.keys(HYPERPARAM_LABELS) as (keyof ModelHyperparameters)[]).map((paramKey) => {
                                const value = modelInfo.hyperparameters?.[paramKey];
                                const formatted = formatHyperparameterValue(value);
                                const valueClass = typeof value === "number" && !Number.isNaN(value)
                                  ? "text-white"
                                  : "text-slate-500";
                                return (
                                  <div
                                    key={paramKey}
                                    className="rounded-xl border border-cyan-400/15 bg-slate-950/60 p-4"
                                  >
                                    <p className="text-[11px] uppercase tracking-[0.3em] text-cyan-300">
                                      {HYPERPARAM_LABELS[paramKey]}
                                    </p>
                                    <p className={`mt-2 text-xl font-semibold ${valueClass}`}>{formatted}</p>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                          <div className="space-y-3">
                            <p className="text-[11px] uppercase tracking-[0.4em] text-cyan-400">Training info</p>
                            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                              <div className="rounded-xl border border-cyan-400/15 bg-slate-950/60 p-4">
                                <p className="text-[11px] uppercase tracking-[0.3em] text-cyan-300">Training samples</p>
                                <p className="mt-2 text-xl font-semibold text-white">
                                  {modelInfo.training_info?.training_samples ?? "—"}
                                </p>
                              </div>
                              <div className="rounded-xl border border-cyan-400/15 bg-slate-950/60 p-4 sm:col-span-2 lg:col-span-1">
                                <p className="text-[11px] uppercase tracking-[0.3em] text-cyan-300">Trained at</p>
                                <p className="mt-2 text-sm text-slate-200">
                                  {formatTimestamp(modelInfo.training_info?.trained_at)}
                                </p>
                              </div>
                            </div>
                            {hasTrainingFeatures ? (
                              <div>
                                <p className="text-[11px] uppercase tracking-[0.3em] text-cyan-300">
                                  Features ({trainingFeatures.length})
                                </p>
                                <div className="mt-2 flex flex-wrap gap-2">
                                  {trainingFeatures.map((feature) => (
                                    <span
                                      key={feature}
                                      className="rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-1 text-xs text-cyan-200"
                                    >
                                      {feature}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            ) : null}
                          </div>
                        </div>
                      ) : (
                        <div className="text-sm text-slate-400">Model details are unavailable right now.</div>
                      )}
                    </div>
                  </div>
                  <div className="w-full rounded-2xl border border-cyan-400/20 bg-slate-950/70 p-6 text-left text-sm text-slate-200">
                    <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
                      <label className="flex w-full flex-col gap-2 text-xs uppercase tracking-[0.4em]">
                        <span className="text-cyan-400">Upload CSV</span>
                        <input
                          type="file"
                          accept=".csv"
                          onChange={handlePretrainedFileChange}
                          className="w-full rounded-xl border border-cyan-400/30 bg-slate-950/60 px-4 py-3 text-sm text-white focus:border-cyan-300 focus:outline-none focus:ring-2 focus:ring-cyan-500/40"
                        />
                        <span className="text-[10px] text-slate-500">Required columns: {requiredHeaders.join(', ')}</span>
                      </label>
                      <label className="flex w-full max-w-xs flex-col gap-2 text-xs uppercase tracking-[0.4em]">
                        <span className="text-cyan-400">Session (optional)</span>
                        <input
                          type="text"
                          value={pretrainedSession}
                          onChange={(event) => setPretrainedSession(event.target.value)}
                          placeholder="session id"
                          className="w-full rounded-xl border border-cyan-400/30 bg-slate-950/60 px-4 py-3 text-sm text-white focus:border-cyan-300 focus:outline-none focus:ring-2 focus:ring-cyan-500/40"
                        />
                        <span className="text-[10px] text-slate-500">Leave blank to use the base model.</span>
                      </label>
                    </div>
                    <div className="mt-4 flex flex-wrap items-center gap-3">
                      <button
                        type="button"
                        onClick={handlePretrainedSubmit}
                        disabled={pretrainedLoading}
                        className={`rounded-full px-5 py-2 text-sm font-semibold transition ${pretrainedLoading ? 'bg-cyan-400/30 text-slate-600' : 'bg-cyan-500 text-slate-950 shadow-[0_0_25px_rgba(34,211,238,0.35)] hover:bg-cyan-400'}`}
                      >
                        {pretrainedLoading ? 'Running...' : 'Run Pretrained Model'}
                      </button>
                      {pretrainedFile ? (
                        <span className="text-xs text-slate-400">Selected: {pretrainedFile.name}</span>
                      ) : null}
                    </div>
                    {pretrainedError ? <p className="mt-3 text-sm text-rose-300">{pretrainedError}</p> : null}
                    {pretrainedResult ? (
                      <div className="mt-4 rounded-2xl border border-cyan-400/20 bg-slate-950/80 p-4 text-left">
                        <p className="text-sm font-semibold text-cyan-200">Inference response</p>
                        <pre className="mt-2 max-h-64 overflow-auto rounded-xl bg-slate-900/70 p-4 text-xs text-slate-100">
{JSON.stringify(pretrainedResult, null, 2)}
                        </pre>
                      </div>
                    ) : null}
                  </div>

                  <div className="flex flex-col items-center gap-4 text-cyan-300">
                    <ArrowDown />
                    <div className="w-full rounded-2xl border border-cyan-400/30 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6 text-center shadow-[0_0_40px_rgba(34,211,238,0.12)]">
                      <p className="text-sm uppercase tracking-[0.4em] text-cyan-400">Result from our model</p>
                      <p className="mt-3 text-xl font-semibold text-white">ROC 0.96 | F1 0.89 | Latency 42s</p>
                      <p className="mt-2 text-xs text-slate-400">Auto-generated diagnostic deck with candidate shortlist</p>
                    </div>
                  </div>
                </div>
              ) : null}

              {modelType === "user" ? (
                <div className="flex w-full max-w-3xl flex-col items-center space-y-6 text-center rounded-2xl border border-cyan-400/15 bg-slate-950/40 p-8 shadow-[0_0_35px_rgba(34,211,238,0.1)]">
                  <header className="flex items-center justify-center gap-3 text-sm font-semibold text-cyan-200">
                    <span className="rounded-full border border-cyan-400/40 bg-cyan-400/10 px-3 py-1 text-xs uppercase tracking-widest">
                      User Trained Flow
                    </span>
                    <span className="text-slate-400">Customizable exploration</span>
                  </header>
                  <div className="w-full rounded-2xl border border-cyan-400/30 bg-slate-950/70 p-6 text-left text-sm text-slate-200">
                    <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
                      <label className="flex w-full flex-col gap-2 text-xs uppercase tracking-[0.4em]">
                        <span className="text-cyan-400">Training CSV</span>
                        <input
                          type="file"
                          accept=".csv"
                          onChange={handleUserTrainFileChange}
                          className="w-full rounded-xl border border-cyan-400/30 bg-slate-950/60 px-4 py-3 text-sm text-white focus:border-cyan-300 focus:outline-none focus:ring-2 focus:ring-cyan-500/40"
                        />
                        <span className="text-[10px] text-slate-500">Required columns: {requiredHeaders.join(', ')}</span>
                      </label>
                      <div className="rounded-xl border border-cyan-400/20 bg-slate-900/60 p-4 text-xs text-slate-300">
                        <p className="text-sm font-semibold text-cyan-200">Hyperparameters</p>
                        <p className="mt-1 leading-relaxed">Adjust learning rate, tree depth, and ensemble size before retraining.</p>
                      </div>
                    </div>
                    <div className="mt-6 grid gap-4 sm:grid-cols-3">
                      <InputField
                        label="Learning Rate"
                        helper="Float between 0 and 1"
                        placeholder="0.03"
                        type="number"
                        min={0}
                        max={1}
                        step={0.0001}
                        value={userHyperparams.learningRate}
                        onChange={(value) => handleUserHyperparamChange("learningRate", value)}
                      />
                      <InputField
                        label="Max Tree Depth"
                        helper="Integer (1-2048)"
                        placeholder="8"
                        type="number"
                        min={1}
                        max={2048}
                        value={userHyperparams.maxDepth}
                        onChange={(value) => handleUserHyperparamChange("maxDepth", value)}
                      />
                      <InputField
                        label="Number of Trees"
                        helper="Up to 5000 estimators"
                        placeholder="700"
                        type="number"
                        min={1}
                        max={5000}
                        value={userHyperparams.numTrees}
                        onChange={(value) => handleUserHyperparamChange("numTrees", value)}
                      />
                    </div>
                    <div className="mt-4 flex flex-wrap items-center gap-3">
                      <button
                        type="button"
                        onClick={handleUserTrainSubmit}
                        disabled={userTrainLoading}
                        className={`rounded-full px-5 py-2 text-sm font-semibold transition ${userTrainLoading ? 'bg-cyan-400/30 text-slate-600' : 'bg-cyan-500 text-slate-950 shadow-[0_0_25px_rgba(34,211,238,0.35)] hover:bg-cyan-400'}`}
                      >
                        {userTrainLoading ? 'Retraining...' : 'Submit for Retraining'}
                      </button>
                      {userTrainFile ? (
                        <span className="text-xs text-slate-400">Selected: {userTrainFile.name}</span>
                      ) : null}
                    </div>
                    {userTrainError ? <p className="mt-3 text-sm text-rose-300">{userTrainError}</p> : null}
                    {userTrainResult ? (
                      <div className="mt-4 rounded-2xl border border-cyan-400/20 bg-slate-950/80 p-4 text-left">
                        <p className="text-sm font-semibold text-cyan-200">Retrain response</p>
                        <pre className="mt-2 max-h-64 overflow-auto rounded-xl bg-slate-900/70 p-4 text-xs text-slate-100">
{JSON.stringify(userTrainResult, null, 2)}
                        </pre>
                      </div>
                    ) : null}
                  </div>
                  <div className="flex flex-col items-center gap-4 text-cyan-300">
                    <ArrowDown />
                    <div className="w-full rounded-2xl border border-cyan-400/30 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6 text-center shadow-[0_0_40px_rgba(34,211,238,0.12)]">
                      <p className="text-sm uppercase tracking-[0.4em] text-cyan-400">Results and Graphs</p>
                      <p className="mt-3 text-xl font-semibold text-white">
                        {userTrainResult ? 'Review the metrics above to populate dashboards.' : 'Interactive ROC, PR, SHAP, and transit fits'}
                      </p>
                      <p className="mt-2 text-xs text-slate-400">
                        {userTrainResult ? 'Use the response payload to drive custom visualizations.' : 'Exportable as slides, CSVs, and WinnHacks report pack'}
                      </p>
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </section>
        </main>
      </div>
      <ManualInputModal
        open={isManualModalOpen}
        onClose={handleManualClose}
        fields={manualFieldDefinitions}
        formValues={manualForm}
        onFieldChange={handleManualFieldChange}
        onAddRow={handleManualAddRow}
        error={manualError}
        rows={manualRows}
        onDownload={handleManualDownloadCsv}
      />
    </div>
  );
}

function ActionButton({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`group flex min-w-[180px] items-center justify-center gap-2 rounded-2xl border px-6 py-4 text-sm font-semibold transition
        ${active ? "border-cyan-400 bg-cyan-400/20 text-white shadow-[0_0_30px_rgba(34,211,238,0.35)]" : "border-white/10 bg-slate-950/60 text-slate-300 hover:border-cyan-400/50 hover:text-cyan-200"}`}
    >
      {label}
    </button>
  );
}

function ToggleButton({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-full px-6 py-2 text-sm font-semibold transition
        ${active ? "bg-cyan-500 text-slate-950 shadow-[0_0_20px_rgba(34,211,238,0.45)]" : "bg-slate-950/60 text-slate-300 hover:bg-cyan-500/20 hover:text-cyan-200"}`}
    >
      {label}
    </button>
  );
}

function InputField({
  label,
  helper,
  placeholder,
  type = "text",
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string;
  helper?: string;
  placeholder?: string;
  type?: string;
  min?: number;
  max?: number;
  step?: number;
  value?: string;
  onChange?: (value: string) => void;
}) {
  return (
    <label className="block space-y-2 text-center text-sm">
      <span className="text-xs uppercase tracking-[0.4em] text-cyan-400">{label}</span>
      <input
        type={type}
        min={min}
        max={max}
        step={step}
        placeholder={placeholder}
        value={value}
        onChange={onChange ? (event) => onChange(event.target.value) : undefined}
        className="w-full rounded-xl border border-cyan-400/30 bg-slate-950/70 px-4 py-3 text-center text-sm text-white placeholder:text-slate-500 focus:border-cyan-300 focus:outline-none focus:ring-2 focus:ring-cyan-500/40"
      />
      {helper ? <span className="block text-xs text-slate-400">{helper}</span> : null}
    </label>
  );
}

function ArrowDown() {
  return (
    <div className="flex flex-col items-center text-cyan-400">
      <div className="h-12 w-px bg-gradient-to-b from-transparent via-cyan-500 to-transparent" />
      <span className="text-2xl">v</span>
    </div>
  );
}

type ManualInputModalProps = {
  open: boolean;
  onClose: () => void;
  fields: ManualField[];
  formValues: ManualFormState;
  onFieldChange: (key: ManualFieldKey, value: string) => void;
  onAddRow: () => void;
  error: string | null;
  rows: ManualRow[];
  onDownload: () => void;
};

function ManualInputModal({
  open,
  onClose,
  fields,
  formValues,
  onFieldChange,
  onAddRow,
  error,
  rows,
  onDownload,
}: ManualInputModalProps) {
  if (!open) {
    return null;
  }

  const hasRows = rows.length > 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 py-10">
      <div className="relative w-full max-w-5xl overflow-hidden rounded-3xl border border-cyan-400/30 bg-slate-950/95 shadow-[0_0_50px_rgba(34,211,238,0.25)]">
        <header className="flex items-start justify-between gap-4 border-b border-cyan-400/20 bg-slate-950/80 px-8 py-6">
          <div>
            <p className="text-xs uppercase tracking-[0.6em] text-cyan-400">Manual Entry</p>
            <h3 className="mt-2 text-2xl font-semibold text-white">Add KOI Feature Rows</h3>
            <p className="mt-2 text-sm text-slate-300">
              Provide numeric values for each KOI feature. Every feature must be filled for each row before it can be added.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full border border-white/10 px-3 py-1 text-sm text-slate-300 transition hover:border-cyan-400/50 hover:text-cyan-200"
          >
            Close
          </button>
        </header>

        <div className="max-h-[70vh] overflow-y-auto px-8 py-6">
          <div className="grid gap-4 sm:grid-cols-2">
            {fields.map((field) => (
              <label key={field.key} className="flex flex-col gap-2 text-left text-sm">
                <span className="font-medium text-white">{field.helper ?? field.label}</span>
                <input
                  type="number"
                  min={field.min}
                  max={field.max}
                  step={field.step ?? 0.0001}
                  placeholder={field.placeholder}
                  value={formValues[field.key]}
                  onChange={(event) => onFieldChange(field.key, event.target.value)}
                  className="rounded-xl border border-cyan-400/30 bg-slate-900 px-4 py-3 text-sm text-white placeholder:text-slate-500 focus:border-cyan-300 focus:outline-none focus:ring-2 focus:ring-cyan-500/40"
                />
                <span className="text-xs uppercase tracking-[0.3em] text-slate-500">{field.key}</span>
              </label>
            ))}
          </div>

          {error ? <p className="mt-4 text-sm text-rose-300">{error}</p> : null}

          <div className="mt-6 flex flex-wrap items-center justify-between gap-3">
            <button
              type="button"
              onClick={onAddRow}
              className="rounded-full bg-cyan-500 px-5 py-2 text-sm font-semibold text-slate-950 shadow-[0_0_25px_rgba(34,211,238,0.35)] transition hover:bg-cyan-400"
            >
              Add Row
            </button>
            <button
              type="button"
              onClick={onDownload}
              disabled={!hasRows}
              className={`rounded-full px-5 py-2 text-sm font-semibold transition ${
                hasRows
                  ? "bg-emerald-500 text-slate-950 shadow-[0_0_25px_rgba(52,211,153,0.35)] hover:bg-emerald-400"
                  : "border border-white/15 bg-transparent text-slate-500"
              }`}
            >
              Download CSV
            </button>
          </div>

          {hasRows ? (
            <div className="mt-6 overflow-x-auto rounded-2xl border border-cyan-400/20 bg-slate-950/80">
              <table className="min-w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-900/80 text-cyan-200">
                  <tr>
                    {fields.map((field) => (
                      <th key={field.key} className="px-4 py-3 font-semibold uppercase tracking-[0.3em]">
                        {field.key}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, rowIndex) => (
                    <tr key={rowIndex} className="odd:bg-slate-900/60">
                      {fields.map((field) => (
                        <td key={field.key} className="px-4 py-2 text-slate-200">
                          {row[field.key]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="mt-6 text-sm text-slate-400">No rows added yet. Fill the fields above and click &ldquo;Add Row&rdquo;.</p>
          )}
        </div>
      </div>
    </div>
  );
}









function UploadCard({ fields, error, info, onFileChange, onTemplateDownload }: UploadCardProps) {
  return (
    <div className="rounded-2xl border border-cyan-400/20 bg-slate-950/70 p-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="text-left">
          <p className="text-xs uppercase tracking-[0.6em] text-cyan-400">Upload Dataset</p>
          <h3 className="mt-2 text-lg font-semibold text-white">Import KOI feature CSV</h3>
          <p className="mt-2 text-sm text-slate-300">
            Ensure the header row uses the exact feature names listed below. Column order does not matter, but the labels must
            match.
          </p>
        </div>
        <button
          type="button"
          onClick={onTemplateDownload}
          className="rounded-full border border-cyan-400/40 px-4 py-2 text-sm font-semibold text-cyan-200 transition hover:border-cyan-300 hover:text-cyan-100"
        >
          Download template CSV
        </button>
      </div>

      <div className="mt-6 flex flex-col gap-6 lg:flex-row">
        <label className="flex w-full max-w-sm cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-cyan-400/40 bg-slate-950/60 p-6 text-center text-sm text-slate-200 transition hover:border-cyan-300 hover:text-cyan-100">
          <span className="text-base font-semibold text-white">Select CSV file</span>
          <span className="text-xs text-slate-400">Accepted columns only · .csv</span>
          <input type="file" accept=".csv" className="hidden" onChange={onFileChange} />
        </label>
        <div className="flex-1 rounded-2xl border border-white/10 bg-slate-950/50 p-6 text-left text-xs text-slate-300">
          <p className="text-sm font-semibold text-cyan-200">Required features</p>
          <ul className="mt-3 grid gap-2 sm:grid-cols-2">
            {fields.map((field) => (
              <li key={field.key} className="rounded-lg border border-cyan-400/10 bg-slate-900/60 px-3 py-2 font-mono text-[11px]">
                {field.key}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {error ? <p className="mt-4 text-sm text-rose-300">{error}</p> : null}
      {info ? <p className="mt-2 text-sm text-emerald-300">{info}</p> : null}
    </div>
  );
}
