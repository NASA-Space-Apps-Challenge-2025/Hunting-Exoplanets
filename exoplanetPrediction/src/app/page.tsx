"use client";

import { useState } from "react";

type InputMode = "manual" | "upload";
type ModelType = "pretrained" | "user";

export default function Home() {
  const [inputMode, setInputMode] = useState<InputMode>("manual");
  const [modelType, setModelType] = useState<ModelType>("pretrained");

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-950 to-slate-900 text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-6xl flex-col gap-12 px-6 py-12 lg:px-10">
        <header className="rounded-3xl border border-cyan-400/20 bg-slate-900/60 p-8 shadow-[0_0_60px_rgba(34,211,238,0.15)]">
          <div className="flex flex-wrap items-center gap-6">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-cyan-400/50 bg-slate-950/80 text-lg font-semibold text-cyan-300">
              Logo
            </div>
            <div className="space-y-2">
              <h1 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">
                WinnHacks – Exoplanet Prediction Model
              </h1>
              <p className="text-sm uppercase tracking-[0.4em] text-cyan-400">Competition Project</p>
            </div>
          </div>
          <div className="mt-8 grid gap-4 lg:grid-cols-[2fr_3fr]">
            <div className="rounded-2xl border border-cyan-400/20 bg-slate-950/70 p-6 text-sm text-slate-300">
              <h2 className="text-base font-semibold text-cyan-300">Scope of the Website</h2>
              <p className="mt-3 leading-relaxed text-slate-300">
                Configure, visualize, and compare exoplanet detection models for the WinnHacks hackathon. Tune hyperparameters,
                upload custom experiment logs, and benchmark against our pretrained baseline — all inside a responsive dark
                interface.
              </p>
            </div>
            <div className="rounded-2xl border border-cyan-400/20 bg-slate-950/40 p-6 text-sm text-slate-300">
              <p className="leading-relaxed">
                This cockpit ties together data ingestion, model selection, and performance analytics. Use it to orchestrate
                rapid experimentation, capture winning configurations, and craft a presentation-ready story for the judges.
              </p>
            </div>
          </div>
        </header>

        <main className="space-y-12">
          <section className="rounded-3xl border border-cyan-400/20 bg-slate-900/70 p-8 shadow-[0_0_50px_rgba(34,211,238,0.1)]">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="text-sm uppercase tracking-[0.6em] text-cyan-500">Input Pipeline</p>
                <h2 className="mt-2 text-xl font-semibold text-white">Hyperparameters of the ML model we have</h2>
              </div>
            </div>

            <div className="mt-8 flex flex-col items-center gap-4 text-sm sm:flex-row">
              <ActionButton
                label="Enter Manually"
                active={inputMode === "manual"}
                onClick={() => setInputMode("manual")}
              />
              <span className="text-xs uppercase tracking-[0.6em] text-slate-400">or</span>
              <ActionButton
                label="Upload ⬇️"
                active={inputMode === "upload"}
                onClick={() => setInputMode("upload")}
              />
            </div>

            <div className="mt-8 flex flex-wrap gap-3 text-sm">
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

            <div className="mt-10 grid gap-8 lg:grid-cols-2">
              <div className="space-y-6">
                <header className="flex items-center gap-3 text-sm font-semibold text-cyan-200">
                  <span className="rounded-full border border-cyan-400/40 bg-cyan-400/10 px-3 py-1 text-xs uppercase tracking-widest">
                    Pretrained Flow
                  </span>
                  <span className="text-slate-400">Baseline evaluation</span>
                </header>
                <div className="rounded-2xl border border-cyan-400/20 bg-slate-950/60 p-6 text-sm text-slate-300">
                  <p>
                    Deploy our curated baseline trained on historical TESS + Kepler light curves. One click delivers vetted metrics
                    and highlights comparative gains from your custom experiments.
                  </p>
                </div>
                <div className="flex flex-col items-center gap-4 text-cyan-300">
                  <ArrowDown />
                  <div className="w-full rounded-2xl border border-cyan-400/30 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6 text-center shadow-[0_0_40px_rgba(34,211,238,0.12)]">
                    <p className="text-sm uppercase tracking-[0.4em] text-cyan-400">Result from our model</p>
                    <p className="mt-3 text-xl font-semibold text-white">ROC 0.96 · F1 0.89 · Latency 42s</p>
                    <p className="mt-2 text-xs text-slate-400">Auto-generated diagnostic deck + candidate shortlist</p>
                  </div>
                </div>
              </div>

              <div className="space-y-6">
                <header className="flex items-center gap-3 text-sm font-semibold text-cyan-200">
                  <span className="rounded-full border border-cyan-400/40 bg-cyan-400/10 px-3 py-1 text-xs uppercase tracking-widest">
                    User Trained Flow
                  </span>
                  <span className="text-slate-400">Customizable exploration</span>
                </header>
                <div className="rounded-2xl border border-cyan-400/30 bg-slate-950/70 p-6">
                  <p className="text-sm uppercase tracking-[0.4em] text-cyan-400">Hyperparameters</p>
                  <div className="mt-6 grid gap-4">
                    <InputField
                      label="Learning Rate"
                      helper="Float between 0 and 1"
                      placeholder="0.001"
                      type="number"
                      min={0}
                      max={1}
                      step={0.0001}
                    />
                    <InputField
                      label="Max Tree Depth"
                      helper="Integer (1 – 2048)"
                      placeholder="12"
                      type="number"
                      min={1}
                      max={2048}
                    />
                    <InputField
                      label="Number of Trees"
                      helper="Up to 5000 estimators"
                      placeholder="500"
                      type="number"
                      min={1}
                      max={5000}
                    />
                    <InputField
                      label="Hessian Gain"
                      helper="Custom optimization signal"
                      placeholder="Auto-detect"
                    />
                  </div>
                </div>

                <div className="flex flex-col items-center gap-4 text-cyan-300">
                  <ArrowDown />
                  <div className="w-full rounded-2xl border border-cyan-400/30 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-6 text-center shadow-[0_0_40px_rgba(34,211,238,0.12)]">
                    <p className="text-sm uppercase tracking-[0.4em] text-cyan-400">Results and Graphs</p>
                    <p className="mt-3 text-xl font-semibold text-white">Interactive ROC · PR · SHAP · Transit fits</p>
                    <p className="mt-2 text-xs text-slate-400">Exportable as slides, CSVs, and WinnHacks report pack</p>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </main>
      </div>
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
}: {
  label: string;
  helper?: string;
  placeholder?: string;
  type?: string;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <label className="block space-y-2 text-sm">
      <span className="text-xs uppercase tracking-[0.4em] text-cyan-400">{label}</span>
      <input
        type={type}
        min={min}
        max={max}
        step={step}
        placeholder={placeholder}
        className="w-full rounded-xl border border-cyan-400/30 bg-slate-950/70 px-4 py-3 text-sm text-white placeholder:text-slate-500 focus:border-cyan-300 focus:outline-none focus:ring-2 focus:ring-cyan-500/40"
      />
      {helper ? <span className="text-xs text-slate-400">{helper}</span> : null}
    </label>
  );
}

function ArrowDown() {
  return (
    <div className="flex flex-col items-center text-cyan-400">
      <div className="h-12 w-px bg-gradient-to-b from-transparent via-cyan-500 to-transparent" />
      <span className="text-2xl">▼</span>
    </div>
  );
}
