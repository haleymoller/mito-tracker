"use client";
import { useEffect, useState } from "react";
import Image from "next/image";

export default function Home() {
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
  const [file, setFile] = useState<File | null>(null);
  const [overlayUrl, setOverlayUrl] = useState<string | null>(null);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [tta, setTta] = useState(false);
  const [useLLM, setUseLLM] = useState(true);
  const [pixelSizeNm, setPixelSizeNm] = useState<string>("");
  const [metrics, setMetrics] = useState<any[]>([]);
  const [segRan, setSegRan] = useState(false);
  const [loading, setLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [exampleId, setExampleId] = useState<string>("");
  const [analysisText, setAnalysisText] = useState<string | null>(null);
  const DEFAULT_EXAMPLE_NM_PER_PX = "2.288"; // 2.288 nm/px

  // Auto-fill pixel size for curated examples
  useEffect(() => {
    if (exampleId) {
      setPixelSizeNm(DEFAULT_EXAMPLE_NM_PER_PX);
    }
  }, [exampleId]);

  async function handleSeg() {
    if (!file && !exampleId) return;
    setLoading(true);
    const form = new FormData();
    if (exampleId) {
      form.append("example_id", exampleId);
    } else if (file) {
      form.append("image", file);
    }
    form.append("threshold", String(threshold));
    form.append("tta", String(tta));
    form.append("use_llm", String(useLLM));
    const pxNm = (exampleId && pixelSizeNm.trim().length === 0)
      ? DEFAULT_EXAMPLE_NM_PER_PX
      : pixelSizeNm;
    if (pxNm.trim().length > 0) form.append("pixel_size_nm", pxNm);
    // request LLM short analysis for every run
    form.append("analyze_text", "true");
    try {
      const res = await fetch(API_URL + "/seg", { method: "POST", body: form });
      if (!res.ok) {
        const txt = await res.text();
        console.error("/seg error", res.status, txt);
        alert("Segmentation failed: " + res.status);
        setLoading(false);
        return;
      }
      const out = await res.json();
      setOverlayUrl(out.overlay_png_b64 ? "data:image/png;base64," + out.overlay_png_b64 : null);
      setMaskUrl(out.mask_png_b64 ? "data:image/png;base64," + out.mask_png_b64 : null);
      setMetrics(Array.isArray(out.metrics) ? out.metrics : []);
      setAnalysisText(out.analysis_text ?? null);
      setSegRan(true);
    } catch (e) {
      console.error(e);
      alert("Network error during segmentation");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-transparent">
      <section className="bg-transparent border-b">
        <div className="max-w-6xl mx-auto px-4 py-12 text-center">
          <div className="flex items-center justify-center gap-3">
            <Image src="/mito.png" alt="Mito" width={64} height={64} />
            <h1 className="text-5xl font-semibold tracking-tight">Mito Tracker</h1>
          </div>
          <p className="text-[rgba(48,6,85,0.89)] mt-4 max-w-3xl mx-auto">
            Upload electron micrographs of mitochondria and get automatic segmentation, labeling,
            and comprehensive metrics analysis powered by AI.
          </p>
        </div>
      </section>

      <div className="max-w-6xl mx-auto px-4 py-8 space-y-6 bg-transparent">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-6 bg-[rgba(23,53,142,0.18)] rounded-2xl shadow">
            <h2 className="mb-6">Upload</h2>
            <input
              id="file"
              type="file"
              accept="image/png,image/tiff,image/tif,image/jpeg"
              onChange={(e) => { setFile(e.target.files?.[0] ?? null); setExampleId(""); }}
              className="hidden"
            />
            <div className="mt-3 text-sm">
              <label className="block mb-1">Example (choose one)</label>
              <select
                value={exampleId}
                onChange={(e) => { setExampleId(e.target.value); if (e.target.value) setFile(null); }}
                className="w-full border rounded px-2 py-1"
              >
                <option value="">None</option>
                <option value="1">Example 1</option>
                <option value="2">Example 2</option>
                <option value="3">Example 3</option>
                <option value="4">Example 4</option>
              </select>
            </div>
            <div className="mt-2 text-center text-[rgba(48,6,85,0.89)] text-sm font-medium">OR</div>
            <button
              type="button"
              onClick={() => document.getElementById("file")?.click()}
              className="mt-2 mx-auto block rounded-md bg-[rgba(152,178,176,0.58)] text-[rgba(48,6,85,0.89)] text-sm px-3 py-1.5 border border-[rgba(48,6,85,0.89)] hover:font-bold"
            >
              Upload your own EM
            </button>
            <label className="text-sm block mt-4 text-[rgba(48,6,85,0.89)]">Confidence threshold: {threshold.toFixed(2)}</label>
            <input
              type="range" min={0.1} max={0.9} step={0.05}
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full h-2 rounded"
              style={{ accentColor: "rgba(48, 6, 85, 0.89)", background: "rgba(152,178,176,0.58)" }}
            />
            <label className="flex items-center gap-2 text-sm mt-3 text-[rgba(48,6,85,0.89)]">
              <input
                type="checkbox"
                checked={tta}
                onChange={(e) => setTta((e.target as HTMLInputElement).checked)}
                style={{ accentColor: "rgba(48, 6, 85, 0.89)" }}
              />
              Use TTA (slower, sometimes more accurate)
            </label>
            <label className="flex items-center gap-2 text-sm mt-2 text-[rgba(48,6,85,0.89)]">
              <input
                type="checkbox"
                checked={useLLM}
                onChange={(e) => setUseLLM((e.target as HTMLInputElement).checked)}
                style={{ accentColor: "rgba(48, 6, 85, 0.89)" }}
              />
              Use LLM labels (sometimes more accurate)
            </label>
            <div className="mt-3 text-sm">
              <label className="block mb-1">Pixel size (nm / pixel, optional)</label>
              <input
                type="number" min={0} step={0.01} placeholder="e.g. 5"
                value={pixelSizeNm}
                onChange={(e) => setPixelSizeNm(e.target.value)}
                className="w-full border rounded px-2 py-1"
              />
            </div>
            <button
              onClick={handleSeg}
              disabled={(!!exampleId ? false : !file) || loading}
              className="mt-4 mx-auto block rounded-lg bg-[rgba(152,178,176,0.58)] text-[rgba(48,6,85,0.89)] text-lg px-5 py-2.5 border border-[rgba(48,6,85,0.89)] hover:font-bold disabled:opacity-50"
            >
              {loading ? "Segmenting…" : "Segment"}
            </button>
          </div>

          <div className="p-6 bg-[rgba(23,53,142,0.18)] rounded-2xl shadow">
            <h2 className="font-medium mb-2">Original</h2>
            {exampleId
              ? <img src={`${API_URL}/examples/${exampleId}/image.png`} className="w-full rounded-md" />
              : (file
                ? <img src={URL.createObjectURL(file)} className="w-full rounded-md" />
                : <div className="text-sm text-[rgba(48,6,85,0.89)]">No image yet.</div>)}
          </div>

          <div className="p-6 bg-[rgba(23,53,142,0.18)] rounded-2xl shadow">
            <h2 className="font-medium mb-2">Overlay</h2>
            {overlayUrl
              ? <img src={overlayUrl} className="w-full rounded-md" />
              : <div className="text-sm text-[rgba(48,6,85,0.89)]">Run segmentation to see overlay.</div>}
            {analysisText && (
              <p className="text-sm text-[rgba(48,6,85,0.89)] mt-3">{analysisText}</p>
            )}
          </div>
        </div>

        <div className="p-6 bg-[rgba(23,53,142,0.18)] rounded-2xl shadow">
          <h2 className="font-medium mb-3">Mask & Metrics</h2>
          {maskUrl && <img src={maskUrl} className="w-64 rounded mb-3" />}
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left border-b">
                  <th className="py-2 pr-4">ID</th>
                  <th className="py-2 pr-4">Area (px²)</th>
                  <th className="py-2 pr-4">Perimeter (px)</th>
                  <th className="py-2 pr-4">Circularity</th>
                  <th className="py-2 pr-4">Area (µm²)</th>
                  <th className="py-2 pr-4">Perimeter (µm)</th>
                  <th className="py-2 pr-4">Length (µm)</th>
                  <th className="py-2 pr-4">Width (µm)</th>
                </tr>
              </thead>
              <tbody>
                {segRan && metrics.length === 0 && (
                  <tr><td className="py-2 text-[rgba(48,6,85,0.89)]">No mitochondria detected</td></tr>
                )}
                {metrics.map((m) => (
                  <tr key={m.id} className="border-b">
                    <td className="py-2 pr-4">{m.id}</td>
                    <td className="py-2 pr-4">{m.area_px ?? "-"}</td>
                    <td className="py-2 pr-4">{m.perimeter_px ?? "-"}</td>
                    <td className="py-2 pr-4">{typeof m.circularity === 'number' ? m.circularity.toFixed(3) : (m.circularity ?? "-")}</td>
                    <td className="py-2 pr-4">{m.area_um2 ?? "-"}</td>
                    <td className="py-2 pr-4">{m.perimeter_um ?? "-"}</td>
                    <td className="py-2 pr-4">{m.length_um ?? "-"}</td>
                    <td className="py-2 pr-4">{m.width_um ?? "-"}</td>
                  </tr>
                ))}
                {metrics.length === 0 && <tr><td className="py-2 text-[rgba(48,6,85,0.89)]">No metrics yet.</td></tr>}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </main>
  );
}
