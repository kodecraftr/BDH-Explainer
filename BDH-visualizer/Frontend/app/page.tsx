"use client";

import React, { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Card, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Sparkles } from "lucide-react";
import { EncodingVisualizer } from "@/components/encoding-visualizer";
import { ProbabilityVisualizer } from "@/components/probability-visualizer";
import { BDHContent } from "@/components/bdh-content";
import { analyzeText } from "@/lib/api";

interface TokenData {
  token_index: number;
  token_text: string;
  token_id: number;
  original_embedding: number[];
  rope_embedding: number[];
}

export default function Page() {
  const router = useRouter();
  const EXAMPLE_PHRASES = [
    "Artificial Intelligence is transforming the",
    "As the spaceship was approaching the",
    "On the deserted planet they discovered a",
    "IEEE VIS conference highlights the",
    "The city of Mumbai is",
  ];

  const [inputText, setInputText] = useState(EXAMPLE_PHRASES[0]);
  const [generatedText, setGeneratedText] = useState("");
  const [predictedToken, setPredictedToken] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [temperature, setTemperature] = useState([0.8]);
  const [sampling, setSampling] = useState([5]);
  const [tokenIds, setTokenIds] = useState<number[]>([]);
  const [tokensData, setTokensData] = useState<TokenData[]>([]);
  const [logits, setLogits] = useState<any[]>([]);
  const [activations, setActivations] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [hasInitialData, setHasInitialData] = useState(false);
  const [showExamples, setShowExamples] = useState(false);
  const examplesRef = useRef<HTMLDivElement>(null);
  const encodingRef = useRef<HTMLDivElement>(null);
  const synapticRef = useRef<HTMLDivElement>(null);
  const nextRef = useRef<HTMLDivElement>(null);

  // Close examples dropdown when clicking outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (examplesRef.current && !examplesRef.current.contains(e.target as Node)) {
        setShowExamples(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // Load state from sessionStorage on mount
  useEffect(() => {
    const savedState = sessionStorage.getItem("bdh-visualizer-state");
    if (savedState) {
      try {
        const parsed = JSON.parse(savedState);
        setInputText(parsed.inputText || "Once there was a");
        setGeneratedText(parsed.generatedText || "");
        setTemperature(parsed.temperature || [0.7]);
        setSampling(parsed.sampling || [40]);
        // setSampling(parsed.sampling || [40]);
        setTokenIds(parsed.tokenIds || []);
        setTokensData(parsed.tokensData || []);
        setLogits(parsed.logits || []);
        setActivations(parsed.activations || null);
        // Check if we have actual generated data
        if (parsed.tokensData && parsed.tokensData.length > 0) {
          setHasInitialData(true);
        }
      } catch (e) {
        console.error("Failed to parse saved state", e);
      }
    }
    setIsLoaded(true);
  }, []);

  // Save state to sessionStorage whenever it changes
  useEffect(() => {
    if (isLoaded) {
      const stateToSave = {
        inputText,
        generatedText,
        temperature,
        sampling,
        tokenIds,
        tokensData,
        logits,
        activations,
      };
      sessionStorage.setItem(
        "bdh-visualizer-state",
        JSON.stringify(stateToSave),
      );
    }
  }, [
    inputText,
    generatedText,
    temperature,
    sampling,
    tokenIds,
    tokensData,
    activations,
    isLoaded,
  ]);

  const handleGenerate = async (overrideText?: string) => {
    // Remove trailing spaces before sending to model
    const trimmedInput = (overrideText ?? inputText).trimEnd();
    setGeneratedText(trimmedInput);
    setPredictedToken(null); // Reset predicted token
    setIsGenerating(true); // Start loading animation

    // Create a minimum delay promise for smooth UX
    const minDelay = new Promise(resolve => setTimeout(resolve, 2000));

    let data = {
      text: trimmedInput,
      temperature: temperature[0],
      top_k: sampling[0],
    };

    console.log("Generating with:", data);

    // Wait for both the API response and minimum delay
    const [response] = await Promise.all([analyzeText(data), minDelay]);
    console.log("Response:", response);

    setIsGenerating(false); // Stop loading animation

    // Generate IDs for tokens
    const newIds = response.data.embeddings.token_ids;
    // Logits
    const logitValues = response.data.logits;
    //activations
    const activations = response.data.activations;
    //embedding matrix
    const embeddings = response.data.embeddings.tokens;

    setLogits(logitValues);
    setTokenIds(newIds);
    setTokensData(embeddings);
    setActivations(activations);

    // Extract the top predicted token and append to input
    if (logitValues && logitValues.length > 0) {
      // Sort by logit value descending and get the top token
      const sortedLogits = [...logitValues].sort((a: any, b: any) => b.logit - a.logit);
      const topToken = sortedLogits[0]?.token || "";
      if (topToken) {
        setPredictedToken(topToken);
        // Append predicted token to input (with proper spacing)
        setInputText(trimmedInput + " " + topToken.trim());
      }
    }
  };

  // Auto-generate on initial load if no saved data
  useEffect(() => {
    if (isLoaded && !hasInitialData && inputText) {
      handleGenerate();
      setHasInitialData(true);
    }
  }, [isLoaded, hasInitialData]);

  const generatedCount = generatedText
    .trim()
    .split(/\s+/)
    .filter((w) => w.length > 0).length;

  const inputWordCount = inputText
    .trim()
    .split(/\s+/)
    .filter((w) => w.length > 0).length;
  const isOverLimit = inputWordCount > 8;

  if (!isLoaded) return null; // Prevent hydration mismatch

  // Animated flow connector between cards — shows data particles travelling along a line
  const FlowConnector = ({ label, delay = 0 }: { label: string; delay?: number }) => (
    <>
      {/* Desktop: horizontal connector */}
      <div className="hidden lg:flex flex-col items-center justify-center flex-shrink-0 w-16 gap-1 py-4 select-none">
        {/* Animated particles flowing left→right */}
        <div className="relative w-full h-8 flex items-center justify-center">
          {/* Static dashed line */}
          <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 border-t border-dashed border-neutral-300 dark:border-neutral-700" />

          {/* Flowing particle track */}
          <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 overflow-hidden h-3 flex items-center">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="absolute w-2 h-2 rounded-full bg-indigo-400/70 dark:bg-indigo-500/60 shadow-[0_0_6px_1px_rgba(129,140,248,0.5)]"
                style={{
                  animation: `flowParticle 2.4s ${delay + i * 0.8}s ease-in-out infinite`,
                }}
              />
            ))}
          </div>

          {/* Arrowhead at end */}
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none" className="absolute right-0 top-1/2 -translate-y-1/2 text-indigo-400 dark:text-indigo-500">
            <polyline points="2,1 8,5 2,9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>

        {/* Label */}
        <p className="text-[7px] font-semibold uppercase tracking-[0.2em] text-neutral-400 dark:text-neutral-600 text-center leading-tight">
          {label}
        </p>
      </div>

      {/* Mobile: vertical connector */}
      <div className="flex lg:hidden flex-row items-center justify-center w-full h-10 gap-2 select-none -my-4">
        <div className="relative w-8 h-full flex items-center justify-center">
          {/* Static dashed line */}
          <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 border-l border-dashed border-neutral-300 dark:border-neutral-700" />
          {/* Flowing particles */}
          <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 overflow-hidden w-3 flex justify-center">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="absolute w-2 h-2 rounded-full bg-indigo-400/70 dark:bg-indigo-500/60 shadow-[0_0_6px_1px_rgba(129,140,248,0.5)]"
                style={{
                  animation: `flowParticleDown 2.4s ${delay + i * 0.8}s ease-in-out infinite`,
                }}
              />
            ))}
          </div>
          {/* Arrowhead at bottom */}
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none" className="absolute bottom-0 left-1/2 -translate-x-1/2 text-indigo-400 dark:text-indigo-500">
            <polyline points="1,2 5,8 9,2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
        <p className="text-[8px] font-semibold uppercase tracking-[0.2em] text-neutral-400 dark:text-neutral-600">
          {label}
        </p>
      </div>
    </>
  );

  // Connector lines removed: visuals simplified for a centered layout

  return (
    <div className="min-h-screen bg-neutral-50 flex flex-col items-center justify-start px-3 pt-4 sm:px-5 sm:pt-5 dark:bg-neutral-950 transition-colors">
      <Card className="w-full max-w-[95vw] 2xl:max-w-[1700px] mx-auto bg-transparent border-0 shadow-none ring-0 overflow-visible">
        <CardContent className="p-0">
          <div className="flex flex-col lg:flex-row items-center gap-6 justify-between">
            {/* Title Section */}
            <div className="flex items-center gap-3 min-w-fit">
              <div className="flex flex-col">
                <img src="/title1.png" alt="BDH Visualizer" className="h-8 lg:h-8 w-auto" />
              </div>
            </div>

            <div className="h-8 w-px bg-neutral-200 dark:bg-neutral-800 hidden lg:block" />

            {/* Controls Section - Horizontal Layout */}
            <div className="flex flex-col lg:flex-row items-center gap-6 flex-1 w-full">
              <div className="flex-1 w-full flex flex-col gap-2 h-10">
                <div className="flex gap-2">
                  {/* Unified input bar: Examples toggle + text input */}
                  <div ref={examplesRef} className={`relative flex flex-1 h-10 items-center rounded-md border bg-white dark:bg-neutral-950 ${isOverLimit ? "border-red-500" : "border-neutral-300 dark:border-neutral-700"} focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-0 overflow-visible`}>
                    {/* Examples toggle section */}
                    <button
                      onClick={() => setShowExamples((v) => !v)}
                      disabled={isGenerating}
                      className={`flex items-center justify-center gap-1.5 px-3 h-full text-sm font-medium text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100 hover:bg-neutral-50 dark:hover:bg-neutral-900 transition-colors rounded-l-md flex-shrink-0 ${showExamples ? "bg-neutral-50 dark:bg-neutral-900 text-neutral-900 dark:text-neutral-100" : ""} disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                      Examples
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="13"
                        height="13"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className={`transition-transform duration-200 ${showExamples ? "rotate-180" : ""}`}
                      >
                        <polyline points="6 9 12 15 18 9" />
                      </svg>
                    </button>
                    {/* Vertical divider */}
                    <div className="w-px h-full bg-neutral-200 dark:bg-neutral-700" />
                    {/* Text input section */}
                    <div className="relative flex-1 h-full">
                      {/* Styled text layer showing predicted token in different color */}
                      <div className="absolute inset-0 flex items-center px-3 pointer-events-none overflow-hidden whitespace-nowrap text-sm font-semibold">
                        {isGenerating ? (
                          <>
                            <span className="text-neutral-900 dark:text-neutral-100">
                              {inputText.trimEnd()}
                            </span>
                            <span className="text-indigo-600 dark:text-indigo-400 font-bold ml-2 text-lg">
                              <span className="inline-flex gap-0.5">
                                <span className="animate-bounce" style={{ animationDelay: '0ms' }}>.</span>
                                <span className="animate-bounce" style={{ animationDelay: '150ms' }}>.</span>
                                <span className="animate-bounce" style={{ animationDelay: '300ms' }}>.</span>
                              </span>
                            </span>
                          </>
                        ) : predictedToken ? (
                          <>
                            <span className="text-neutral-900 dark:text-neutral-100">
                              {inputText.slice(0, inputText.length - predictedToken.trim().length).trimEnd()}
                            </span>
                            <span className="text-indigo-600 dark:text-indigo-400 font-bold animate-in fade-in duration-300">
                              &nbsp;{predictedToken.trim()}
                            </span>
                          </>
                        ) : (
                          <span className="text-transparent">{inputText}</span>
                        )}
                      </div>
                      {/* Actual input (transparent text when predicted token exists) */}
                      <Input
                        type="text"
                        placeholder="Enter your prompt here..."
                        value={inputText}
                        onChange={(e) => {
                          setInputText(e.target.value);
                          setPredictedToken(null);
                        }}
                        disabled={isGenerating}
                        className={`h-full py-0 w-full font-semibold border-0 shadow-none focus-visible:ring-0 rounded-none ${predictedToken || isGenerating ? "text-transparent caret-neutral-900 dark:caret-neutral-100" : ""}`}
                        style={{ background: "transparent" }}
                      />
                    </div>
                    {/* Examples dropdown */}
                    {showExamples && (
                      <div className="absolute left-0 top-full mt-1.5 z-50 w-80 bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-700 rounded-lg shadow-lg overflow-hidden animate-in fade-in slide-in-from-top-2 duration-150">
                        <div className="px-3 py-2 text-xs font-semibold text-neutral-400 dark:text-neutral-500 uppercase tracking-wider border-b border-neutral-100 dark:border-neutral-800">
                          Select a phrase
                        </div>
                        {EXAMPLE_PHRASES.map((phrase, i) => (
                          <button
                            key={i}
                            onClick={() => {
                              setInputText(phrase);
                              setPredictedToken(null);
                              setShowExamples(false);
                              handleGenerate(phrase);
                            }}
                            className={`w-full text-left px-4 py-2.5 text-sm transition-colors hover:bg-neutral-50 dark:hover:bg-neutral-800 ${inputText === phrase
                              ? "text-indigo-600 dark:text-indigo-400 font-semibold bg-indigo-50 dark:bg-indigo-950/30"
                              : "text-neutral-700 dark:text-neutral-300"
                              }`}
                          >
                            {phrase}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                  <Button
                    onClick={() => handleGenerate()}
                    disabled={isOverLimit || inputWordCount === 0 || isGenerating}
                    className="bg-indigo-600 hover:bg-indigo-700 text-white h-10 px-6 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:shadow-md hover:shadow-indigo-500/25 active:scale-[0.97]"
                  >
                    {isGenerating ? (
                      <span className="flex items-center gap-2">
                        <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2.5" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                        </svg>
                        Running
                      </span>
                    ) : "Generate"}
                  </Button>
                </div>
                {isOverLimit && (
                  <div className="flex items-center gap-2 text-red-500 dark:text-red-400 text-xs font-medium px-1 animate-in fade-in slide-in-from-top-1">
                    <span>
                      ⚠️ Warning: Maximum 8 words allowed (current:{" "}
                      {inputWordCount})
                    </span>
                  </div>
                )}
              </div>

              {/* Sliders */}
              <div className="flex items-center gap-6 w-full lg:w-auto">
                {/* Temperature */}
                <div className="flex flex-col gap-1.5 w-full lg:w-32">
                  <div className="flex justify-between items-center text-xs">
                    <Label
                      htmlFor="temperature"
                      className="text-neutral-600 dark:text-neutral-400 font-medium"
                    >
                      Temp
                    </Label>
                    <span className="font-mono text-neutral-500 bg-neutral-100 dark:bg-neutral-800 px-1.5 rounded">
                      {temperature[0]}
                    </span>
                  </div>
                  <Slider
                    id="temperature"
                    min={0.1}
                    max={2.0}
                    step={0.1}
                    value={temperature}
                    onValueChange={setTemperature}
                    className="[&_.range-thumb]:bg-indigo-600 [&_.range-track]:bg-neutral-200 dark:[&_.range-track]:bg-neutral-800"
                  />
                </div>

                {/* Sampling */}
                <div className="flex flex-col gap-1.5 w-full lg:w-32">
                  <div className="flex justify-between items-center text-xs">
                    <Label
                      htmlFor="sampling"
                      className="text-neutral-600 dark:text-neutral-400 font-medium"
                    >
                      Top-K
                    </Label>
                    <span className="font-mono text-neutral-500 bg-neutral-100 dark:bg-neutral-800 px-1.5 rounded">
                      {sampling[0]}
                    </span>
                  </div>
                  <Slider
                    id="sampling"
                    min={5}
                    max={10}
                    step={1}
                    value={sampling}
                    onValueChange={setSampling}
                    className="[&_.range-thumb]:bg-indigo-600 [&_.range-track]:bg-neutral-200 dark:[&_.range-track]:bg-neutral-800"
                  />
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Visualizers Grid */}
      <div
        className={`relative w-full mx-auto max-w-[95vw] 2xl:max-w-[1700px] mt-2 flex flex-col gap-8 transition-all duration-500 px-2 sm:px-4 ${generatedCount > 0 ? "lg:flex-row lg:gap-0 lg:items-stretch lg:justify-center" : ""
          }`}
      >
        {/* Encoding Visualization Section */}
        <div
          ref={encodingRef}
          className={`transition-all duration-500 ${generatedCount > 0 ? "w-full lg:w-[31%] lg:min-w-0 lg:flex lg:justify-end lg:items-center" : "w-full"}`}
        >
          {/* Use updated key to force re-render on generation if needed, or pass prop */}
          <EncodingVisualizer
            inputText={generatedText}
            tokenIds={tokenIds}
            tokensData={tokensData}
          />
        </div>

        {/* Flow connector: Encoding → Synaptic Core */}
        {generatedCount > 0 && <FlowConnector label="embed" delay={0} />}

        {generatedCount > 0 && (
          <div ref={synapticRef} className="w-full lg:w-[31%] lg:min-w-0 lg:mx-0 lg:self-center">
            <Card
              className="h-[500px] xl:h-[640px] border-neutral-200 dark:border-neutral-800 overflow-hidden cursor-pointer group relative transition-all duration-300 hover:ring-2 hover:ring-indigo-500/50 hover:shadow-xl hover:shadow-indigo-500/10"
              onClick={() => router.push(`/neuron?n=${generatedCount}`)}
            >
              <div className="absolute inset-0 bg-neutral-900 z-0">
                <img
                  src="/neuron-ui.gif"
                  alt="Synaptic Core"
                  className="w-full h-full object-cover opacity-60 group-hover:opacity-80 transition-opacity duration-500"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent" />
              </div>

              <div className="relative z-10 h-full flex flex-col p-6">
                {/* Top: step badge + header label — mirrors light cards */}

                
                <div className="mb-auto">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-[10px] font-mono text-white/30 tracking-widest tabular-nums">02</span>
                    <p className="text-[11px] font-bold uppercase tracking-[0.15em] text-white/40">
                      Synaptic Core
                    </p>
                  </div>
                  <p className="text-[10px] text-white/25 pl-6 mb-5">
                    Transformer layers &amp; hidden-state activations
                  </p>

                  <div className="w-8 h-8 rounded-full bg-white/10 backdrop-blur-md flex items-center justify-center border border-white/20 group-hover:scale-110 transition-transform">
                    <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse shadow-[0_0_8px_2px_rgba(74,222,128,0.5)]" />
                  </div>
                </div>

                {/* Bottom: main title + meta */}
                <div>
                  <h3 className="text-2xl font-bold text-white mb-2 leading-tight">
                    Synaptic Core
                  </h3>
                  <p className="text-neutral-300 text-sm mb-4 line-clamp-2">
                    Visualize neural activation patterns and hidden states.
                  </p>

                  <div className="flex items-center text-xs text-indigo-300 font-mono opacity-80 group-hover:opacity-100 transition-opacity">
                    <span className="mr-2">INPUTS: {generatedCount}</span>
                    <span className="w-px h-3 bg-indigo-500/50 mx-2" />
                    <span>ACTIVE</span>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        )}

        {/* Probability Section */}
        {generatedCount > 0 && (
          <>
            {/* Flow connector: Synaptic Core → Probability */}
            <FlowConnector label="logits" delay={0.4} />
            <div ref={nextRef} className="w-full h-[500px] lg:h-auto lg:w-[31%] lg:min-w-0 lg:flex lg:justify-start lg:items-center">
              <ProbabilityVisualizer
                inputText={generatedText}
                temperature={temperature[0]}
                topK={sampling[0]}
                logits={logits}
                error={error}
              />
            </div>
          </>
        )}
      </div>

      {
        generatedCount === 0 && (
          <div className="w-full max-w-3xl mt-12 text-center space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-300">
            <div className="inline-block p-4 rounded-full bg-neutral-100 dark:bg-neutral-900 mb-2">
              <Sparkles className="w-6 h-6 text-neutral-400" />
            </div>
            <h3 className="text-lg font-medium text-neutral-600 dark:text-neutral-400">
              Enter a prompt above to visualize the model's internal state
            </h3>
          </div>
        )
      }

      {/* BDH Content Section */}
      <div className="w-full max-w-7xl mt-48 px-4 sm:px-6 lg:px-8">
        <BDHContent />
      </div>
    </div >
  );
}
