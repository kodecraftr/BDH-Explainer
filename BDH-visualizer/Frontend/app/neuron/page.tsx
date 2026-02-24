"use client";

import { Suspense, useState, useEffect } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import Neuron from "@/components/Neuron";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ArrowRight } from "lucide-react";

function NeuronPageContent() {
  const searchParams = useSearchParams();
  const n = searchParams.get("n");
  const nInputs = n ? Math.max(1, parseInt(n)) : 5; // Ensure at least 1 input
  const router = useRouter();
  const [activations, setActivations] = useState(null);

  useEffect(() => {
    const savedState = sessionStorage.getItem("bdh-visualizer-state");
    if (savedState) {
      try {
        const parsed = JSON.parse(savedState);
        if (parsed.activations) {
          setActivations(parsed.activations);
        }
      } catch (e) {
        console.error("Failed to load activations", e);
      }
    }
  }, []);

  return (
    <div className="relative w-full h-screen bg-black">
      <Neuron nInputs={nInputs} activations={activations} />

      {/* Navigation Controls */}
      <div className="absolute top-6 right-6 z-50">
        <Button
          onClick={() => router.push("/")}
          className="bg-white/10 hover:bg-white/20 text-white hover:text-white border-white/20 backdrop-blur-md gap-2"
          variant="outline"
        >
          Next <ArrowRight className="w-4 h-4" />
        </Button>
      </div>
    </div>
  );
}

export default function NeuronPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center w-full h-screen bg-black text-white">
          Initializing Synaptic Core...
        </div>
      }
    >
      <NeuronPageContent />
    </Suspense>
  );
}
