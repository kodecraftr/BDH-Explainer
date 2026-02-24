import React, { useState, useRef, useEffect, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  CameraControls,
  Instances,
  Instance,
  Environment,
  Line,
  Text,
  Billboard, // Added Billboard import
} from "@react-three/drei";
import { EffectComposer, Bloom } from "@react-three/postprocessing";
import * as THREE from "three";

// --- Constants ---
const NEURON_COUNT = 1000;
// const N_INPUTS = 5;
const TOTAL_NEIGHBORS = 10;
const SPHERE_RADIUS = 35;

// --- Helper: Seeded Random Generator ---
const mulberry32 = (a) => {
  return () => {
    var t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

// --- Helper: Find neighbors & Values ---
// --- Helper: Find neighbors & Values ---
const getClusterData = (allPositions, n, layer, head, activations) => {
  const seed = layer * 1000 + head * 100;
  const rng = mulberry32(seed);

  // Use real activations if available
  let activationData = [];
  if (activations && activations[`layer_${layer}`]) {
    const layerData = activations[`layer_${layer}`];
    // Head indices in JSON seem to be 0-based arrays inside the layer object
    // JSON structure: { "layer_1": [ [ {id, v}, ... ], [ ... ] ... ] }
    if (layerData[head - 1]) {
      activationData = layerData[head - 1];
    }
  }

  const indices = allPositions.map((_, i) => i);
  // Shuffle indices for random clustering if no activations or as a base
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const selectedCenters = indices.slice(0, n);
  const clusters = [];

  selectedCenters.forEach((centerIdx, i) => {
    const centerNode = allPositions[centerIdx];

    const withDist = allPositions.map((pos, idx) => {
      const dx = pos.x - centerNode.x;
      const dy = pos.y - centerNode.y;
      const dz = pos.z - centerNode.z;
      return {
        id: idx,
        dist: Math.sqrt(dx * dx + dy * dy + dz * dz),
      };
    });

    const neighbors = withDist
      .sort((a, b) => a.dist - b.dist)
      .slice(1, TOTAL_NEIGHBORS + 1)
      .map((x) => x.id);

    let centerValue = 0;
    let neighborValues = neighbors.map(() => 0);

    if (activationData.length > 0) {
      // Logic to map activation data to clusters
      // We can take the top activated neurons from the data, or just map sequentially/randomly from the provided data
      // defined strategies:
      // 1. Map `centerIdx` to a specific `id` in activationData if possible (e.g. id % count)
      // 2. Or just take values from the start of the array since we have limited clusters

      // Simple strategy: Use the index of the cluster to pick from activation data
      // We have 'n' clusters.
      // Let's spread the activation data across them.
      // The JSON has many values. Let's try to find high values.

      // Better: Sort activation data by value 'v' descending to show "active" features?
      // Or just map deterministically.

      // Let's use the provided `id` in JSON to map to neuron index if possible,
      // but neuron count is 1000 and ids might be larger.
      // Let's use `id % NEURON_COUNT`?

      // For the center:
      // We need a value.
      // Let's pick a value from activationData based on the cluster index.
      const dataIndex = i % activationData.length;
      centerValue = activationData[dataIndex].v;

      // For neighbors:
      // We need values.
      neighborValues = neighbors.map((_, nI) => {
        const nDataIndex = (dataIndex + nI + 1) % activationData.length;
        return activationData[nDataIndex].v;
      });
    } else {
      // Fallback random
      centerValue = parseFloat(rng().toFixed(2));
      neighborValues = neighbors.map(() => parseFloat(rng().toFixed(2)));
    }

    clusters.push({
      center: centerIdx,
      neighbors,
      centerValue,
      neighborValues,
    });
  });

  return { clusters };
};

// --- Component: Keyboard Zoom Handler ---
const KeyboardZoom = ({ controlsRef }) => {
  const [zoomDir, setZoomDir] = useState(0);

  useEffect(() => {
    const handleDown = (e) => {
      if (e.altKey) {
        if (e.code === "KeyQ") setZoomDir(1);
        if (e.code === "KeyW") setZoomDir(-1);
      }
    };

    const handleUp = (e) => {
      if (e.code === "KeyQ" || e.code === "KeyW") setZoomDir(0);
      if (e.key === "Alt") setZoomDir(0);
    };

    window.addEventListener("keydown", handleDown);
    window.addEventListener("keyup", handleUp);
    return () => {
      window.removeEventListener("keydown", handleDown);
      window.removeEventListener("keyup", handleUp);
    };
  }, []);

  useFrame((_, delta) => {
    if (zoomDir !== 0 && controlsRef.current) {
      const speed = 40 * delta;
      controlsRef.current.dolly(zoomDir * speed, true);
    }
  });

  return null;
};

// --- Component: Value Labels (Updated for Billboard) ---
const NeuronValues = ({ activeData, positions, showAll, hoveredId }) => {
  if (!activeData) return null;

  // Collect all potential labels
  const labels = [];
  activeData.clusters.forEach((cluster) => {
    // Center
    labels.push({
      id: cluster.center,
      pos: positions[cluster.center],
      val: cluster.centerValue,
      isCenter: true,
    });

    // Neighbors
    cluster.neighbors.forEach((nIdx, i) => {
      labels.push({
        id: nIdx,
        pos: positions[nIdx],
        val: cluster.neighborValues[i],
        isCenter: false,
      });
    });
  });

  return (
    <group>
      {labels.map((item, i) => {
        // Show text if 'Spacebar' toggle is ON OR if this specific neuron is hovered
        const isVisible = showAll || item.id === hoveredId;

        if (!isVisible) return null;

        return (
          // Wrap Text in Billboard to always face camera
          <Billboard
            key={i}
            position={[item.pos.x, item.pos.y + 0.5, item.pos.z]}
          >
            <Text
              fontSize={1.2}
              color={item.isCenter ? "#ffaa00" : "white"}
              anchorX="center"
              anchorY="bottom"
              outlineWidth={0.05}
              outlineColor="#000000"
              renderOrder={100} // Ensure text renders on top
            >
              {item.val.toFixed(2)}
            </Text>
          </Billboard>
        );
      })}
    </group>
  );
};

// --- Component: Cluster Bubbles (Step 4) ---
const ClusterBubbles = ({ clusters, positions }) => {
  const colors = ["#ff0055", "#00ffaa", "#5500ff", "#ffff00", "#ffaa00"];

  return (
    <group>
      {clusters.map((cluster, i) => {
        const center = positions[cluster.center];
        return (
          <mesh key={i} position={[center.x, center.y, center.z]}>
            <sphereGeometry args={[4, 32, 32]} />
            <meshStandardMaterial
              color={colors[i % colors.length]}
              transparent
              opacity={0.15}
              roughness={0.1}
              metalness={0.1}
              depthWrite={false}
            />
          </mesh>
        );
      })}
    </group>
  );
};

// --- Component: Links & Pulses ---
const PulseLines = ({ step, clusters, positions }) => {
  const PulsePackets = () => {
    if (step !== 5 && step !== 6) return null;

    return (
      <group>
        {clusters.map((cluster, cI) =>
          cluster.neighbors.map((nIdx, nI) => (
            <PulseCylinder
              key={`${cI}-${nI}`}
              step={step}
              start={positions[cluster.center]}
              end={positions[nIdx]}
            />
          )),
        )}
      </group>
    );
  };

  return (
    <group>
      {step >= 5 &&
        clusters.map((cluster, cI) => (
          <group key={cI}>
            {cluster.neighbors.map((nIdx, i) => {
              const start = positions[cluster.center];
              const end = positions[nIdx];
              const width = step === 7 ? 0.5 : 1;
              const opacity = step === 7 ? 0.1 : 0.4;

              return (
                <Line
                  key={`line-${cI}-${i}`}
                  points={[
                    [start.x, start.y, start.z],
                    [end.x, end.y, end.z],
                  ]}
                  color="white"
                  lineWidth={width}
                  transparent
                  opacity={opacity}
                />
              );
            })}
          </group>
        ))}

      {step === 7 &&
        clusters.map((cluster, i) => {
          if (i === clusters.length - 1) return null;
          const start = positions[cluster.center];
          const next = positions[clusters[i + 1].center];
          return (
            <Line
              key={`global-${i}`}
              points={[
                [start.x, start.y, start.z],
                [next.x, next.y, next.z],
              ]}
              color="#ffaa00"
              lineWidth={3}
              transparent
              opacity={0.6}
            />
          );
        })}

      <PulsePackets />
    </group>
  );
};

// --- Sub-component: Cylindrical Pulse ---
const PulseCylinder = ({ step, start, end }) => {
  const meshRef = useRef();

  const startVec = useMemo(
    () => new THREE.Vector3(start.x, start.y, start.z),
    [start],
  );
  const endVec = useMemo(() => new THREE.Vector3(end.x, end.y, end.z), [end]);

  const quaternion = useMemo(() => {
    const direction = new THREE.Vector3()
      .subVectors(endVec, startVec)
      .normalize();
    const up = new THREE.Vector3(0, 1, 0);
    return new THREE.Quaternion().setFromUnitVectors(up, direction);
  }, [startVec, endVec]);

  useFrame((state) => {
    if (!meshRef.current) return;
    const time = state.clock.elapsedTime;
    const t = (time * 1.5) % 1;

    const stopAt = 0.9; //stops the pulse before, seems visually more correct

    if (step === 5) {
      meshRef.current.position.lerpVectors(startVec, endVec, t * stopAt);
    } else if (step === 6) {
      meshRef.current.position.lerpVectors(endVec, startVec, t * stopAt);
    }
  });

  const color = step === 5 ? "#00ffff" : "#ffaa00";

  return (
    <mesh ref={meshRef} quaternion={quaternion}>
      <cylinderGeometry args={[0.08, 0.08, 1.5, 6]} />
      <meshBasicMaterial color={color} toneMapped={false} />
    </mesh>
  );
};

// --- Component: Neurons ---
const NeuralNetwork = ({
  step,
  activeData,
  positions,
  controlsRef,
  showValues,
}) => {
  const meshRef = useRef();
  const stepStartRef = useRef(0);
  // New State: Track hovered neuron index
  const [hoveredNeuron, setHoveredNeuron] = useState(null);

  useEffect(() => {
    stepStartRef.current = Date.now();
  }, [step]);

  const assignments = useMemo(() => {
    return positions.map(() => (Math.random() > 0.5 ? "red" : "blue"));
  }, [positions]);

  const cDark = new THREE.Color("#050a05");
  const cBlue = new THREE.Color("#0044aa");
  const cRed = new THREE.Color("#880000");
  const cWhite = new THREE.Color("#ffffff");
  const cGold = new THREE.Color("#ffaa00");

  useFrame((state) => {
    if (!meshRef.current || !activeData) return;

    const timeSinceStep = (Date.now() - stepStartRef.current) / 1000;
    const { clusters } = activeData;

    const activeSet = new Set();
    const centerSet = new Set();
    clusters.forEach((c) => {
      activeSet.add(c.center);
      centerSet.add(c.center);
      c.neighbors.forEach((n) => activeSet.add(n));
    });

    for (let i = 0; i < positions.length; i++) {
      let finalColor = cDark;
      const zPos = positions[i].z;
      const isCluster = activeSet.has(i);
      const isCenter = centerSet.has(i);

      if (step === 1) {
        finalColor = cDark;
      } else if (step === 2) {
        const waveZ = timeSinceStep * 25 - 30;
        if (zPos < waveZ) {
          if (isCluster) finalColor = cBlue;
          else finalColor = assignments[i] === "red" ? cRed : cBlue;
        } else {
          finalColor = cDark;
        }
      } else if (step === 3) {
        const waveZ = timeSinceStep * 25 - 30;
        if (zPos < waveZ) {
          if (isCluster) finalColor = cWhite;
          else finalColor = cDark;
        } else {
          if (isCluster) finalColor = cBlue;
          else finalColor = assignments[i] === "red" ? cRed : cBlue;
        }
      } else if (step === 4) {
        if (isCluster) finalColor = cWhite;
        else finalColor = cDark;
      } else if (step === 5) {
        if (isCluster) finalColor = cWhite;
        else finalColor = cDark;
      } else if (step === 6) {
        if (isCenter) finalColor = cGold;
        else if (isCluster) finalColor = cWhite;
        else finalColor = cDark;
      } else if (step === 7) {
        if (isCenter) finalColor = cGold;
        else if (isCluster) finalColor = cWhite;
        else finalColor = cDark;
      }

      meshRef.current.setColorAt(i, finalColor);
    }
    meshRef.current.instanceColor.needsUpdate = true;
  });

  // --- CAMERA LOGIC ---
  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls || !activeData || activeData.clusters.length === 0) return;

    const firstCenterIdx = activeData.clusters[0].center;
    const firstCenter = positions[firstCenterIdx];

    if (step === 1) {
      controls.setLookAt(0, 0, 60, 0, 0, 0, true);
    } else if (step === 2 || step === 3) {
      controls.setLookAt(50, 10, 10, 0, 0, 0, true);
    } else if (step === 4) {
      controls.setLookAt(0, 0, 50, 0, 0, 0, true);
    } else if (step === 5 || step === 6) {
      controls.setLookAt(
        firstCenter.x + 5,
        firstCenter.y + 5,
        firstCenter.z + 10,
        firstCenter.x,
        firstCenter.y,
        firstCenter.z,
        true,
      );
    } else if (step === 7) {
      controls.setLookAt(0, 0, 70, 0, 0, 0, true);
    }
  }, [step, activeData, positions, controlsRef]);

  return (
    <>
      <Instances range={NEURON_COUNT} ref={meshRef}>
        <sphereGeometry args={[0.2, 16, 16]} />
        <meshPhysicalMaterial
          roughness={0.2}
          metalness={0.5}
          emissiveIntensity={1.5}
          toneMapped={false}
          clearcoat={1}
        />
        {positions.map((p, i) => (
          <Instance
            key={i}
            position={[p.x, p.y, p.z]}
            // Add Pointer Events for Hover
            onPointerOver={(e) => {
              e.stopPropagation();
              setHoveredNeuron(i);
            }}
            onPointerOut={(e) => {
              setHoveredNeuron(null);
            }}
          />
        ))}
      </Instances>

      {/* Cluster Bubbles (Step 4) */}
      {step === 4 && activeData && (
        <ClusterBubbles clusters={activeData.clusters} positions={positions} />
      )}

      {/* Value Text Labels - VISIBLE ONLY AFTER STEP 3 */}
      {/* Passing 'hoveredNeuron' and 'showValues' to handle logic inside */}
      {step > 3 && (
        <NeuronValues
          activeData={activeData}
          positions={positions}
          showAll={showValues}
          hoveredId={hoveredNeuron}
        />
      )}

      {/* Links & Pulses */}
      {activeData && (
        <PulseLines
          step={step}
          clusters={activeData.clusters}
          positions={positions}
        />
      )}
    </>
  );
};

// --- Main App ---
export default function Neuron({ nInputs = 5, activations = null }) {
  const [step, setStep] = useState(1);
  const [activeData, setActiveData] = useState(null);
  const cameraControlsRef = useRef();

  // New State: Layers and Heads
  const [currentLayer, setCurrentLayer] = useState(1);
  const [currentHead, setCurrentHead] = useState(1);

  // New State: Show Values
  const [showValues, setShowValues] = useState(false);

  // Spacebar Toggle Listener
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.code === "Space") {
        e.preventDefault(); // Prevent scrolling
        setShowValues((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // 1. Generate Static Data
  const positions = useMemo(() => {
    const pos = [];
    for (let i = 0; i < NEURON_COUNT; i++) {
      const phi = Math.random() * Math.PI * 2;
      const costheta = Math.random() * 2 - 1;
      const u = Math.random();

      const theta = Math.acos(costheta);
      const r = SPHERE_RADIUS * Math.cbrt(u);

      const x = r * Math.sin(theta) * Math.cos(phi);
      const y = r * Math.sin(theta) * Math.sin(phi);
      const z = r * Math.cos(theta);

      pos.push({ x, y, z });
    }
    return pos;
  }, []);

  // 2. Calculate Clusters based on Layer/Head Combination
  useEffect(() => {
    // Re-calculate whenever layer or head changes
    const data = getClusterData(
      positions,
      nInputs,
      currentLayer,
      currentHead,
      activations,
    );
    setActiveData(data);
  }, [positions, nInputs, currentLayer, currentHead, activations]);

  const nextStep = () => setStep((s) => (s >= 7 ? 1 : s + 1));
  const prevStep = () => setStep((s) => (s <= 1 ? 7 : s - 1));

  const stepsInfo = {
    1: "1. Overview: Initial State",
    2: "2. Forward Pass: Wave Activation (Color)",
    3: "3. Activation: ReLU Wave (Cleaning)",
    4: "4. Processing: Active Regions (Bubbles)",
    5: "5. Context: Blue Pulse (Outward)",
    6: "6. Attention: Golden Pulse (Inward)",
    7: "7. Output: Global Connections",
  };

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: "black",
        position: "relative",
      }}
    >
      <Canvas
        camera={{ fov: 60 }}
        gl={{
          powerPreference: "high-performance",
          antialias: false,
          depth: true,
        }}
        dpr={[1, 1.5]}
      >
        <color attach="background" args={["#010101"]} />
        <Environment preset="city" />
        <ambientLight intensity={0.2} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <pointLight position={[-10, -10, -10]} color="#00ff41" intensity={2} />

        <CameraControls ref={cameraControlsRef} smoothTime={0.8} />

        <KeyboardZoom controlsRef={cameraControlsRef} />

        <NeuralNetwork
          step={step}
          activeData={activeData}
          positions={positions}
          controlsRef={cameraControlsRef}
          showValues={showValues}
        />

        <EffectComposer disableNormalPass>
          <Bloom luminanceThreshold={0.5} intensity={1.5} radius={0.4} />
        </EffectComposer>
      </Canvas>

      {/* --- Top Left Control Bar: Layer/Head --- */}
      <div
        style={{
          position: "absolute",
          top: "20px",
          left: "20px",
          display: "flex",
          flexDirection: "column",
          gap: "10px",
          fontFamily: "monospace",
          zIndex: 10,
        }}
      >
        {/* Layer Controls */}
        <div style={controlRowStyle}>
          <span style={{ color: "#888", width: "50px" }}>Layer:</span>
          {[1, 2, 3, 4, 5, 6].map((l) => (
            <button
              key={l}
              onClick={() => setCurrentLayer(l)}
              style={l === currentLayer ? activeBtnStyle : inactiveBtnStyle}
            >
              {l}
            </button>
          ))}
        </div>

        {/* Head Controls */}
        <div style={controlRowStyle}>
          <span style={{ color: "#888", width: "50px" }}>Head:</span>
          {[1, 2, 3, 4].map((h) => (
            <button
              key={h}
              onClick={() => setCurrentHead(h)}
              style={h === currentHead ? activeBtnStyle : inactiveBtnStyle}
            >
              {h}
            </button>
          ))}
        </div>

        {/* Value Toggle Hint */}
        <div style={{ marginTop: "5px", fontSize: "11px", color: "#666" }}>
          [Spacebar] Toggle Values: {showValues ? "ON" : "OFF"}
        </div>
      </div>

      {/* --- Bottom Controls --- */}
      <div
        style={{
          position: "absolute",
          bottom: "30px",
          left: "50%",
          transform: "translateX(-50%)",
          background: "rgba(20, 20, 20, 0.9)",
          border: "1px solid #333",
          borderRadius: "12px",
          padding: "15px 30px",
          display: "flex",
          alignItems: "center",
          gap: "20px",
          color: "white",
          fontFamily: "monospace",
          boxShadow: "0 0 20px rgba(0, 100, 255, 0.2)",
        }}
      >
        <button onClick={prevStep} style={navBtnStyle}>
          {" "}
          &lt;{" "}
        </button>
        <div style={{ textAlign: "center", minWidth: "350px" }}>
          <div style={{ fontSize: "12px", color: "#888", marginBottom: "4px" }}>
            STEP {step} / 7
          </div>
          <div
            style={{
              color: step > 5 ? "#ffaa00" : "#00ffff",
              fontWeight: "bold",
            }}
          >
            {stepsInfo[step]}
          </div>
        </div>
        <button onClick={nextStep} style={navBtnStyle}>
          {" "}
          &gt;{" "}
        </button>
      </div>

      {/* Zoom Instructions */}
      <div
        style={{
          position: "absolute",
          bottom: "20px",
          right: "20px",
          color: "rgba(255,255,255,0.4)",
          fontSize: "12px",
          fontFamily: "monospace",
          pointerEvents: "none",
          border: "1px solid #333",
          borderRadius: "12px",
          padding: "15px 30px",
        }}
      >
        Alt + Q : Zoom In <br />
        Alt + W : Zoom Out
      </div>
    </div>
  );
}

// --- Styles ---
const controlRowStyle = {
  display: "flex",
  alignItems: "center",
  gap: "5px",
  background: "rgba(0,0,0,0.6)",
  padding: "8px 12px",
  borderRadius: "8px",
  border: "1px solid #333",
};

const baseBtnStyle = {
  border: "none",
  borderRadius: "4px",
  padding: "5px 10px",
  cursor: "pointer",
  fontFamily: "monospace",
  fontSize: "12px",
  transition: "all 0.2s",
};

const activeBtnStyle = {
  ...baseBtnStyle,
  background: "#00ffff",
  color: "black",
  fontWeight: "bold",
};

const inactiveBtnStyle = {
  ...baseBtnStyle,
  background: "#333",
  color: "#aaa",
};

const navBtnStyle = {
  background: "#333",
  border: "none",
  color: "white",
  padding: "10px 15px",
  borderRadius: "6px",
  cursor: "pointer",
  fontSize: "16px",
  fontWeight: "bold",
  transition: "0.2s",
};
