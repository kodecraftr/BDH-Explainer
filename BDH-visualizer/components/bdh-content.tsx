"use client";

import "katex/dist/katex.min.css";
import { InlineMath, BlockMath } from "react-katex";

export function BDHContent() {
  return (
    <div id="description" className="bdh-content">
      <div className="article-section ">
        <h1>What is BDH?</h1>

        <p>
          BDH (Baby Dragon Hatchling) is a novel neural network architecture that provides a link between state-of-the-art tensor-based models and biologically inspired models of the human brain. Introduced in the seminal paper &quot;The Dragon Hatchling: The Missing Link Between the Transformer and Models of the Brain&quot; in 2025, it serves as a practical, attention-based state-space sequence learning architecture.
        </p>
        <p>
          Fundamentally, text-generative BDH models operate on the principle of <strong>local interacting particle systems</strong> performing <strong>next-token prediction</strong>. The core innovation of BDH lies in its use of an edge-reweighting kernel, which leverages Hebbian learning dynamics over a neuron interaction network. This allows the model to process sequences by storing its working memory directly in synaptic plasticity—strengthening connections when concepts fire together—using sparse, positive activation vectors to capture context and dependencies in a highly interpretable way.
        </p>
        <p>
          The BDH-GPU variant is a prominent example of how this graph-based architecture is implemented efficiently using tensor operations on modern hardware. Evaluated at scales ranging from 10 million to 1 billion parameters, BDH-GPU successfully rivals the performance of the GPT-2 Transformer architecture on the same training data. While it represents a fundamental shift toward distributed graph dynamics, it exhibits familiar Transformer-like scaling laws and explicitly demonstrates monosemanticity, making it an ideal starting point for understanding how biological reasoning can be built into scalable, state-of-the-art AI.
        </p>
      </div>

      <div className="article-section">
        <h1>BDH Architecture</h1>

        <p>
          Every text-generative BDH-GPU model consists of <strong>three key components</strong>:
        </p>
        <ol>
          <li>
            <strong className="bold-purple">Embedding</strong>: A linear encoding function that converts input tokens into <InlineMath math="d" />-dimensional numerical vectors that enter the first layer of the model.
          </li>
          <li>
            <strong className="bold-purple">BDH Layer</strong>: The fundamental building block of the model that processes and transforms input data using a local interacting particle system. Each layer includes:
            <ul>
              <li>
                <strong>Linear Attention Mechanism</strong>: The core memory component of the BDH block that operates in the model&apos;s large neuronal dimension, <InlineMath math="n" />. It aggregates key-value correlations over time, maintaining a highly localized state matrix that allows tokens to communicate based on sparse, positive activation vectors.
              </li>
              <li>
                <strong>ReLU-Lowrank Feed-Forward Network</strong>: A feed-forward block that replaces standard MLPs. While the attention layer routes information, this block acts as a signal propagation dynamic. It uses matrices <InlineMath math="D_x" />, <InlineMath math="D_y" />, and <InlineMath math="E" /> alongside a ReLU gate to suppress noise and reinforce signals within modular clusters.
              </li>
            </ul>
          </li>
          <li>
            <strong className="bold-purple">Output Probabilities</strong>: A linear token decoder function that extracts logits from the final layer&apos;s outputs and transforms them into probabilities, enabling the model to predict the next token in a sequence.
          </li>
        </ol>
      </div>

      <div className="article-section" id="embedding">
        <h2>Embedding</h2>
        <p>
          To generate text using a BDH model, a prompt like <code>Once upon a time</code> needs to be converted into a format the model can understand. This is where embedding comes in: it transforms the text into a numerical representation.
        </p>
        <p>
          To convert a prompt into an embedding for the BDH architecture, two initial steps are required before passing the data into the network&apos;s layers: 1) tokenizing the input, and 2) obtaining token embeddings.
        </p>
        <div className="article-subsection">
          <h3>Step 1: Tokenization</h3>
          <p>
            Tokenization is the process of breaking down input text into smaller pieces called tokens. These tokens can be words, subwords, or individual characters. For example, the words <code>&quot;Once&quot;</code> and <code>&quot;upon&quot;</code> might be unique tokens, while <code>&quot;time&quot;</code> could be split into subwords depending on the vocabulary. BDH is flexible with tokenization; experimental versions have even been trained on character-level tokens. The full vocabulary of tokens is decided before the model is trained. Once the input text is split into tokens, we can obtain their vector representation.
          </p>
        </div>
        <div className="article-subsection" id="article-token-embedding">
          <h3>Step 2: Token Embedding</h3>
          <p>
            BDH represents each token in the vocabulary as a multidimensional vector. In the tensor-based BDH-GPU architecture, this vector&apos;s dimension is tied to the model&apos;s low-rank &quot;synaptic dimension&quot; (denoted as <InlineMath math="d" />, typically a 256-dimensional vector). These embedding vectors are stored in an embedding matrix, allowing the model to assign semantic meaning to each token. Tokens with similar usage or meaning are placed close together in this high-dimensional space. This vector representation captures the semantic meaning of the tokens and serves as the input to the first layer of the BDH model.
          </p>
        </div>
      </div>

      <div className="article-section">
        <h2>The BDH Layer in Action</h2>

        <p>
          To process a prompt, BDH-GPU passes data through a series of stacked layers (in this case, 6 layers), each divided into parallel channels called &quot;heads&quot; (here, 4 heads). While traditional Transformers use this structure to calculate massive attention grids, BDH uses it to simulate a highly efficient, local interacting particle system governed by state-space equations. Here is the math and architecture powering the 7 steps of our visualization:
        </p>

        <div className="article-subsection-l2">
          <h4>Step 1: The Dormant State</h4>
          <p>
            Before a token arrives, the &quot;neuronal space&quot; (dimension <InlineMath math="n" />, containing tens of thousands of neurons) is inactive. The model&apos;s memory of all past tokens is stored in a persistent, fixed-size state matrix, denoted mathematically as <InlineMath math="\rho_{t-1,l}" />.
          </p>
        </div>

        <div className="article-subsection-l2">
          <h4>Step 2: Neural Expansion (<InlineMath math="D_x" /> Projection)</h4>
          <p>
            When a new token (represented as the synaptic value vector <InlineMath math="v_{t,l-1}^*" />) enters the layer, it is multiplied by the expansion matrix <InlineMath math="D_x" />. This causes neurons to light up across the network. The blue particles represent positive values, while the red particles represent negative values resulting from the calculation <InlineMath math="D_x v_{t,l-1}^*" />.
          </p>
        </div>

        <div className="article-subsection-l2">
          <h4>Step 3: Sparse Activation (The ReLU Gate)</h4>
          <p>
            BDH applies a Rectified Linear Unit (ReLU) activation gate. This strips away all negative numbers, deactivating the red neurons and leaving only the positive blue ones active. This yields the highly sparse activation vector <InlineMath math="x_{t,l}" />, which serves as both the Query (Q) and Key (K) for the attention mechanism:
          </p>
          <div className="text-center py-4">
            <BlockMath math="x_{t,l} := x_{t,l-1} + (D_x v_{t,l-1}^*)^+" />
          </div>
        </div>

        <div className="article-subsection-l2">
          <h4>Step 4: Clustering & Positional Encoding (RoPE)</h4>
          <p>
            The active neurons organize themselves into functional communities. Because the model relies on a low-rank matrix product paired with a ReLU gate, it naturally learns a modular structure where related semantic concepts physically cluster together.
          </p>
          <p>
            <strong>Crucially, before they communicate, their sequence order is injected.</strong> The model applies Rotary Position Embedding (RoPE) to the active neurons, mathematically rotating their multi-dimensional angles. This dynamic rotation ensures the clusters know the token&apos;s relative position in the sentence before interacting with the memory.
          </p>
        </div>

        <div className="article-subsection-l2">
          <h4>Step 5: Intra-Cluster Broadcasting (Sharing Info)</h4>
          <p>
            Within a cluster, a neuron fires and shares its signal with the rest of the cluster. In BDH&apos;s Linear Attention, this broadcast is calculated by taking the outer product of the active, rotated neurons (Key <InlineMath math="x_{t,l}" />) and the token&apos;s synaptic data (Value <InlineMath math="v_{t,l-1}^*" />). This creates a rank-1 update matrix <InlineMath math="v_{t,l-1}^* x_{t,l}^T" />, representing the neuron broadcasting its current context to the module.
          </p>
        </div>

        <div className="article-subsection-l2">
          <h4>Step 6: Receiving Context (Hebbian Memory Update)</h4>
          <p>
            The same neuron also receives information from others and updates the model&apos;s memory. Following Hebbian learning (&quot;neurons that fire together wire together&quot;), the model mathematically adds the broadcasted signal to the persistent state matrix:
          </p>
          <div className="text-center py-4">
            <BlockMath math="\rho_{t,l} := (\rho_{t-1,l} + v_{t,l-1}^* x_{t,l}^T) U" />
          </div>
          <p>
            Simultaneously, the active rotated neurons query this newly updated memory to retrieve the specific historical context (<InlineMath math="a_{t,l}^*" />) they need from the rest of the network.
          </p>
        </div>

        <div className="article-subsection-l2">
          <h4>Step 7: Inter-Cluster Routing (The 4 Heads & Final Gating)</h4>
          <p>
            Finally, the distinct clusters communicate to form the final output. The retrieved context passes through the 4 independent &quot;heads&quot; (acting as LayerNorm circuits to route information properly), is projected back by the decoder matrix <InlineMath math="D_y" />, and is gated by another ReLU. To ensure the network remains highly efficient, this result is element-wise multiplied (<InlineMath math="\odot" />) by the original active neurons <InlineMath math="x_{t,l}" /> from Step 3. This ensures only the correct modular clusters fire in the final sparse output vector <InlineMath math="y_{t,l}" />, ready for the next layer.
          </p>
        </div>
      </div>

      <div className="article-section" id="article-prob">
        <h2>Output Probabilities</h2>
        <p>
          After the token representations have passed through all <InlineMath math="L" /> stacked BDH blocks, the model must translate these rich, high-dimensional mathematical representations back into human language. This final phase converts the network&apos;s output into a probability distribution over the entire vocabulary to predict the next token.
        </p>
        <p>
          This conversion happens through a specific sequence of operations: <code>Logits -&gt; Scaled Logits (/Temp) -&gt; Top-K Filter -&gt; Softmax -&gt; Probabilities</code>.
        </p>

        <ul>
          <li>
            <strong>Logits (Linear Readout):</strong> The final layer produces a compressed synaptic vector <InlineMath math="v^*" />. BDH uses a linear token decoder function (<InlineMath math="f_d" />) to map this vector to raw, unnormalized scores called <strong>logits</strong>.
          </li>
          <li>
            <strong>Temperature Scaling:</strong> To control the randomness of the generated text, these raw logits are divided by a <strong>Temperature</strong> parameter (e.g., <code>0.8</code>). A temperature less than <code>1</code> increases the gap between high and low scores, making the model more confident and deterministic. A temperature higher than <code>1</code> flattens the scores, making the model more &quot;creative&quot; and random.
          </li>
          <li>
            <strong>Top-K Filter:</strong> To prevent the model from generating irrelevant or bizarre words, it applies a <strong>Top-K</strong> filter. This isolates the top <InlineMath math="K" /> (e.g., <code>5</code>) most likely tokens and discards the rest by setting their probabilities to <InlineMath math="-\infty" />.
          </li>
          <li>
            <strong>Softmax and Probabilities:</strong> Finally, the scaled and filtered logits are passed through a <strong>Softmax</strong> function. This mathematical operation converts the raw scores into probabilities that sum to <code>100%</code>. The model then samples from this final probability distribution to generate the next token.
          </li>
        </ul>
      </div>

      <div className="article-section">
        <h2>Auxiliary Architectural Features</h2>

        <p>
          Several auxiliary architectural features enhance the performance and stability of BDH models. While the edge-reweighting kernel and sparse activations form the core of the architecture, components like Layer Normalization, Dropout, and Residual Connections are crucial for training and overall convergence. Layer Normalization stabilizes high-dimensional signals, Dropout prevents overfitting, and Residual Connections allow information to bypass certain transformations, preventing the vanishing gradient problem in deep networks.
        </p>

        <div className="article-subsection" id="article-ln">
          <h3>Layer Normalization</h3>
          <p>
            Layer Normalization helps stabilize the training process by ensuring the mean and variance of the activations remain consistent across the network. In BDH-GPU, Layer Normalization is applied differently than in standard Transformers and is parameter-free. It is strategically applied to the small synaptic dimension (<InlineMath math="d" />) and is used to separate the different &quot;heads&quot; of the model. Instead of projecting through multiple complex matrices, BDH uses a simple LayerNorm to normalize the outcomes of the linear attention mechanism separately for each head. This acts as an inhibitory circuit before the data is passed to the ReLU gate, reducing sensitivity to initial weights and allowing the model to learn more effectively.
          </p>
        </div>

        <div className="article-subsection" id="article-dropout">
          <h3>Dropout</h3>
          <p>
            Dropout is a regularization technique used to prevent neural networks from overfitting by randomly setting a fraction of model weights to zero during training. In BDH, which already relies on extreme sparsity (typically around 5% active neurons), a minimal Dropout (e.g., 0.01 to 0.1) is still applied to the final output vector (<InlineMath math="y" />) of each layer. This encourages the model to learn more robust features without relying too heavily on any single active neuron, ensuring it generalizes well to new, unseen data. During text generation (inference), dropout is deactivated to maximize performance.
          </p>
        </div>

        <div className="article-subsection" id="article-residual">
          <h3>Residual Connections</h3>
          <p>
            Residual connections (or skip connections) act as shortcuts that bypass one or more layers, adding the original input of a layer directly to its output. This is essential for training deep neural networks, as it allows gradients to flow backward through the network more easily, mitigating the vanishing gradient problem. In BDH-GPU, residual connections are used to carry forward the token&apos;s core representations at two crucial stages: both when expanding the token&apos;s state into the massive neuronal dimension (updating <InlineMath math="x" />) and when compressing it back into the synaptic dimension (updating the <InlineMath math="v^*" /> vector). This ensures that vital contextual information is not lost as it passes through the successive, highly-sparse ReLU gates, maintaining stability across all layers.
          </p>
        </div>
      </div>

      <div className="article-section">
        <h1>Interactive Features</h1>
        <p>
          The BDH Visualizer is built to be interactive and allows you to explore the inner workings of the Baby Dragon Hatchling architecture. Here are some of the interactive features you can use:
        </p>

        <ul>
          <li>
            <strong>Input your own text sequence</strong> to see how the model processes it and predicts the next token. Watch as your prompt is encoded, expanded into the neuronal dimension, and processed through the local graph dynamics.
          </li>
          <li>
            <strong>Interact with real neuron activations</strong> to see how the network evaluates concepts. Because BDH uses a ReLU gate to enforce sparsity, you can hover over the active (blue) particles within a cluster to view their real activation values in the <InlineMath math="n" />-dimensional space.
          </li>
          <li>
            <strong>Use the temperature slider</strong> to control the randomness of the model&apos;s predictions. Explore how you can make the model&apos;s text generation more deterministic or creative by adjusting the temperature.
          </li>
          <li>
            <strong>Select the top-k sampling method</strong> to adjust the model&apos;s filtering behavior during inference. Experiment with different <InlineMath math="K" /> values to see how the probability distribution changes, discarding unlikely tokens to influence the model&apos;s final predictions.
          </li>
        </ul>
      </div>

      <style jsx>{`
        .bdh-content {
          padding-bottom: 3rem;
          margin-left: auto;
          margin-right: auto;
          max-width: 80ch;
          width: 100%;
        }

        .bold-purple {
          color: #6A2DD5;
          font-weight: bold;
        }

        code {
          color: rgb(17, 24, 39);
          font-family: ui-monospace, monospace;
          font-weight: 600;
          word-break: break-word;
        }

        .article-section {
          padding-bottom: 2rem;
        }

        .bdh-content h1 {
          color: #6A2DD5;
          font-size: clamp(1.5rem, 3vw, 2.2rem);
          font-weight: 500;
          padding-top: 1rem;
        }

        .bdh-content h2 {
          color: #6A2DD5;
          font-size: clamp(1.3rem, 2.5vw, 2rem);
          font-weight: 500;
          padding-top: 1rem;
        }

        .bdh-content h3 {
          color: rgb(17, 24, 39);
          font-size: clamp(1.1rem, 2vw, 1.6rem);
          font-weight: 600;
          padding-top: 1rem;
        }

        .bdh-content h4 {
          color: rgb(17, 24, 39);
          font-size: clamp(1rem, 1.8vw, 1.4rem);
          font-weight: 600;
          padding-top: 1rem;
        }

        .bdh-content p {
          margin: 1rem 0;
          color: rgb(17, 24, 39);
          line-height: 1.6;
          font-size: clamp(0.875rem, 1.1vw, 1rem);
        }

        .bdh-content ol {
          margin-left: 1.5rem;
          list-style-type: decimal;
        }

        .bdh-content li {
          margin: 0.6rem 0;
          color: rgb(0, 0, 0);
          line-height: 1.6;
          font-size: clamp(0.875rem, 1.1vw, 1rem);
        }

        .bdh-content ul {
          list-style-type: disc;
          margin-left: 1.5rem;
          margin-bottom: 1rem;
        }

        .article-subsection {
          margin-left: 0.5rem;
        }

        .article-subsection-l2 {
          margin-left: 0.5rem;
        }

        @media (min-width: 640px) {
          .bdh-content ol {
            margin-left: 3rem;
          }

          .bdh-content ul {
            margin-left: 2.5rem;
          }

          .article-subsection {
            margin-left: 1rem;
          }

          .article-subsection-l2 {
            margin-left: 1rem;
          }
        }

        @media (prefers-color-scheme: dark) {
          .bold-purple {
            color: #6A2DD5;
          }

          code {
            color: rgb(17, 24, 39);
            font-weight: 600;
          }

          .bdh-content h1,
          .bdh-content h2 {
            color: #6A2DD5;
          }

          .bdh-content h3,
          .bdh-content h4 {
            color: rgb(17, 24, 39);
          }

          .bdh-content p,
          .bdh-content li {
            color: rgb(17, 24, 39);
          }
        }
      `}</style>
    </div>
  );
}
