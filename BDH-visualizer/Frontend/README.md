# BDH Visualizer - Frontend

The frontend for the **BDH Visualizer** is a modern, responsive web application built with **Next.js** and **React Three Fiber**. It provides the interactive user interface for exploring the Baby Dragon Hatchling (BDH) architectures in 3D, and visualizing complex embeddings and neuron activations directly in the browser.

---

## 🚀 Quick Start

### Prerequisites

Ensure you have **Node.js 16+** installed on your system.

### Installation

1. Navigate to the `Frontend` directory from the repository root:
   ```bash
   cd BDH-visualizer/Frontend
   ```
2. Install the necessary dependencies:
   ```bash
   npm install
   ```

### Running the Development Server

To start the Next.js development server:

```bash
npm run dev
```

The application will be available at [http://localhost:3000](http://localhost:3000). 
*(Note: Ensure the Python FastAPI backend is also running for data fetching to work correctly.)*

---

## 🛠️ Technology Stack

- **Framework**: [Next.js](https://nextjs.org/) (App Router)
- **3D Rendering**: [React Three Fiber](https://docs.pmnd.rs/react-three-fiber/getting-started/introduction) & [Three.js](https://threejs.org/)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/)
- **Components**: [Shadcn UI](https://ui.shadcn.com/) & [Radix UI](https://www.radix-ui.com/)
- **Icons**: [Lucide React](https://lucide.dev/)
- **Math Formatting**: [KaTeX](https://katex.org/)

---

## 📂 Project Structure

```text
Frontend/
├── app/               # Next.js 13+ App Router pages and layouts
├── components/        # React components (UI elements, 3D Graph, Embedding Cards)
├── lib/               # Shared utilities, API fetchers, and formatting logic
├── public/            # Static assets (images, SVGs)
├── styles/            # Global CSS and Tailwind directives (if applicable)
├── package.json       # Node.js dependencies and scripts
└── next.config.ts     # Next.js configuration
```

---

## 🎨 Modifying the 3D Graph

The 3D neural network graph utilizes a Barabási-Albert model scaling logic to position nodes. If you need to tweak the physics (gravity, repulsion, spring forces), look inside `components/` for the WebGL/Canvas components controlling the `React Three Fiber` scene.

---

## 🔗 Main Repository Documentation

For information regarding the **FastAPI Backend**, model training scripts, and general dataset utilities, please refer back to the [Main BDH Visualizer README](../README.md).
