export interface AnalyzeRequest {
  text: string;
  temperature: number;
  top_k: number;
}

export interface AnalyzeResponse {
  tokens?: string[];
  probs?: number[];
  top_k_tokens?: string[][];
  top_k_probs?: number[][];
  [key: string]: any;
}

// Update the base URL to point to the Next.js rewrite path
const API_BASE_URL = "/api/backend";

export async function analyzeText(
  data: AnalyzeRequest,
): Promise<AnalyzeResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `API request failed with status ${response.status}: ${errorText}`,
      );
    }

    return await response.json();
  } catch (error) {
    console.error("Error in analyzeText:", error);
    throw error;
  }
}