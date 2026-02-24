import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: false,
  async rewrites() {
    return [
      {
        source: "/api/backend/:path*",
        destination: "http://34.131.231.223:8000/:path*", // Proxy to your GCP VM
      },
    ];
  },
};

export default nextConfig;