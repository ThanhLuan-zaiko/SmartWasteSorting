import type { NextConfig } from "next";

const localAgentBase =
  process.env.LOCALAGENT_API_BASE ?? "http://127.0.0.1:8080";
const distDir = process.env.NEXT_DIST_DIR ?? ".next";

const nextConfig: NextConfig = {
  distDir,
  async rewrites() {
    return [
      {
        source: "/api/localagent/:path*",
        destination: `${localAgentBase}/:path*`,
      },
    ];
  },
};

export default nextConfig;
