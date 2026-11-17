import type { NextConfig } from "next";

// "standalone" emits a minimal server bundle (.next/standalone) for the Docker image.
const config: NextConfig = { reactStrictMode: true, output: "standalone" };
export default config;
