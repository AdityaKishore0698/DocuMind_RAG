import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The route indicator overlaps the sidebar's account actions; compile/runtime errors are still shown.
  devIndicators: false,
};

export default nextConfig;
