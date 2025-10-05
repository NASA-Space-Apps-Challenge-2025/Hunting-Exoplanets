import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
};
const isProd = process.env.NODE_ENV === 'production';

module.exports = {
  output: 'export',                 // static export
  images: { unoptimized: true },    // no Image Optimization on GH Pages
  basePath: '/Hunting-Exoplanets',  // repo name
  assetPrefix: '/Hunting-Exoplanets/',
  trailingSlash: true               // helps with GH Pages routing
};
export default nextConfig;
