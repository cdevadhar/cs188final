import { reactRouter } from "@react-router/dev/vite";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";
import tsconfigPaths from "vite-tsconfig-paths";

export default defineConfig({
  base: "./",
  plugins: [tailwindcss(), reactRouter(), tsconfigPaths()],
  build: {
    outDir: "docs", // If you want it to build directly into your docs folder
  }
});
