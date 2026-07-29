import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Deployed as a GitHub Pages project site at chrisbergin.github.io/swing-frames/,
// so assets must resolve under that subpath rather than the domain root.
// https://vite.dev/config/
export default defineConfig({
  base: '/swing-frames/',
  plugins: [react()],
})
