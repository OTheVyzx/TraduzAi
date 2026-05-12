# TraduzAI Landing And Logo Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a public landing page for TraduzAI and add a reusable product logo asset while preserving the existing web dashboard at `/dashboard`.

**Architecture:** The `site/` workspace remains the public web surface. The React router serves a public landing page at `/`, while authenticated operational routes stay under the existing protected shell. The logo is a deterministic SVG asset so it stays sharp in the app, browser, docs, and future exports.

**Tech Stack:** React 19, React Router, TypeScript, Vite, Tailwind base pipeline, custom CSS, lucide-react.

---

### Task 1: Add Product Logo Asset

**Files:**
- Create: `site/public/assets/traduzai-logo.svg`
- Modify: `site/src/App.tsx`
- Modify: `site/src/styles.css`

**Steps:**
1. Create a vector SVG logo with a dark square mark, gradient cyan/purple monogram, and wordmark.
2. Use the logo asset in the public landing header.
3. Keep the protected dashboard sidebar logo compatible with the new visual identity.

### Task 2: Add Public Landing Page

**Files:**
- Modify: `site/src/App.tsx`
- Modify: `site/src/styles.css`

**Steps:**
1. Add a `Landing` component for `/`.
2. Keep `/dashboard`, `/novo`, `/job/:id`, `/resultados/:id`, `/settings`, `/legal`, and `/admin` protected or existing.
3. Add CTA links to `/dashboard` and product sections for local-first processing, workflow, privacy, and credits.

### Task 3: Validate

**Commands:**
- `cd site && npm run build`
- Open `http://127.0.0.1:5174/` and `http://127.0.0.1:5174/dashboard` in Chrome.

**Expected:**
- Build succeeds.
- Landing renders at `/`.
- Dashboard remains available and unchanged as an operational panel.
