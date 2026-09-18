# Job Scout

React + TypeScript job finder using either the OpenAI Responses API or the Gemini API, both with
built-in web search.

## Run locally

```bash
pnpm install
pnpm dev
```

Open http://localhost:5173. There's no bundled API key: the app will prompt you to pick a
provider (ChatGPT/OpenAI or Gemini/Google) and enter your own API key for it on first run (see
"What it does" below).

## What it does
- On first visit, pick an AI provider (ChatGPT or Gemini) and enter your own API key for it. It's
  stored only in your browser's `localStorage` and sent per-request - never saved on any server.
  You can change the provider or key later from the menu → Settings.
- Then upload your CV (PDF). It's sent to the selected provider to extract a candidate profile,
  which is also saved in `localStorage` and reused for every future search - no need to re-upload.
- Shows one job at a time, carousel-style, instead of a list.
- Only ever surfaces roles published in the last 3 days, matched against your uploaded CV.
- Asks for original employer / ATS application URLs, avoiding paywalled boards.
- Deduplicates against every job you've ever been shown (by URL and by company+title), so the same posting never comes back twice.
- "Apply" opens the original posting in a new tab. "Applied" or "Ignore" records your decision and immediately loads the next unseen job - applied/ignored jobs are excluded from all future searches.
- Each offer includes a short, ready-to-copy "why working for us" answer for that specific posting, deliberately written short and casual rather than corporate-sounding.
- "Tailor CV for this role" generates a PDF of your CV re-emphasized for the offer you're viewing (same facts, reordered/reworded skills and bullets - never invents anything), in one of 5 layouts (Classic, Sidebar, Bold Header, Minimal, Timeline), each a single accent color.
- The menu (hamburger icon, top left) lists every job you've applied to and opens Settings, where you can switch provider, update your API key and pick the tailored-CV layout.

Locally, state is persisted per-browser in `data/clients/<clientId>.json`. There is no
server-side API key fallback - every request always carries the key from the browser's
Settings. `.env` (copy `.env.example`) only lets you override which model each provider uses.

## Deploy to Firebase

The app is set up as **Firebase Hosting** (the React frontend) + **Cloud Functions** (the Express API) + **Firestore** (per-visitor job storage, since Cloud Functions have no persistent local disk - this is why the deployed backend uses Firestore instead of the local JSON files used in `pnpm dev`).

One-time project setup (from the Firebase console, or CLI as noted):
1. **Enable Firestore** for the project (Firestore Database → Create database, native mode, pick a region). Required - it's not enabled yet.
2. **Upgrade the project to the Blaze (pay-as-you-go) plan.** Cloud Functions v2 (used here) require it. Each visitor supplies their own provider API key from Settings, so this doesn't cause API cost on your own key - it's just a Cloud Functions requirement.
3. (Optional) Override the models by creating `functions/.env` from `functions/.env.example`.

Then, whenever you want to deploy:
```bash
pnpm deploy
```
(equivalent to `vite build && firebase deploy`, which deploys hosting, functions and Firestore rules together)

Firestore is locked down (`firestore.rules` denies all direct client access) since the Cloud Function - using the Admin SDK - is the only thing that talks to it.
