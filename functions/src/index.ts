import { onRequest } from 'firebase-functions/v2/https';
import { initializeApp } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';
import express from 'express';
import cors from 'cors';
import OpenAI from 'openai';
import PDFDocument from 'pdfkit';
import crypto from 'node:crypto';

initializeApp();
const db = getFirestore();

type Status = 'new' | 'applied' | 'ignored';

type Provider = 'openai' | 'gemini';

type Job = {
  id: string;
  title: string;
  company: string;
  location: string;
  remote: string;
  publishedAt: string;
  url: string;
  source: string;
  stack: string[];
  match: number;
  why: string;
  eligibility: string;
  whyWorkingForUs: string;
  status: Status;
};

type CvExperience = {
  company: string;
  role: string;
  dates: string;
  location: string;
  bullets: string[];
};

type CvEducation = {
  school: string;
  detail: string;
  dates: string;
};

type CvData = {
  name: string;
  title: string;
  location: string;
  email: string;
  phone: string;
  summary: string;
  skills: string[];
  experience: CvExperience[];
  education: CvEducation[];
};

const CLIENT_ID_RE = /^[a-f0-9-]{16,64}$/i;
const DAYS_FRESH = 3;
const BATCH_SIZE = 5;

const jobsCollection = (clientId: string) =>
  db.collection('clients').doc(clientId).collection('jobs');

async function readJobs(clientId: string): Promise<Job[]> {
  const snap = await jobsCollection(clientId).get();
  return snap.docs.map((d) => d.data() as Job);
}

async function findNextNew(clientId: string): Promise<Job | null> {
  const snap = await jobsCollection(clientId).where('status', '==', 'new').limit(1).get();
  if (snap.empty) return null;
  return snap.docs[0].data() as Job;
}

async function writeNewJobs(clientId: string, jobs: Job[]): Promise<void> {
  const batch = db.batch();
  const col = jobsCollection(clientId);
  for (const job of jobs) {
    batch.set(col.doc(job.id), job);
  }
  await batch.commit();
}

async function updateJobStatus(
  clientId: string,
  id: string,
  status: Status
): Promise<Job | null> {
  const ref = jobsCollection(clientId).doc(id);
  const snap = await ref.get();
  if (!snap.exists) return null;
  await ref.update({ status });
  return { ...(snap.data() as Job), status };
}

async function getStats(clientId: string) {
  const col = jobsCollection(clientId);
  const [reviewed, applied, ignored] = await Promise.all([
    col.where('status', '!=', 'new').count().get(),
    col.where('status', '==', 'applied').count().get(),
    col.where('status', '==', 'ignored').count().get(),
  ]);
  return {
    reviewed: reviewed.data().count,
    applied: applied.data().count,
    ignored: ignored.data().count,
  };
}

async function listJobs(clientId: string, status?: string): Promise<Job[]> {
  const col = jobsCollection(clientId);
  const snap = status ? await col.where('status', '==', status).get() : await col.get();
  return snap.docs.map((d) => d.data() as Job);
}

const normalizeKey = (job: { company: string; title: string }) =>
  `${job.company}|${job.title}`.toLowerCase().trim().replace(/\s+/g, ' ');

const PROFILE_EXTRACTION_PROMPT = `Read the attached CV/resume carefully and return ONLY JSON (no markdown, no code fences) with this exact shape:
{"profile":"ONE dense paragraph, plain text, describing this candidate for the purpose of matching them against live job postings: name, seniority level and years of experience, primary tech stack and tools, notable strengths/specializations, most recent roles with companies and locations, and any explicit remote-work, relocation or country-eligibility preference you can infer. Also mention anything the candidate clearly should NOT be matched with. Do not invent facts not supported by the CV.","cv":{"name":"full name as written on the CV","title":"the candidate's professional title/headline, e.g. Senior Frontend Engineer","location":"city/country as written on the CV","email":"","phone":"","summary":"2-3 sentence professional summary, based only on the CV content","skills":["flat list of concrete skills/technologies exactly as they appear or are clearly implied in the CV"],"experience":[{"company":"","role":"","dates":"","location":"","bullets":["verbatim or lightly cleaned-up bullet points from this role, one per string, do not invent achievements not present in the CV"]}],"education":[{"school":"","detail":"degree/field as written","dates":""}]}}
Order "experience" most recent first, exactly as it appears in the CV. Do not fabricate any company, dates, title or bullet that is not supported by the CV content. Leave a field as an empty string/array if the CV genuinely does not contain that information.`;

function buildSearchPrompt(profile: string, existing: Job[]): string {
  const alreadyShown = existing
    .slice(-80)
    .map((j) => `${j.company} - ${j.title}`)
    .join('; ');

  return `Search the live web for ${BATCH_SIZE} currently open job postings that are an excellent fit for the candidate described below. Infer the right job titles and seniority level to search for directly from the candidate profile - do not restrict yourself to a single fixed query. Only include jobs published or freshly listed within the last ${DAYS_FRESH} days. Only include jobs the candidate is genuinely eligible to work in, based on the location/remote preferences described in their profile. Do not include jobs that are already closed. Prefer the ORIGINAL employer application URL and never return paywalled job-board links that require a login to view the posting. Verify geography and freshness from available evidence. Each returned job must be a genuinely distinct posting (different company or different role) - never list near-duplicates of each other in the same response.${
    alreadyShown
      ? ` IMPORTANT: the candidate has ALREADY been shown these jobs in previous searches, do NOT include them again, find different postings: ${alreadyShown}.`
      : ''
  }

Candidate profile: ${profile}

Return ONLY JSON with shape {"jobs":[{"title":"","company":"","location":"","remote":"","publishedAt":"human-readable age/date","url":"https://...","source":"official/ATS domain","stack":["React"],"match":0,"why":"1-2 concise sentences on why this role fits the candidate","eligibility":"short explanation of why the candidate is eligible to apply","whyWorkingForUs":"a SHORT first-person answer (max 2 short sentences, under 40 words) to \\"Why do you want to work for us?\\", specific to this company/role, plain everyday language - avoid corporate buzzwords like thrilled, passionate, align, mission, journey, avoid em dashes, sound like a real candidate typed it quickly rather than an AI: include one small natural imperfection (a minor typo, an informal contraction, a slightly awkward phrasing) - just one or two such slips, keep it understandable"}]}. Match is 0-100 based on fit with the candidate profile. No markdown, no code fences.`;
}

function normalizeProvider(value: unknown): Provider {
  return value === 'gemini' ? 'gemini' : 'openai';
}

function resolveModel(provider: Provider): string {
  return provider === 'gemini'
    ? process.env.GEMINI_MODEL || 'gemini-2.5-flash'
    : process.env.OPENAI_MODEL || 'gpt-4o';
}

function resolveApiKey(provided: unknown): string | undefined {
  return typeof provided === 'string' && provided.trim() ? provided.trim() : undefined;
}

function missingKeyError(provider: Provider): string {
  return `Set a${provider === 'gemini' ? ' Gemini' : 'n OpenAI'} API key in Settings first.`;
}

function splitDataUrl(dataUrl: string): { mimeType: string; base64: string } {
  const match = /^data:([^;]+);base64,([\s\S]*)$/.exec(dataUrl);
  return match ? { mimeType: match[1], base64: match[2] } : { mimeType: 'application/pdf', base64: dataUrl };
}

// @google/genai is ESM-only; this project builds to CommonJS, so it must be
// loaded via a dynamic import rather than a static one.
const importGenAI = () => import('@google/genai');
let genAIModulePromise: ReturnType<typeof importGenAI> | undefined;
async function newGoogleGenAI(apiKey: string) {
  if (!genAIModulePromise) genAIModulePromise = importGenAI();
  const { GoogleGenAI } = await genAIModulePromise;
  return new GoogleGenAI({ apiKey });
}

// Runs a plain text-in/JSON-out prompt against the selected provider.
async function generateJson(provider: Provider, apiKey: string, prompt: string): Promise<string> {
  if (provider === 'gemini') {
    const ai = await newGoogleGenAI(apiKey);
    const response = await ai.models.generateContent({
      model: resolveModel(provider),
      contents: prompt,
      config: { responseMimeType: 'application/json' },
    });
    return response.text ?? '';
  }

  const client = new OpenAI({ apiKey });
  const response = await client.responses.create({
    model: resolveModel(provider),
    text: { format: { type: 'json_object' } },
    input: prompt,
  });
  return response.output_text;
}

// Same as generateJson but attaches a file (e.g. the CV PDF) alongside the prompt.
async function generateJsonFromFile(
  provider: Provider,
  apiKey: string,
  prompt: string,
  fileDataUrl: string,
  filename: string
): Promise<string> {
  if (provider === 'gemini') {
    const { mimeType, base64 } = splitDataUrl(fileDataUrl);
    const ai = await newGoogleGenAI(apiKey);
    const response = await ai.models.generateContent({
      model: resolveModel(provider),
      contents: [
        { role: 'user', parts: [{ text: prompt }, { inlineData: { mimeType, data: base64 } }] },
      ],
      config: { responseMimeType: 'application/json' },
    });
    return response.text ?? '';
  }

  const client = new OpenAI({ apiKey });
  const response = await client.responses.create({
    model: resolveModel(provider),
    text: { format: { type: 'json_object' } },
    input: [
      {
        role: 'user',
        content: [
          { type: 'input_text', text: prompt },
          { type: 'input_file', file_data: fileDataUrl, filename: filename || 'cv.pdf' },
        ],
      },
    ],
  });
  return response.output_text;
}

// Runs a prompt with live web-search grounding enabled (used for job search).
async function searchWithGrounding(provider: Provider, apiKey: string, prompt: string): Promise<string> {
  if (provider === 'gemini') {
    const ai = await newGoogleGenAI(apiKey);
    const response = await ai.models.generateContent({
      model: resolveModel(provider),
      contents: prompt,
      config: { tools: [{ googleSearch: {} }] },
    });
    return response.text ?? '';
  }

  const client = new OpenAI({ apiKey });
  const response = await client.responses.create({
    model: resolveModel(provider),
    tools: [{ type: 'web_search' }],
    input: prompt,
  });
  return response.output_text;
}

async function fetchFreshJobs(
  existing: Job[],
  profile: string,
  provider: Provider,
  apiKey: string
): Promise<Job[]> {
  const raw = await searchWithGrounding(provider, apiKey, buildSearchPrompt(profile, existing));
  const text = raw
    .trim()
    .replace(/^```json\s*/, '')
    .replace(/```$/, '');
  const parsed = JSON.parse(text);

  const knownUrls = new Set(existing.map((j) => j.url));
  const knownKeys = new Set(existing.map(normalizeKey));
  const seenInBatch = new Set<string>();

  const fresh: Job[] = (parsed.jobs || [])
    .filter((x: any) => /^https?:\/\//.test(x.url))
    .filter((x: any) => !knownUrls.has(x.url))
    .filter((x: any) => !knownKeys.has(normalizeKey(x)))
    .filter((x: any) => {
      const key = normalizeKey(x);
      if (seenInBatch.has(key)) return false;
      seenInBatch.add(key);
      return true;
    })
    .map((x: any) => ({
      ...x,
      id: crypto.createHash('sha1').update(x.url).digest('hex').slice(0, 12),
      match: Math.max(0, Math.min(100, Number(x.match) || 0)),
      whyWorkingForUs: x.whyWorkingForUs || '',
      status: 'new' as Status,
    }));

  return fresh;
}

const app = express();
app.use(cors());
app.use(express.json({ limit: '20mb' }));

app.post('/api/profile', async (req, res) => {
  try {
    const { base64, filename, apiKey, provider: providerRaw } = req.body || {};
    const provider = normalizeProvider(providerRaw);
    if (!base64 || typeof base64 !== 'string') {
      return res.status(400).json({ error: 'Missing CV file.' });
    }
    const resolvedKey = resolveApiKey(apiKey);
    if (!resolvedKey) {
      return res.status(500).json({ error: missingKeyError(provider) });
    }

    const text = await generateJsonFromFile(
      provider,
      resolvedKey,
      PROFILE_EXTRACTION_PROMPT,
      base64,
      filename || 'cv.pdf'
    );
    const parsed = JSON.parse(text.trim());
    const profile = String(parsed.profile || '').trim();
    if (!profile) throw new Error('Could not read the CV.');
    res.json({ profile, cv: parsed.cv || null });
  } catch (e: any) {
    console.error(e);
    res.status(500).json({ error: e?.message || 'Could not analyze the CV.' });
  }
});

// PDFKit's own auto-pagination can revert fillColor mid-text-block when a
// single .text() call straddles a page boundary. Always pre-check remaining
// space and break ourselves so no .text() call ever needs to split a page.
function ensureRoom(doc: PDFKit.PDFDocument, height: number) {
  const bottom = doc.page.height - doc.page.margins.bottom;
  if (doc.y + height > bottom) doc.addPage();
}

function ensureRoomAt(doc: PDFKit.PDFDocument, y: number, height: number, resetTo: number): number {
  const bottom = doc.page.height - doc.page.margins.bottom;
  if (y + height > bottom) {
    doc.addPage();
    return resetTo;
  }
  return y;
}

const CV_LAYOUTS = ['classic', 'sidebar', 'bold', 'minimal', 'timeline'] as const;
type CvLayout = (typeof CV_LAYOUTS)[number];

function renderClassic(doc: PDFKit.PDFDocument, cv: CvData) {
  const ACCENT = '#1F3B8C';
  const left = doc.page.margins.left;
  const right = doc.page.width - doc.page.margins.right;

  doc.font('Helvetica-Bold').fontSize(24).fillColor('#111111').text(cv.name || 'Candidate');
  if (cv.title) {
    doc.font('Helvetica-Bold').fontSize(12.5).fillColor(ACCENT).text(cv.title.toUpperCase(), {
      characterSpacing: 0.6,
    });
  }
  doc.moveDown(0.3);
  const contact = [cv.location, cv.email, cv.phone].filter(Boolean).join('   |   ');
  if (contact) {
    doc.font('Helvetica').fontSize(9.5).fillColor('#555555').text(contact);
  }
  doc.moveDown(0.4);
  doc.moveTo(left, doc.y).lineTo(right, doc.y).lineWidth(1.5).strokeColor(ACCENT).stroke();
  doc.moveDown(0.8);
  doc.fillColor('#000000');

  function section(label: string) {
    doc.font('Helvetica-Bold').fontSize(11.5).fillColor(ACCENT).text(label.toUpperCase(), {
      characterSpacing: 0.8,
    });
    doc.moveDown(0.15);
    doc.moveTo(left, doc.y).lineTo(right, doc.y).lineWidth(0.75).strokeColor(ACCENT).stroke();
    doc.moveDown(0.35);
    doc.fillColor('#000000');
  }

  if (cv.summary) {
    section('Profile');
    doc.font('Helvetica').fontSize(10).fillColor('#222222').text(cv.summary);
    doc.moveDown(0.8);
  }

  if (cv.skills?.length) {
    section('Skills');
    doc.font('Helvetica').fontSize(10).fillColor('#222222').text(cv.skills.join('   ·   '));
    doc.moveDown(0.8);
  }

  if (cv.experience?.length) {
    section('Experience');
    for (const job of cv.experience) {
      ensureRoom(doc, 44);
      doc.font('Helvetica-Bold').fontSize(11).fillColor('#111111').text(job.role, { continued: false });
      doc.font('Helvetica').fontSize(10).fillColor(ACCENT).text(job.company);
      const meta = [job.dates, job.location].filter(Boolean).join('   ·   ');
      if (meta) doc.font('Helvetica-Oblique').fontSize(9).fillColor('#666666').text(meta);
      doc.moveDown(0.25);
      for (const bullet of job.bullets || []) {
        const t = `•  ${bullet}`;
        ensureRoom(doc, doc.heightOfString(t, { width: right - left - 4 }) + 1);
        doc.font('Helvetica').fontSize(9.5).fillColor('#222222').text(t, { indent: 4 });
      }
      doc.moveDown(0.6);
    }
  }

  if (cv.education?.length) {
    section('Education');
    for (const edu of cv.education) {
      doc.font('Helvetica-Bold').fontSize(10.5).fillColor('#111111').text(edu.school);
      const line = [edu.detail, edu.dates].filter(Boolean).join('   ·   ');
      if (line) doc.font('Helvetica').fontSize(9.5).fillColor('#555555').text(line);
      doc.moveDown(0.4);
    }
  }
}

function renderSidebar(doc: PDFKit.PDFDocument, cv: CvData) {
  const ACCENT = '#0E7C86';
  const pageW = doc.page.width;
  const sidebarW = 180;

  const paintSidebar = () => doc.rect(0, 0, sidebarW, doc.page.height).fill(ACCENT);
  paintSidebar();
  doc.on('pageAdded', paintSidebar);

  const padX = 26;
  let sy = 44;
  const sw = sidebarW - padX * 2;

  doc.fillColor('#ffffff').font('Helvetica-Bold').fontSize(18).text(cv.name || 'Candidate', padX, sy, {
    width: sw,
  });
  sy = doc.y + 6;

  if (cv.title) {
    doc.font('Helvetica').fontSize(10.5).fillColor('#D8F3F1').text(cv.title, padX, sy, { width: sw });
    sy = doc.y + 16;
  }

  function sideHeading(label: string) {
    doc.font('Helvetica-Bold').fontSize(9.5).fillColor('#ffffff').text(label.toUpperCase(), padX, sy, {
      width: sw,
      characterSpacing: 1,
    });
    sy = doc.y + 6;
    doc.moveTo(padX, sy).lineTo(padX + sw, sy).lineWidth(0.75).strokeColor('#ffffff88').stroke();
    sy += 8;
  }

  const contactLines = [cv.location, cv.email, cv.phone].filter(Boolean);
  if (contactLines.length) {
    sideHeading('Contact');
    doc.font('Helvetica').fontSize(8.7).fillColor('#EAFBFA');
    for (const line of contactLines) {
      doc.text(line, padX, sy, { width: sw });
      sy = doc.y + 4;
    }
    sy += 10;
  }

  if (cv.skills?.length) {
    sideHeading('Skills');
    doc.font('Helvetica').fontSize(8.7).fillColor('#EAFBFA');
    for (const skill of cv.skills) {
      doc.text(`•  ${skill}`, padX, sy, { width: sw });
      sy = doc.y + 3;
    }
  }

  // Main column keeps its own cursor independent of the sidebar's, always
  // resuming on the page the experience section started on.
  const mainX = sidebarW + 34;
  const mainW = pageW - mainX - doc.page.margins.right;
  let my = 48;

  function mainHeading(label: string) {
    doc.font('Helvetica-Bold').fontSize(12).fillColor(ACCENT).text(label.toUpperCase(), mainX, my, {
      width: mainW,
      characterSpacing: 0.6,
    });
    my = doc.y + 6;
    doc.moveTo(mainX, my).lineTo(mainX + mainW, my).lineWidth(0.75).strokeColor(ACCENT).stroke();
    my += 10;
  }

  if (cv.summary) {
    mainHeading('Profile');
    doc.font('Helvetica').fontSize(10).fillColor('#222222').text(cv.summary, mainX, my, { width: mainW });
    my = doc.y + 18;
  }

  if (cv.experience?.length) {
    mainHeading('Experience');
    for (const job of cv.experience) {
      my = ensureRoomAt(doc, my, 40, 48);
      doc.font('Helvetica-Bold').fontSize(10.5).fillColor('#111111').text(`${job.role} — ${job.company}`, mainX, my, {
        width: mainW,
      });
      my = doc.y + 2;

      const meta = [job.dates, job.location].filter(Boolean).join('   ·   ');
      if (meta) {
        doc.font('Helvetica-Oblique').fontSize(9).fillColor(ACCENT).text(meta, mainX, my, { width: mainW });
        my = doc.y + 6;
      }

      for (const bullet of job.bullets || []) {
        const t = `•  ${bullet}`;
        const h = doc.heightOfString(t, { width: mainW });
        my = ensureRoomAt(doc, my, h, 48);
        doc.font('Helvetica').fontSize(9.5).fillColor('#222222').text(t, mainX, my, { width: mainW });
        my = doc.y + 3;
      }
      my += 12;
    }
  }

  if (cv.education?.length) {
    mainHeading('Education');
    for (const edu of cv.education) {
      doc.font('Helvetica-Bold').fontSize(10).fillColor('#111111').text(edu.school, mainX, my, { width: mainW });
      my = doc.y + 2;
      const line = [edu.detail, edu.dates].filter(Boolean).join('   ·   ');
      if (line) {
        doc.font('Helvetica').fontSize(9).fillColor('#555555').text(line, mainX, my, { width: mainW });
        my = doc.y + 8;
      }
    }
  }
}

function renderBoldHeader(doc: PDFKit.PDFDocument, cv: CvData) {
  const ACCENT = '#E0532B';
  const pageW = doc.page.width;
  const left = doc.page.margins.left;
  const right = pageW - doc.page.margins.right;
  const bandH = 118;

  doc.rect(0, 0, pageW, bandH).fill(ACCENT);
  doc.fillColor('#ffffff').font('Helvetica-Bold').fontSize(26).text(cv.name || 'Candidate', left, 40, {
    width: right - left,
  });
  if (cv.title) {
    doc.font('Helvetica').fontSize(13).fillColor('#FFE4DA').text(cv.title, left, 74, { width: right - left });
  }
  const contact = [cv.location, cv.email, cv.phone].filter(Boolean).join('   |   ');
  if (contact) {
    doc.font('Helvetica').fontSize(9.5).fillColor('#FFD9CB').text(contact, left, 96, { width: right - left });
  }

  doc.y = bandH + 28;
  doc.x = left;

  function section(label: string) {
    doc.font('Helvetica-Bold').fontSize(12).fillColor(ACCENT).text(label.toUpperCase(), left, doc.y, {
      width: right - left,
      characterSpacing: 0.8,
    });
    doc.moveDown(0.45);
    doc.fillColor('#000000');
  }

  if (cv.summary) {
    section('Profile');
    doc.font('Helvetica').fontSize(10).fillColor('#222222').text(cv.summary, { width: right - left });
    doc.moveDown(0.9);
  }

  if (cv.skills?.length) {
    section('Skills');
    let cx = left;
    let cy = doc.y;
    const gap = 6;
    const rowH = 20;
    for (const skill of cv.skills) {
      const w = doc.font('Helvetica').fontSize(9).widthOfString(skill) + 16;
      if (cx + w > right) {
        cx = left;
        cy += rowH + gap;
      }
      doc.roundedRect(cx, cy, w, rowH, 10).lineWidth(1).strokeColor(ACCENT).stroke();
      doc.font('Helvetica').fontSize(9).fillColor(ACCENT).text(skill, cx + 8, cy + 5.5, { lineBreak: false });
      cx += w + gap;
    }
    doc.y = cy + rowH + 16;
    doc.fillColor('#000000');
  }

  if (cv.experience?.length) {
    section('Experience');
    for (const job of cv.experience) {
      ensureRoom(doc, 48);
      doc.font('Helvetica-Bold').fontSize(11).fillColor('#111111').text(job.role, { width: right - left });
      doc.font('Helvetica-Bold').fontSize(9.5).fillColor(ACCENT).text(job.company, { width: right - left });
      const meta = [job.dates, job.location].filter(Boolean).join('   ·   ');
      if (meta) doc.font('Helvetica-Oblique').fontSize(9).fillColor('#666666').text(meta, { width: right - left });
      doc.moveDown(0.25);
      for (const bullet of job.bullets || []) {
        const t = `•  ${bullet}`;
        ensureRoom(doc, doc.heightOfString(t, { width: right - left - 4 }) + 1);
        doc.font('Helvetica').fontSize(9.5).fillColor('#222222').text(t, { width: right - left, indent: 4 });
      }
      doc.moveDown(0.6);
    }
  }

  if (cv.education?.length) {
    section('Education');
    for (const edu of cv.education) {
      doc.font('Helvetica-Bold').fontSize(10.5).fillColor('#111111').text(edu.school, { width: right - left });
      const line = [edu.detail, edu.dates].filter(Boolean).join('   ·   ');
      if (line) doc.font('Helvetica').fontSize(9.5).fillColor('#555555').text(line, { width: right - left });
      doc.moveDown(0.4);
    }
  }
}

function renderMinimal(doc: PDFKit.PDFDocument, cv: CvData) {
  const ACCENT = '#1E7145';
  const left = doc.page.margins.left;
  const right = doc.page.width - doc.page.margins.right;

  doc.font('Helvetica-Bold').fontSize(25).fillColor('#161616').text(cv.name || 'Candidate');
  if (cv.title) {
    doc.font('Helvetica').fontSize(12).fillColor('#444444').text(cv.title);
  }
  doc.moveDown(0.5);
  const contact = [cv.location, cv.email, cv.phone].filter(Boolean).join('    ');
  if (contact) {
    doc.font('Helvetica').fontSize(9).fillColor(ACCENT).text(contact);
  }
  doc.moveDown(0.3);
  doc.moveTo(left, doc.y).lineTo(right, doc.y).lineWidth(1.25).strokeColor(ACCENT).stroke();
  doc.moveDown(1.1);
  doc.fillColor('#000000');

  function section(label: string) {
    doc.font('Helvetica-Bold').fontSize(10.5).fillColor(ACCENT).text(label.toUpperCase(), {
      characterSpacing: 2.2,
    });
    doc.moveDown(0.45);
    doc.fillColor('#000000');
  }

  function bullet(text: string) {
    ensureRoom(doc, doc.heightOfString(text, { width: right - left - 12 }) + 1);
    const y = doc.y;
    doc.rect(left + 1, y + 3.5, 3, 3).fill(ACCENT);
    doc.fillColor('#2a2a2a').font('Helvetica').fontSize(9.7).text(text, left + 12, y, {
      width: right - left - 12,
    });
  }

  if (cv.summary) {
    section('Profile');
    doc.font('Helvetica').fontSize(10).fillColor('#2a2a2a').text(cv.summary);
    doc.moveDown(1.1);
  }

  if (cv.skills?.length) {
    section('Skills');
    doc.font('Helvetica').fontSize(9.7).fillColor('#2a2a2a').text(cv.skills.join('   —   '));
    doc.moveDown(1.1);
  }

  if (cv.experience?.length) {
    section('Experience');
    for (const job of cv.experience) {
      ensureRoom(doc, 40);
      doc.x = left;
      doc.font('Helvetica-Bold').fontSize(10.5).fillColor('#161616').text(`${job.role}, ${job.company}`, left, doc.y, {
        width: right - left,
      });
      const meta = [job.dates, job.location].filter(Boolean).join('   ·   ');
      if (meta) {
        doc.x = left;
        doc.font('Helvetica').fontSize(8.7).fillColor('#777777').text(meta, left, doc.y, { width: right - left });
      }
      doc.moveDown(0.35);
      for (const b of job.bullets || []) {
        bullet(b);
        doc.x = left;
        doc.moveDown(0.3);
      }
      doc.moveDown(0.55);
    }
  }

  if (cv.education?.length) {
    section('Education');
    for (const edu of cv.education) {
      doc.font('Helvetica-Bold').fontSize(10).fillColor('#161616').text(edu.school);
      const line = [edu.detail, edu.dates].filter(Boolean).join('   ·   ');
      if (line) doc.font('Helvetica').fontSize(9).fillColor('#666666').text(line);
      doc.moveDown(0.5);
    }
  }
}

function renderTimeline(doc: PDFKit.PDFDocument, cv: CvData) {
  const ACCENT = '#5B3FBF';
  const left = doc.page.margins.left;
  const right = doc.page.width - doc.page.margins.right;

  doc.font('Helvetica-Bold').fontSize(24).fillColor('#111111').text(cv.name || 'Candidate');
  if (cv.title) {
    doc.font('Helvetica').fontSize(12.5).fillColor(ACCENT).text(cv.title);
  }
  const contact = [cv.location, cv.email, cv.phone].filter(Boolean).join('   ·   ');
  if (contact) {
    doc.moveDown(0.25);
    doc.font('Helvetica').fontSize(9.5).fillColor('#555555').text(contact);
  }
  doc.moveDown(0.9);
  doc.fillColor('#000000');

  function section(label: string) {
    doc.font('Helvetica-Bold').fontSize(11.5).fillColor(ACCENT).text(label.toUpperCase(), {
      characterSpacing: 0.8,
    });
    doc.moveDown(0.4);
    doc.fillColor('#000000');
  }

  if (cv.summary) {
    section('Profile');
    doc.font('Helvetica').fontSize(10).fillColor('#222222').text(cv.summary);
    doc.moveDown(0.9);
  }

  if (cv.skills?.length) {
    section('Skills');
    doc.font('Helvetica').fontSize(10).fillColor('#222222').text(cv.skills.join('   ·   '));
    doc.moveDown(0.9);
  }

  if (cv.experience?.length) {
    section('Experience');
    const lineX = left + 4;
    for (const job of cv.experience) {
      const heading = `${job.role} — ${job.company}`;
      const textX = left + 18;
      const textW = right - textX;
      const headingH = doc.heightOfString(heading, { width: textW });
      ensureRoom(doc, headingH + 16);

      const entryTop = doc.y;
      doc.circle(lineX, entryTop + 4, 3.2).fill(ACCENT);

      doc.font('Helvetica-Bold').fontSize(10.5).fillColor('#111111').text(heading, textX, entryTop, {
        width: textW,
      });
      let y = doc.y;
      const meta = [job.dates, job.location].filter(Boolean).join('   ·   ');
      if (meta) {
        doc.font('Helvetica-Oblique').fontSize(9).fillColor(ACCENT).text(meta, textX, y, { width: textW });
        y = doc.y;
      }
      y += 3;
      for (const bullet of job.bullets || []) {
        const t = `•  ${bullet}`;
        const h = doc.heightOfString(t, { width: textW });
        y = ensureRoomAt(doc, y, h, doc.page.margins.top);
        doc.font('Helvetica').fontSize(9.5).fillColor('#222222').text(t, textX, y, { width: textW });
        y = doc.y;
      }

      if (y >= entryTop) {
        doc.moveTo(lineX, entryTop + 8).lineTo(lineX, y + 10).lineWidth(1.5).strokeColor('#D9D2F5').stroke();
      }

      doc.y = y + 14;
      doc.x = left;
    }
  }

  if (cv.education?.length) {
    section('Education');
    for (const edu of cv.education) {
      doc.font('Helvetica-Bold').fontSize(10.5).fillColor('#111111').text(edu.school);
      const line = [edu.detail, edu.dates].filter(Boolean).join('   ·   ');
      if (line) doc.font('Helvetica').fontSize(9.5).fillColor('#555555').text(line);
      doc.moveDown(0.4);
    }
  }
}

const CV_RENDERERS: Record<CvLayout, (doc: PDFKit.PDFDocument, cv: CvData) => void> = {
  classic: renderClassic,
  sidebar: renderSidebar,
  bold: renderBoldHeader,
  minimal: renderMinimal,
  timeline: renderTimeline,
};

function renderCvPdf(cv: CvData, layout: string): Promise<Buffer> {
  const render = CV_RENDERERS[layout as CvLayout] || renderClassic;
  return new Promise((resolve, reject) => {
    const doc = new PDFDocument({ margin: 48, size: 'A4' });
    const chunks: Buffer[] = [];
    doc.on('data', (chunk) => chunks.push(chunk));
    doc.on('end', () => resolve(Buffer.concat(chunks)));
    doc.on('error', reject);
    render(doc, cv);
    doc.end();
  });
}


function buildTailorPrompt(
  cv: CvData,
  job: { title: string; company: string; why: string; stack: string[]; eligibility: string }
): string {
  return `You are tailoring a candidate's CV for ONE specific job posting. Return ONLY JSON (no markdown, no code fences) with the exact same shape as the input CV JSON below: {"name","title","location","email","phone","summary","skills":[],"experience":[{"company","role","dates","location","bullets":[]}],"education":[{"school","detail","dates"}]}.

You may: rewrite the "summary" to speak directly to this role, reorder "skills" so the most relevant ones for this job come first, reorder and lightly rephrase "bullets" within each experience entry to foreground what's most relevant to this job. Reordering "experience" entries is NOT allowed (keep original order/dates/companies/titles unchanged).

You must NOT: invent any company, job title, date range, skill, degree or achievement that is not present in the original CV below. Do not change facts. Only re-emphasize, reorder and lightly rephrase what is already true.

Target job:
Title: ${job.title}
Company: ${job.company}
Why it fits: ${job.why}
Stack: ${(job.stack || []).join(', ')}
Eligibility notes: ${job.eligibility}

Original CV JSON:
${JSON.stringify(cv)}`;
}

app.post('/api/tailor-cv', async (req, res) => {
  try {
    const { cv, job, layout, apiKey, provider: providerRaw } = req.body || {};
    const provider = normalizeProvider(providerRaw);
    if (!cv || typeof cv !== 'object') {
      return res.status(400).json({ error: 'Missing CV data. Re-upload your CV.' });
    }
    if (!job || typeof job !== 'object') {
      return res.status(400).json({ error: 'Missing job.' });
    }
    const resolvedKey = resolveApiKey(apiKey);
    if (!resolvedKey) {
      return res.status(500).json({ error: missingKeyError(provider) });
    }

    const text = await generateJson(provider, resolvedKey, buildTailorPrompt(cv, job));
    const tailored: CvData = JSON.parse(text.trim());
    const pdf = await renderCvPdf(tailored, typeof layout === 'string' ? layout : 'classic');

    res.setHeader('Content-Type', 'application/pdf');
    const stamp = new Date().toISOString().replace(/[-:]/g, '').replace('T', '-').slice(0, 15);
    const namePart = String(cv.name || 'CV')
      .trim()
      .replace(/\s+/g, '_')
      .replace(/[^a-zA-Z0-9_]/g, '');
    res.setHeader('Content-Disposition', `attachment; filename="${namePart}_${stamp}.pdf"`);
    res.send(pdf);
  } catch (e: any) {
    console.error(e);
    res.status(500).json({ error: e?.message || 'Could not tailor your CV.' });
  }
});

app.get('/api/stats', async (req, res) => {
  const clientId = String(req.query.clientId || '');
  if (!CLIENT_ID_RE.test(clientId)) {
    return res.json({ reviewed: 0, applied: 0, ignored: 0 });
  }
  res.json(await getStats(clientId));
});

app.get('/api/jobs', async (req, res) => {
  const clientId = String(req.query.clientId || '');
  if (!CLIENT_ID_RE.test(clientId)) return res.json([]);
  const status = req.query.status;
  res.json(await listJobs(clientId, typeof status === 'string' ? status : undefined));
});

app.post('/api/jobs/next', async (req, res) => {
  try {
    const { profile, clientId, apiKey, provider: providerRaw } = req.body || {};
    const provider = normalizeProvider(providerRaw);
    if (!clientId || !CLIENT_ID_RE.test(clientId)) {
      return res.status(400).json({ error: 'Missing or invalid clientId.' });
    }
    if (!profile || typeof profile !== 'string' || profile.trim().length < 10) {
      return res.status(400).json({ error: 'Missing candidate profile. Upload your CV first.' });
    }
    const resolvedKey = resolveApiKey(apiKey);
    if (!resolvedKey) {
      return res.status(500).json({ error: missingKeyError(provider) });
    }

    const next = await findNextNew(clientId);
    if (next) return res.json({ job: next });

    const existing = await readJobs(clientId);
    const fresh = await fetchFreshJobs(existing, profile, provider, resolvedKey);
    if (!fresh.length) return res.json({ job: null });

    await writeNewJobs(clientId, fresh);
    res.json({ job: fresh[0] });
  } catch (e: any) {
    console.error(e);
    res.status(500).json({ error: e?.message || 'Search failed' });
  }
});

app.patch('/api/jobs/:id', async (req, res) => {
  const { status, clientId } = req.body || {};
  if (!clientId || !CLIENT_ID_RE.test(clientId)) {
    return res.status(400).json({ error: 'Missing or invalid clientId.' });
  }
  const updated = await updateJobStatus(clientId, req.params.id, status);
  if (!updated) return res.status(404).json({ error: 'Not found' });
  res.json(updated);
});

export const api = onRequest(app);
