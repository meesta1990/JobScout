import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Container,
  CssBaseline,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  Drawer,
  IconButton,
  InputAdornment,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Skeleton,
  Snackbar,
  Stack,
  TextField,
  ThemeProvider,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material';
import UploadFileRoundedIcon from '@mui/icons-material/UploadFileRounded';
import AutoAwesomeRoundedIcon from '@mui/icons-material/AutoAwesomeRounded';
import OpenInNewRoundedIcon from '@mui/icons-material/OpenInNewRounded';
import CheckCircleOutlineRoundedIcon from '@mui/icons-material/CheckCircleOutlineRounded';
import ArrowForwardRoundedIcon from '@mui/icons-material/ArrowForwardRounded';
import ContentCopyRoundedIcon from '@mui/icons-material/ContentCopyRounded';
import AutoFixHighRoundedIcon from '@mui/icons-material/AutoFixHighRounded';
import MenuRoundedIcon from '@mui/icons-material/MenuRounded';
import SettingsRoundedIcon from '@mui/icons-material/SettingsRounded';
import WorkOutlineRoundedIcon from '@mui/icons-material/WorkOutlineRounded';
import VisibilityRoundedIcon from '@mui/icons-material/VisibilityRounded';
import VisibilityOffRoundedIcon from '@mui/icons-material/VisibilityOffRounded';
import { theme } from './theme';
import './firebase';

type Status = 'new' | 'applied' | 'ignored';

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

type Stats = {
  reviewed: number;
  applied: number;
  ignored: number;
};

type Provider = 'openai' | 'gemini' | 'openrouter';

const PROFILE_KEY = 'jobscout.profile';
const CV_DATA_KEY = 'jobscout.cvData';
const CLIENT_ID_KEY = 'jobscout.clientId';
const API_KEY_KEY = 'jobscout.apiKey';
const GEMINI_API_KEY_KEY = 'jobscout.geminiApiKey';
const OPENROUTER_API_KEY_KEY = 'jobscout.openrouterApiKey';
const PROVIDER_KEY = 'jobscout.provider';
const CV_LAYOUT_KEY = 'jobscout.cvLayout';

const PROVIDERS: { id: Provider; name: string; keyPlaceholder: string; keyLink: string; keyLinkLabel: string }[] = [
  {
    id: 'openai',
    name: 'ChatGPT (OpenAI)',
    keyPlaceholder: 'sk-...',
    keyLink: 'https://platform.openai.com/api-keys',
    keyLinkLabel: 'platform.openai.com/api-keys',
  },
  {
    id: 'gemini',
    name: 'Gemini (Google)',
    keyPlaceholder: 'AIza...',
    keyLink: 'https://aistudio.google.com/apikey',
    keyLinkLabel: 'aistudio.google.com/apikey',
  },
  {
    id: 'openrouter',
    name: 'OpenRouter',
    keyPlaceholder: 'sk-or-...',
    keyLink: 'https://openrouter.ai/keys',
    keyLinkLabel: 'openrouter.ai/keys',
  },
];

const API_KEY_STORAGE: Record<Provider, string> = {
  openai: API_KEY_KEY,
  gemini: GEMINI_API_KEY_KEY,
  openrouter: OPENROUTER_API_KEY_KEY,
};

const CV_LAYOUTS = [
  { id: 'classic', name: 'Classic', color: '#1F3B8C' },
  { id: 'sidebar', name: 'Sidebar', color: '#0E7C86' },
  { id: 'bold', name: 'Bold Header', color: '#E0532B' },
  { id: 'minimal', name: 'Minimal', color: '#1E7145' },
  { id: 'timeline', name: 'Timeline', color: '#5B3FBF' },
] as const;

function getProvider(): Provider {
  try {
    const stored = localStorage.getItem(PROVIDER_KEY);
    return stored === 'gemini' || stored === 'openrouter' ? stored : 'openai';
  } catch {
    return 'openai';
  }
}

function getKeyForProvider(provider: Provider): string {
  try {
    return localStorage.getItem(API_KEY_STORAGE[provider]) || '';
  } catch {
    return '';
  }
}

function getActiveApiKey(): string {
  return getKeyForProvider(getProvider());
}

function getCvLayout(): string {
  try {
    return localStorage.getItem(CV_LAYOUT_KEY) || 'classic';
  } catch {
    return 'classic';
  }
}

function getOrCreateClientId(): string {
  try {
    let id = localStorage.getItem(CLIENT_ID_KEY);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(CLIENT_ID_KEY, id);
    }
    return id;
  } catch {
    return crypto.randomUUID();
  }
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function matchColor(match: number): 'success' | 'primary' | 'default' {
  if (match >= 85) return 'success';
  if (match >= 70) return 'primary';
  return 'default';
}

function CvUpload({
  onProfileReady,
  onOpenMenu,
}: {
  onProfileReady: (profile: string) => void;
  onOpenMenu: () => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState('');

  async function analyze() {
    if (!file) return;
    setAnalyzing(true);
    setError('');

    try {
      const base64 = await fileToDataUrl(file);
      const response = await fetch('/api/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base64,
          filename: file.name,
          provider: getProvider(),
          apiKey: getActiveApiKey() || undefined,
        }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Could not analyze your CV.');
      }

      localStorage.setItem(PROFILE_KEY, data.profile);
      if (data.cv) {
        localStorage.setItem(CV_DATA_KEY, JSON.stringify(data.cv));
      }
      onProfileReady(data.profile);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not analyze your CV.');
    } finally {
      setAnalyzing(false);
    }
  }

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        px: 3,
      }}
    >
      <IconButton onClick={onOpenMenu} sx={{ position: 'fixed', top: 16, left: 16 }}>
        <MenuRoundedIcon />
      </IconButton>

      <Card sx={{ maxWidth: 460, width: '100%' }}>
        <CardContent sx={{ textAlign: 'center', py: 5, px: 4 }}>
          <Typography variant="h4" sx={{ fontWeight: 800 }} gutterBottom>
            Job Scout
          </Typography>
          <Typography color="text.secondary" sx={{ mb: 4 }}>
            Upload your CV and get fresh job offers matched to your background.
          </Typography>

          <Box
            component="label"
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 1.5,
              border: '2px dashed',
              borderColor: file ? 'primary.main' : 'divider',
              borderRadius: 3,
              py: 5,
              px: 3,
              mb: 3,
              cursor: analyzing ? 'default' : 'pointer',
              transition: 'border-color .15s ease',
              '&:hover': analyzing ? {} : { borderColor: 'primary.main' },
            }}
          >
            <input
              hidden
              type="file"
              accept="application/pdf"
              disabled={analyzing}
              onChange={(event) => setFile(event.target.files?.[0] || null)}
            />
            <UploadFileRoundedIcon
              sx={{ fontSize: 42, color: file ? 'primary.main' : 'text.secondary' }}
            />
            <Typography variant="body2" color={file ? 'text.primary' : 'text.secondary'}>
              {file ? file.name : 'Click to choose your CV (PDF)'}
            </Typography>
          </Box>

          <Button
            fullWidth
            size="large"
            variant="contained"
            disabled={!file || analyzing}
            onClick={analyze}
            startIcon={analyzing ? undefined : <AutoAwesomeRoundedIcon />}
          >
            {analyzing ? <CircularProgress size={22} color="inherit" /> : 'Analyze CV'}
          </Button>

          {error && (
            <Alert severity="error" sx={{ mt: 3, textAlign: 'left' }}>
              {error}
            </Alert>
          )}
        </CardContent>
      </Card>
    </Box>
  );
}

function ApiKeySetup({ onSaved }: { onSaved: () => void }) {
  const [provider, setProvider] = useState<Provider>(() => getProvider());
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const providerInfo = PROVIDERS.find((p) => p.id === provider)!;

  function save() {
    if (!apiKey.trim()) return;
    try {
      localStorage.setItem(PROVIDER_KEY, provider);
      localStorage.setItem(API_KEY_STORAGE[provider], apiKey.trim());
    } catch {
      // localStorage unavailable, key will just need to be re-entered next time
    }
    onSaved();
  }

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        px: 3,
      }}
    >
      <Card sx={{ maxWidth: 460, width: '100%' }}>
        <CardContent sx={{ textAlign: 'center', py: 5, px: 4 }}>
          <Typography variant="h4" sx={{ fontWeight: 800 }} gutterBottom>
            Job Scout
          </Typography>
          <Typography color="text.secondary" sx={{ mb: 3 }}>
            This app calls an AI provider on your behalf, so it needs your own API key. It's
            stored only in this browser and sent directly with your requests - never saved on any
            server.
          </Typography>

          <ToggleButtonGroup
            fullWidth
            exclusive
            size="small"
            value={provider}
            onChange={(_event, value) => value && setProvider(value)}
            sx={{ mb: 2 }}
          >
            {PROVIDERS.map((p) => (
              <ToggleButton key={p.id} value={p.id}>
                {p.name}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>

          <TextField
            fullWidth
            size="small"
            type={showKey ? 'text' : 'password'}
            placeholder={providerInfo.keyPlaceholder}
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') save();
            }}
            slotProps={{
              input: {
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton size="small" onClick={() => setShowKey((v) => !v)}>
                      {showKey ? (
                        <VisibilityOffRoundedIcon fontSize="small" />
                      ) : (
                        <VisibilityRoundedIcon fontSize="small" />
                      )}
                    </IconButton>
                  </InputAdornment>
                ),
              },
            }}
          />

          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1.5, mb: 3 }}>
            Don't have one? Grab it from{' '}
            <a href={providerInfo.keyLink} target="_blank" rel="noreferrer" style={{ color: 'inherit' }}>
              {providerInfo.keyLinkLabel}
            </a>
            . You can change it later from the menu → Settings.
          </Typography>

          <Button fullWidth size="large" variant="contained" disabled={!apiKey.trim()} onClick={save}>
            Continue
          </Button>
        </CardContent>
      </Card>
    </Box>
  );
}

function JobCardSkeleton() {
  return (
    <Card>
      <CardContent sx={{ p: 3 }}>
        <Stack
          direction="row"
          sx={{ justifyContent: 'space-between', alignItems: 'flex-start' }}
        >
          <Box sx={{ width: '65%' }}>
            <Skeleton variant="text" width="90%" height={34} />
            <Skeleton variant="text" width="55%" height={24} />
          </Box>
          <Skeleton variant="circular" width={44} height={44} />
        </Stack>

        <Skeleton variant="text" width="60%" sx={{ mt: 2 }} />

        <Stack direction="row" spacing={1} sx={{ mt: 2 }}>
          <Skeleton variant="rounded" width={64} height={26} />
          <Skeleton variant="rounded" width={84} height={26} />
          <Skeleton variant="rounded" width={56} height={26} />
        </Stack>

        <Skeleton variant="text" sx={{ mt: 2.5 }} />
        <Skeleton variant="text" width="85%" />

        <Skeleton variant="rounded" height={80} sx={{ mt: 2 }} />

        <Stack direction="row" spacing={1} sx={{ mt: 3 }}>
          <Skeleton variant="rounded" width="100%" height={42} />
          <Skeleton variant="rounded" width="100%" height={42} />
          <Skeleton variant="rounded" width="100%" height={42} />
        </Stack>
      </CardContent>
    </Card>
  );
}

function SettingsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [provider, setProvider] = useState<Provider>(() => getProvider());
  const [apiKey, setApiKey] = useState(() => getKeyForProvider(getProvider()));
  const [showKey, setShowKey] = useState(false);
  const [layout, setLayout] = useState(() => getCvLayout());
  const providerInfo = PROVIDERS.find((p) => p.id === provider)!;

  function selectProvider(next: Provider) {
    setProvider(next);
    setApiKey(getKeyForProvider(next));
  }

  function save() {
    try {
      localStorage.setItem(PROVIDER_KEY, provider);
      const key = API_KEY_STORAGE[provider];
      if (apiKey.trim()) {
        localStorage.setItem(key, apiKey.trim());
      } else {
        localStorage.removeItem(key);
      }
      localStorage.setItem(CV_LAYOUT_KEY, layout);
    } catch {
      // localStorage unavailable, settings just won't persist across reloads
    }
    onClose();
  }

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="xs">
      <DialogTitle>Settings</DialogTitle>
      <DialogContent>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          AI provider
        </Typography>
        <ToggleButtonGroup
          fullWidth
          exclusive
          size="small"
          value={provider}
          onChange={(_event, value) => value && selectProvider(value)}
          sx={{ mb: 2 }}
        >
          {PROVIDERS.map((p) => (
            <ToggleButton key={p.id} value={p.id}>
              {p.name}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>

        <Typography variant="subtitle2" sx={{ mb: 1 }}>
          Your {providerInfo.name} API key
        </Typography>
        <TextField
          fullWidth
          size="small"
          type={showKey ? 'text' : 'password'}
          placeholder={providerInfo.keyPlaceholder}
          value={apiKey}
          onChange={(event) => setApiKey(event.target.value)}
          slotProps={{
            input: {
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton size="small" onClick={() => setShowKey((v) => !v)}>
                    {showKey ? (
                      <VisibilityOffRoundedIcon fontSize="small" />
                    ) : (
                      <VisibilityRoundedIcon fontSize="small" />
                    )}
                  </IconButton>
                </InputAdornment>
              ),
            },
          }}
        />
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
          Required for every search and CV action from this browser - stored only here, never on
          any server. Clearing it will make those actions fail until you set one again. Grab a key
          from{' '}
          <a href={providerInfo.keyLink} target="_blank" rel="noreferrer" style={{ color: 'inherit' }}>
            {providerInfo.keyLinkLabel}
          </a>
          .
        </Typography>

        <Typography variant="subtitle2" sx={{ mt: 3, mb: 1.5 }}>
          Tailored CV layout
        </Typography>
        <Stack direction="row" sx={{ flexWrap: 'wrap', gap: 1.5 }}>
          {CV_LAYOUTS.map((l) => (
            <Box
              key={l.id}
              onClick={() => setLayout(l.id)}
              sx={{
                width: 88,
                cursor: 'pointer',
                border: '2px solid',
                borderColor: layout === l.id ? l.color : 'divider',
                borderRadius: 2,
                p: 1,
                textAlign: 'center',
              }}
            >
              <Box sx={{ height: 36, borderRadius: 1, mb: 0.75, backgroundColor: l.color }} />
              <Typography variant="caption">{l.name}</Typography>
            </Box>
          ))}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="contained" onClick={save}>
          Save
        </Button>
      </DialogActions>
    </Dialog>
  );
}

function App() {
  const clientIdRef = useRef(getOrCreateClientId());
  const [apiKeyReady, setApiKeyReady] = useState(() => !!getActiveApiKey());
  const [profile, setProfile] = useState<string | null>(() =>
    localStorage.getItem(PROFILE_KEY)
  );

  const [job, setJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [copySnackOpen, setCopySnackOpen] = useState(false);
  const [tailoring, setTailoring] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [appliedJobs, setAppliedJobs] = useState<Job[]>([]);
  const [appliedLoading, setAppliedLoading] = useState(false);
  const [stats, setStats] = useState<Stats>({
    reviewed: 0,
    applied: 0,
    ignored: 0,
  });

  const loadStats = useCallback(async () => {
    try {
      const response = await fetch(`/api/stats?clientId=${clientIdRef.current}`);
      setStats(await response.json());
    } catch {
      // stats are non-critical, ignore failures
    }
  }, []);

  async function openMenu() {
    setMenuOpen(true);
    setAppliedLoading(true);
    try {
      const response = await fetch(
        `/api/jobs?clientId=${clientIdRef.current}&status=applied`
      );
      setAppliedJobs(await response.json());
    } catch {
      setAppliedJobs([]);
    } finally {
      setAppliedLoading(false);
    }
  }

  const loadNext = useCallback(async () => {
    if (!profile) return;
    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/jobs/next', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profile,
          clientId: clientIdRef.current,
          provider: getProvider(),
          apiKey: getActiveApiKey() || undefined,
        }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Search failed');
      }

      setJob(data.job);
    } catch (error) {
      setJob(null);
      setError(error instanceof Error ? error.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  }, [profile]);

  useEffect(() => {
    if (profile && apiKeyReady) {
      loadNext();
      loadStats();
    }
  }, [profile, apiKeyReady]);

  async function decide(status: Status) {
    if (!job) return;

    await fetch(`/api/jobs/${job.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status, clientId: clientIdRef.current }),
    });

    await loadStats();
    await loadNext();
  }

  function changeCv() {
    localStorage.removeItem(PROFILE_KEY);
    localStorage.removeItem(CV_DATA_KEY);
    localStorage.removeItem(CLIENT_ID_KEY);
    window.location.reload();
  }

  async function tailorCv() {
    if (!job) return;
    const cvDataRaw = localStorage.getItem(CV_DATA_KEY);
    if (!cvDataRaw) {
      setError('No CV data found on this browser - use "Change CV" to re-upload it.');
      return;
    }

    setTailoring(true);
    setError('');

    try {
      const cvData = JSON.parse(cvDataRaw);
      const response = await fetch('/api/tailor-cv', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cv: cvData,
          job: {
            title: job.title,
            company: job.company,
            why: job.why,
            stack: job.stack,
            eligibility: job.eligibility,
          },
          layout: getCvLayout(),
          provider: getProvider(),
          apiKey: getActiveApiKey() || undefined,
        }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || 'Could not tailor your CV.');
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const stamp = new Date()
        .toISOString()
        .replace(/[-:]/g, '')
        .replace('T', '-')
        .slice(0, 15);
      const namePart = String(cvData.name || 'CV')
        .trim()
        .replace(/\s+/g, '_')
        .replace(/[^a-zA-Z0-9_]/g, '');
      const link = document.createElement('a');
      link.href = url;
      link.download = `${namePart}_${stamp}.pdf`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not tailor your CV.');
    } finally {
      setTailoring(false);
    }
  }

  async function copyWhyUs() {
    if (!job) return;
    try {
      await navigator.clipboard.writeText(job.whyWorkingForUs);
      setCopySnackOpen(true);
    } catch {
      // clipboard API unavailable, user can still select the text manually
    }
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />

      {!apiKeyReady && <ApiKeySetup onSaved={() => setApiKeyReady(true)} />}

      {apiKeyReady && !profile && (
        <CvUpload onProfileReady={setProfile} onOpenMenu={openMenu} />
      )}

      {apiKeyReady && profile && (
      <Container maxWidth="sm" sx={{ py: 6 }}>
        <Stack
          direction="row"
          sx={{ justifyContent: 'space-between', alignItems: 'flex-start', mb: 3 }}
        >
          <Stack direction="row" spacing={1.5} sx={{ alignItems: 'flex-start' }}>
            <IconButton onClick={openMenu} sx={{ mt: 0.5 }}>
              <MenuRoundedIcon />
            </IconButton>
            <Box>
              <Typography variant="h4" sx={{ fontWeight: 800 }}>
                Job Scout
              </Typography>
              <Typography color="text.secondary">
                One offer at a time, never older than 3 days.
              </Typography>
            </Box>
          </Stack>

          <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
            <Button size="small" onClick={changeCv}>
              Change CV
            </Button>
          </Stack>
        </Stack>

        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          <b>{stats.reviewed}</b> reviewed · <b>{stats.applied}</b> applied ·{' '}
          <b>{stats.ignored}</b> ignored
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {loading && <JobCardSkeleton />}

        {!loading && !error && !job && (
          <Card sx={{ textAlign: 'center' }}>
            <CardContent sx={{ py: 6 }}>
              <Typography color="text.secondary" sx={{ mb: 2 }}>
                No new offers found right now.
              </Typography>
              <Button variant="outlined" onClick={loadNext}>
                Retry
              </Button>
            </CardContent>
          </Card>
        )}

        {!loading && job && (
          <Card>
            <CardContent sx={{ p: 3 }}>
              <Stack direction="row" spacing={2} sx={{ justifyContent: 'space-between' }}>
                <Box>
                  <Typography variant="h6" sx={{ fontWeight: 700, lineHeight: 1.3 }}>
                    {job.title}
                  </Typography>
                  <Typography color="text.secondary">{job.company}</Typography>
                </Box>

                <Chip
                  label={`${job.match}%`}
                  color={matchColor(job.match)}
                  sx={{ fontWeight: 800, fontSize: 15, height: 34, flexShrink: 0 }}
                />
              </Stack>

              <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                {job.location} · {job.remote} · {job.publishedAt}
              </Typography>

              <Stack direction="row" sx={{ flexWrap: 'wrap', gap: 1, mt: 2 }}>
                {job.stack.map((technology) => (
                  <Chip key={technology} label={technology} size="small" variant="outlined" />
                ))}
              </Stack>

              <Typography sx={{ mt: 2 }}>{job.why}</Typography>

              <Alert severity="success" variant="outlined" sx={{ mt: 2 }}>
                {job.eligibility}
              </Alert>

              {job.whyWorkingForUs && (
                <Box sx={{ mt: 2.5 }}>
                  <Stack
                    direction="row"
                    sx={{ justifyContent: 'space-between', alignItems: 'center', mb: 1 }}
                  >
                    <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 700 }}>
                      WHY WORKING FOR US · ready to copy-paste
                    </Typography>
                    <IconButton size="small" onClick={copyWhyUs}>
                      <ContentCopyRoundedIcon fontSize="small" />
                    </IconButton>
                  </Stack>
                  <TextField
                    fullWidth
                    multiline
                    minRows={3}
                    size="small"
                    value={job.whyWorkingForUs}
                    slotProps={{ input: { readOnly: true } }}
                  />
                </Box>
              )}

              <Stack direction="row" spacing={1} sx={{ mt: 3 }}>
                <Button
                  fullWidth
                  variant="contained"
                  component="a"
                  href={job.url}
                  target="_blank"
                  rel="noreferrer"
                  endIcon={<OpenInNewRoundedIcon />}
                >
                  Apply
                </Button>

                <Button
                  fullWidth
                  variant="outlined"
                  color="success"
                  startIcon={<CheckCircleOutlineRoundedIcon />}
                  onClick={() => decide('applied')}
                >
                  Applied
                </Button>

                <Button
                  fullWidth
                  variant="outlined"
                  color="inherit"
                  endIcon={<ArrowForwardRoundedIcon />}
                  onClick={() => decide('ignored')}
                >
                  Ignore
                </Button>
              </Stack>

              <Button
                fullWidth
                variant="outlined"
                color="primary"
                sx={{ mt: 1.5 }}
                disabled={tailoring}
                onClick={tailorCv}
                startIcon={tailoring ? undefined : <AutoFixHighRoundedIcon />}
              >
                {tailoring ? (
                  <CircularProgress size={20} color="inherit" />
                ) : (
                  'Tailor CV for this role'
                )}
              </Button>
            </CardContent>
          </Card>
        )}
      </Container>
      )}

      <Snackbar
        open={copySnackOpen}
        autoHideDuration={2000}
        onClose={() => setCopySnackOpen(false)}
        message="Copied to clipboard"
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      />

      <Drawer anchor="left" open={menuOpen} onClose={() => setMenuOpen(false)}>
        <Box sx={{ width: 300, p: 2.5 }}>
          <Typography variant="h6" sx={{ fontWeight: 800, mb: 2 }}>
            Menu
          </Typography>

          <Stack direction="row" spacing={0.75} sx={{ alignItems: 'center' }}>
            <WorkOutlineRoundedIcon fontSize="small" color="action" />
            <Typography variant="overline" color="text.secondary">
              Applied jobs
            </Typography>
          </Stack>

          {appliedLoading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 3 }}>
              <CircularProgress size={22} />
            </Box>
          )}

          {!appliedLoading && appliedJobs.length === 0 && (
            <Typography variant="body2" color="text.secondary" sx={{ py: 2 }}>
              No applications yet.
            </Typography>
          )}

          {!appliedLoading && appliedJobs.length > 0 && (
            <List dense sx={{ mb: 1 }}>
              {appliedJobs.map((appliedJob) => (
                <ListItemButton
                  key={appliedJob.id}
                  component="a"
                  href={appliedJob.url}
                  target="_blank"
                  rel="noreferrer"
                  sx={{ borderRadius: 1 }}
                >
                  <ListItemText primary={appliedJob.title} secondary={appliedJob.company} />
                </ListItemButton>
              ))}
            </List>
          )}

          <Divider sx={{ my: 2 }} />

          <ListItemButton
            sx={{ borderRadius: 1 }}
            onClick={() => {
              setMenuOpen(false);
              setSettingsOpen(true);
            }}
          >
            <ListItemIcon sx={{ minWidth: 36 }}>
              <SettingsRoundedIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Settings" />
          </ListItemButton>
        </Box>
      </Drawer>

      <SettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </ThemeProvider>
  );
}

createRoot(document.getElementById('root')!).render(<App />);
