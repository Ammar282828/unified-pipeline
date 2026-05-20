require('dotenv').config();

const express      = require('express');
const multer       = require('multer');
const path         = require('path');
const fs           = require('fs');
const os           = require('os');
const { execFile } = require('child_process');
const sharp        = require('sharp');
const { GoogleGenAI } = require('@google/genai');
const OpenAI       = require('openai');

process.on('uncaughtException', (err) => {
    console.error('[Uncaught Exception]', err.message, err.stack);
});
process.on('unhandledRejection', (reason) => {
    console.error('[Unhandled Rejection]', reason);
});

const PORT = process.env.PORT || 3000;

// ── Provider clients ───────────────────────────────────────────────────────
// Multi-key support: GOOGLE_API_KEYS=key1,key2,... Each key gets its own
// GoogleGenAI client; calls round-robin across them and 429s mark a key
// as cooled-down for 60s so parallel work tolerates per-key quota limits.
// (Backward compat: GOOGLE_API_KEY is also accepted as a single key.)
const _rawKeys = (process.env.GOOGLE_API_KEYS || process.env.GOOGLE_API_KEY || '')
    .split(',').map(s => s.trim()).filter(Boolean);
const USE_VERTEX = process.env.GOOGLE_GENAI_USE_VERTEXAI === 'true'
    || _rawKeys.some(k => k.startsWith('AQ.'));
const geminiClients = _rawKeys.map(apiKey => new GoogleGenAI({ vertexai: USE_VERTEX, apiKey }));
const geminiClient = geminiClients[0] || null;   // legacy single-client refs
const _clientCooldownUntil = geminiClients.map(() => 0);
let _nextClientIdx = 0;
const COOLDOWN_MS = 60_000; // 1 minute after a 429
// Returns a ready key, or — if every key is cooling — waits for the soonest
// one to free up. Prevents cascading 429s when Vertex Express quota is
// exhausted across all 5 keys. (Audit #8.)
async function pickGeminiClient() {
    if (geminiClients.length === 0) return null;
    while (true) {
        const now = Date.now();
        // Round-robin starting at _nextClientIdx, prefer keys not in cooldown
        for (let i = 0; i < geminiClients.length; i++) {
            const idx = (_nextClientIdx + i) % geminiClients.length;
            if (_clientCooldownUntil[idx] <= now) {
                _nextClientIdx = (idx + 1) % geminiClients.length;
                return { client: geminiClients[idx], idx };
            }
        }
        // All cooling — wait for the soonest to be ready, then re-pick.
        const earliest = Math.min(..._clientCooldownUntil);
        const wait = Math.max(250, earliest - now); // never busy-spin
        console.log(`[Gemini] all ${geminiClients.length} keys cooling — waiting ${Math.round(wait / 1000)}s for next slot`);
        await new Promise(r => setTimeout(r, wait));
    }
}
function markClientCooldown(idx, ms = COOLDOWN_MS) {
    if (idx == null) return;
    _clientCooldownUntil[idx] = Date.now() + ms;
}
if (geminiClients.length > 0) {
    console.log(`[Gemini] mode: ${USE_VERTEX ? 'Vertex AI (express)' : 'AI Studio'} · ${geminiClients.length} key${geminiClients.length === 1 ? '' : 's'}`);
}
const openaiClient = process.env.OPENAI_API_KEY  ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY }) : null;

// Multi-brand support — drives brand-specific prompts, assets, captions, and
// overlay logos. The active brand flows in on req.body.brand / req.query.brand
// (default falls back to DEFAULT_BRAND).
const { BRANDS, DEFAULT_BRAND, resolveBrand, listBrands } = require('./brands');

// Which providers are available — only the two NanoBanana tiers are surfaced
// in the UI. OpenAI is wired but hidden — kept available as a fallback target.
const PROVIDERS = {};
if (geminiClient) PROVIDERS.gemini    = { label: 'Nano Banana Pro', tier: 'pro',  canGenerate: true, description: 'Gemini 3 Pro Image — thinking mode, highest fidelity' };
if (geminiClient) PROVIDERS.nanobana2 = { label: 'Nano Banana 2',   tier: 'fast', canGenerate: true, description: 'Gemini 3.1 Flash Image — fast and affordable' };
if (openaiClient) PROVIDERS.openai    = { label: 'OpenAI',          tier: 'hidden', canGenerate: true, hidden: true };

console.log('[Providers]', Object.keys(PROVIDERS).join(', ') || 'NONE — add API keys to .env');

// ── Cost tracking ─────────────────────────────────────────────────────────
// Per-resolution pricing (Gemini API — ai.google.dev/gemini-api/docs/pricing)
const COST_BY_SIZE = {
    gemini: {   // Nano Banana Pro (gemini-3-pro-image-preview)
        '1K': 0.134, '2K': 0.134, '4K': 0.240,
        default: 0.134,
    },
    nanobana2: { // Nano Banana 2 (gemini-3.1-flash-image-preview)
        '512': 0.034, '1K': 0.067, '2K': 0.101, '4K': 0.151,
        default: 0.067,
    },
    openai: {
        default: 0.133,  // GPT Image 1.5 High @ 1024x1024
    },
};

const usageStats = {
    session: {
        gemini:    { images: 0, cost: 0 },
        nanobana2: { images: 0, cost: 0 },
        openai:    { images: 0, cost: 0 },
        total:     { images: 0, cost: 0 },
    },
    history: [],  // last 50 entries
};

function trackUsage(provider, shotId, imageSize) {
    const providerCosts = COST_BY_SIZE[provider] || {};
    const cost = providerCosts[imageSize] || providerCosts.default || 0;
    if (!usageStats.session[provider]) usageStats.session[provider] = { images: 0, cost: 0 };
    usageStats.session[provider].images++;
    usageStats.session[provider].cost += cost;
    usageStats.session.total.images++;
    usageStats.session.total.cost += cost;

    const entry = {
        provider,
        shotId,
        cost,
        timestamp: Date.now(),
    };
    usageStats.history.unshift(entry);
    if (usageStats.history.length > 50) usageStats.history.length = 50;

    console.log(`[Cost] +$${cost.toFixed(3)} (${provider}/${shotId}) — session total: $${usageStats.session.total.cost.toFixed(3)} (${usageStats.session.total.images} images)`);
    return entry;
}

// ── Shot definitions ───────────────────────────────────────────────────────
// Each shot has an id, label, category, and a function that builds the prompt
const SHOT_CATALOG = {
    // ── Ecommerce ──
    ecom_hero: {
        id: 'ecom_hero',
        label: 'Hero / Front',
        category: 'ecommerce',
        description: 'Clean front-facing product shot on pure white',
    },
    ecom_angle: {
        id: 'ecom_angle',
        label: '45° Angle',
        category: 'ecommerce',
        description: 'Three-quarter angle showing depth and dimension',
    },
    ecom_detail: {
        id: 'ecom_detail',
        label: 'Detail Close-up',
        category: 'ecommerce',
        description: 'Extreme macro of the finest detail area',
    },
    ecom_flat: {
        id: 'ecom_flat',
        label: 'Flat Lay',
        category: 'ecommerce',
        description: 'Bird\'s eye flat lay on white surface',
    },
    ecom_stand: {
        id: 'ecom_stand',
        label: 'Display Stand',
        category: 'ecommerce',
        description: 'On a branded display stand with warm backdrop',
    },
    ecom_group: {
        id: 'ecom_group',
        label: 'Scale / Context',
        category: 'ecommerce',
        description: 'Jewelry next to a subtle size reference',
    },

    // ── Model ──
    model_wrist: {
        id: 'model_wrist',
        label: 'Wrist / Hand',
        category: 'model',
        description: 'Jewelry on wrist or hand, tight crop',
    },
    model_neck: {
        id: 'model_neck',
        label: 'Neck / Décolletage',
        category: 'model',
        description: 'Necklace on neck, collarbone framing',
    },
    model_ear: {
        id: 'model_ear',
        label: 'Ear Close-up',
        category: 'model',
        description: 'Earring on ear, jawline framing',
    },
    model_lifestyle: {
        id: 'model_lifestyle',
        label: 'Lifestyle',
        category: 'model',
        description: 'Model wearing jewelry in lifestyle context',
    },

    // ── Marble / Surface ──
    marble: {
        id: 'marble',
        label: 'Marble Surface',
        category: 'marble',
        description: 'Luxury marble surface with soft props',
    },
    marble_dark: {
        id: 'marble_dark',
        label: 'Dark Marble',
        category: 'marble',
        description: 'Moody dark marble with dramatic lighting',
    },

    // Brand-specific extraShots (e.g. Taheri's emerald-walnut signature) live
    // in brands.js and are merged into the catalog only when their brand is
    // active — see buildShotCatalog().
};

// ── Scene templates per shot ───────────────────────────────────────────────
// Tight scene cues. Each is 25–60 words: concrete visual hooks (background,
// framing, light direction, mood). Backgrounds are FLAT — no bokeh, no
// environmental scene from the reference image (the brand intro already tells
// the model "the reference's background/lighting is replaced").
//
// `${ECOM_STAND_BRAND_REF}` is a placeholder we substitute per-brand at lookup
// time — keeps this dictionary brand-agnostic so it can be consumed both by
// buildShotPrompt and by the /shot-scene endpoint.
// Skill-aligned scene blocks. Each starts with a strong verb (Place /
// Display / Capture) per nanobanana-jewelry/SKILL.md, embeds its own
// background, and avoids negative phrasing in the body — unwanted features
// go in the Avoid line assembled by buildShotPrompt.
const SCENE_TEMPLATES = {
    ecom_hero: `Place the piece as the hero subject on a flat pure white (#FFFFFF) seamless studio background. Centered, slight elevation showing the decorative face, piece alone in frame.`,

    ecom_angle: `Display the piece at a three-quarter angle on a flat pure white (#FFFFFF) seamless studio background. Camera 45° to the front, slightly elevated, revealing depth and side profile.`,

    ecom_detail: `Capture an extreme macro close-up of the piece's most intricate area — center stone, fine filigree, or detailed metalwork — on a flat neutral white background. Frame fills with the detail. Razor-sharp focus.`,

    ecom_flat: `Place the piece flat on a pure white surface for an overhead lay. Camera straight down, decorative face up. Bangles fully circular, necklaces in elegant drape, rings angled.`,

    ecom_stand: `Display the piece for \${ECOM_STAND_BRAND_REF} on a flat warm ivory studio background. Pick the right display from the piece type: bangle or cuff on a velvet cushion roll or half-cylinder (never hanging); ring on a slim velvet cone or small cushion; necklace draped over a fabric bust or laid on a velvet tray; earrings on a padded card or low T-bar; tikka flat on velvet. Display in matte cream or soft gold, minimal styling.`,

    ecom_group: `Place the piece on a flat pure white surface beside one subtle size reference — a rose petal, small velvet pouch, or hand mirror. Reference object in soft focus, secondary. Piece tack-sharp, the hero.`,

    model_wrist: `Worn on the wrist of a real woman, late 20s, warm South Asian skin, long slender editorial hands, naturally relaxed pose, clean nude manicure with realistic skin texture. Bangles snug with the decorative face to camera; rings on the ring finger. Arm extended forward, wrist level, fingers loosely curled and pointing down, back of hand to camera. Tight crop on hand, wrist, and forearm — no face. Flat soft cream backdrop.`,

    model_neck: `Worn on the neck of a real woman, late 20s, warm South Asian skin, elegant bone structure. Necklace sits naturally on the collarbone. Slight head tilt, chin slightly lifted. Solid simple top or bare shoulders, hair pulled back. Frame from mid-chest to just below the chin. Flat soft cream backdrop.`,

    model_ear: `Worn on the ear of a real woman, late 20s, warm South Asian skin, elegant jawline. Three-quarter profile presenting the ear; hair tucked back. Tight crop on ear, jawline, and a hint of neck. Flat soft cream backdrop, earring tack-sharp.`,

    model_lifestyle: `Worn by a real woman, late 20s, warm South Asian skin, in a natural candid pose — adjusting the piece, looking away, mid-movement. Simple solid clothing that doesn't compete. Frame medium, waist-up. Flat warm cream studio backdrop.`,

    marble: `Place the piece on a Carrara marble surface with soft grey veining, edge-to-edge. One or two muted props alongside (dried flowers, silk ribbon, or a small gold-rimmed dish). Soft warm window light from the left. Piece sharp, props soft.`,

    marble_dark: `Place the piece on dark emperador or nero marquina marble — deep brown-black with gold or white veining, edge-to-edge. Single focused light from above-right, high contrast, deep shadows. The metal of the piece glows against the dark surface.`,
};

// Reverse map: catalog preset display names → shotIds. Used for the
// backward-compat branch in buildShotPrompt and for the /catalog-name-to-id
// endpoint the UI uses to migrate stale library entries.
const CATALOG_NAME_TO_ID = {
    'Hero / Front': 'ecom_hero',
    '45° Angle': 'ecom_angle',
    'Detail Close-up': 'ecom_detail',
    'Flat Lay': 'ecom_flat',
    'Display Stand': 'ecom_stand',
    'Scale / Context': 'ecom_group',
    'Wrist / Hand': 'model_wrist',
    'Neck / Décolletage': 'model_neck',
    'Ear Close-up': 'model_ear',
    'Lifestyle': 'model_lifestyle',
    'Marble Surface': 'marble',
    'Dark Marble': 'marble_dark',
    'Taheri Signature': 'taheri_signature',
};

// Resolve a (shotId, brandId) to its rendered scene text. Brand extraShots
// (e.g. taheri_signature) override SCENE_TEMPLATES if their id matches.
function getSceneText(shotId, brandId = DEFAULT_BRAND) {
    const brand = resolveBrand(brandId);
    // Brand extras win
    for (const extra of brand.extraShots || []) {
        if (extra.id === shotId && extra.scenePrompt) return extra.scenePrompt;
    }
    let s = SCENE_TEMPLATES[shotId];
    if (!s) return null;
    // Substitute the brand's ecom-stand reference label
    return s.replace('${ECOM_STAND_BRAND_REF}', brand.ecomStandBrandRef || '');
}

// ── Skill master template (nanobanana-jewelry/SKILL.md) ───────────────────
// Universal opening: reference authority + fidelity. Skill says state this
// in the FIRST sentence. No metal-naming — naming silver/gold/platinum makes
// the model default to a generic version of that metal (skill rule #2).
const FIDELITY_OPENER = `Use the attached reference image(s) as the exact product. Preserve every detail faithfully — every stone, prong, metal tone, surface finish, and proportion must match the reference exactly. Do not redesign or reinterpret. Match the metal color in the reference precisely.`;
// Universal composition + camera defaults (skill master template).
const COMPOSITION_LINE = `Composition: centered hero, slight 15° angle, piece fills 55% of the frame. Editorial negative space.`;
const CAMERA_LINE = `Camera: 100mm macro lens, f/5.6, tack-sharp focus on the stone and setting, shallow background blur. Ultra-high detail, 4K, photorealistic, color-accurate.`;
// Universal Avoid baseline. Brand-specific avoid extras get appended.
// Negative phrasing only lives in the Avoid line — never in the prompt body
// (skill rule #4: anything mentioned gets rendered).
const AVOID_BASE = `redesigning the piece, generic ring archetypes (cathedral shoulder, tulip basket, donut gallery, peg-head, cone basket, smooth solitaire taper unless the reference shows them), extra stones or pieces not in the reference, decorative elements not in the reference, text, logos, watermarks`;

// ── Prompt builders per shot ───────────────────────────────────────────────
function buildShotPrompt(shotId, customInstruction, hasAnchor = false, customPrompt = null, brandId = DEFAULT_BRAND) {
    const brand = resolveBrand(brandId);

    // Anchor-consistency block — fires only when /generate's legacy anchor
    // pipeline is used (framework UI sends hasAnchor=false everywhere).
    const anchorBlock = hasAnchor
        ? 'CONSISTENCY ANCHOR: The LAST reference image is a previously-generated clean shot of this same piece — treat it as the visual ground truth and match every stone, prong, and proportion to it.'
        : '';

    // Backward-compat: old library entries persist as "[Catalog: <name>] …"
    // bodies. Map name → shotId, then drop customPrompt so we use the real
    // scene template below. (New flow stores catalogShotId on the entry and
    // routes via angleId; this branch only fires for stale localStorage.)
    let resolvedCustom = customPrompt;
    if (resolvedCustom && /^\s*\[Catalog:/.test(resolvedCustom)) {
        const m = resolvedCustom.match(/^\s*\[Catalog:\s*([^\]]+)\]/);
        if (m) {
            const realId = CATALOG_NAME_TO_ID[m[1].trim()];
            if (realId) {
                console.log(`[CatalogRescue] "${m[1].trim()}" → using shotId=${realId}, dropping customPrompt`);
                shotId = realId;
                resolvedCustom = null;
            }
        }
    }
    // Scene resolution: server-side template wins when no customPrompt;
    // otherwise the user's typed text becomes the scene.
    const sceneBody = resolvedCustom
        ? resolvedCustom
        : (getSceneText(shotId, brandId) || getSceneText('ecom_hero', brandId));

    // Skill master template ordering (nanobanana-jewelry/SKILL.md):
    //   1. Reference authority + fidelity (FIDELITY_OPENER)
    //   2. Brand voice (one short sentence)
    //   3. Anchor (rare, only when hasAnchor=true)
    //   4. Scene block — verb-led, embeds background
    //   5. Lighting (per brand)
    //   6. Composition (universal)
    //   7. Camera (universal)
    //   8. Mood (per brand)
    //   9. Additional direction (optional)
    //  10. Avoid (universal + brand-specific)
    const parts = [
        FIDELITY_OPENER,
        brand.voice ? `Brand: ${brand.voice}` : null,
        anchorBlock,
        sceneBody,
        brand.lighting ? `Lighting: ${brand.lighting}.` : null,
        COMPOSITION_LINE,
        CAMERA_LINE,
        brand.mood ? `Mood: ${brand.mood}.` : null,
        customInstruction ? `Additional direction: ${customInstruction}` : null,
        `Avoid: ${AVOID_BASE}${brand.avoid ? ', ' + brand.avoid : ''}.`,
    ].filter(Boolean);

    const rendered = parts.join('\n\n');
    // Diagnostic: log each distinct rendered prompt once so we can verify
    // every shotId/customPrompt path reaches Gemini cleanly. Hash by first
    // 200 chars to dedupe identical prompts within a run.
    global.__PROMPT_SEEN = global.__PROMPT_SEEN || new Set();
    const sig = rendered.slice(0, 200);
    if (!global.__PROMPT_SEEN.has(sig)) {
        global.__PROMPT_SEEN.add(sig);
        console.log(`\n[PROMPT-DIAG] ─── ${rendered.split(/\s+/).length} words ───`);
        console.log(rendered);
        console.log('[PROMPT-DIAG] ─── end ───\n');
    }
    return rendered;
}

// Build the aspect ratio / size instruction to embed in prompts. NB Pro
// (gemini-3-pro-image-preview) doesn't accept imageConfig on image-to-image
// calls — without this hint it returns square outputs even when 4:3 was
// requested. Always emit when an aspect is set, including 1:1 (so square
// shots still get an explicit framing cue rather than relying on the model's
// default which can drift). (Audit #4.)
function buildImageConfigPrompt(imageOpts) {
    if (!imageOpts || !imageOpts.aspectRatio) return '';
    const ar = imageOpts.aspectRatio;
    const orient = ar === '1:1' ? 'square'
                 : (parseFloat(ar.split(':')[0]) > parseFloat(ar.split(':')[1]) ? 'landscape' : 'portrait');
    return `\nIMAGE FORMAT: Output a ${ar} ${orient} frame — fill the whole canvas naturally; do not letterbox or pillarbox.`;
}

// Upscale image to target resolution if the API ignores imageSize
const SIZE_PIXELS = { '512': 512, '1K': 1024, '2K': 2048, '4K': 4096 };

async function upscaleIfNeeded(base64, targetSize, aspectRatio) {
    const targetPx = SIZE_PIXELS[targetSize];
    if (!targetPx || targetPx <= 1024) return base64; // 1K or below, no upscale needed

    const buf = Buffer.from(base64, 'base64');
    const meta = await sharp(buf).metadata();
    const maxDim = Math.max(meta.width, meta.height);

    if (maxDim >= targetPx * 0.9) return base64; // already close enough

    // Calculate target dimensions preserving aspect ratio
    let w, h;
    if (meta.width >= meta.height) {
        w = targetPx;
        h = Math.round(targetPx * (meta.height / meta.width));
    } else {
        h = targetPx;
        w = Math.round(targetPx * (meta.width / meta.height));
    }

    console.log(`[Upscale] ${meta.width}x${meta.height} → ${w}x${h} (target ${targetSize})`);
    const upscaled = await sharp(buf)
        .resize(w, h, { kernel: sharp.kernel.lanczos3 })
        .png()
        .toBuffer();
    return upscaled.toString('base64');
}

const app    = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } });

app.use(express.json({ limit: '100mb' }));
// Disable caching for HTML so the user always receives the latest UI after server restarts.
// Other assets (fonts/images) can still be cached normally.
app.use(express.static(path.join(__dirname, 'public'), {
    setHeaders: (res, filePath) => {
        if (filePath.endsWith('.html')) {
            res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0');
            res.setHeader('Pragma', 'no-cache');
            res.setHeader('Expires', '0');
        }
    },
}));

app.get('/health', (_req, res) => res.json({ status: 'ok' }));

// ── Apply overlay to existing images ─────────────────────────────────────────
app.post('/apply-overlay', upload.array('images', 50), async (req, res) => {
    try {
        const weightText = (req.body.weightText || '').trim();
        const brandId   = BRANDS[req.body.brand] ? req.body.brand : DEFAULT_BRAND;
        const files = req.files || [];
        if (files.length === 0) return res.status(400).json({ error: 'No images provided' });

        console.log(`[Overlay] brand=${brandId} applying to ${files.length} image(s) — weight: "${weightText || 'none'}"`);

        const results = [];
        for (const file of files) {
            const b64 = file.buffer.toString('base64');
            const overlaid = await applyOverlay(b64, weightText, brandId);
            const buf = Buffer.from(overlaid, 'base64');
            const meta = await sharp(buf).metadata();
            results.push({
                name: file.originalname.replace(/\.[^.]+$/, '') + '_overlay.png',
                data: overlaid,
                width: meta.width,
                height: meta.height,
            });
        }

        res.json({ success: true, results });
    } catch (err) {
        console.error('[Overlay Error]', err.message);
        res.status(500).json({ error: err.message });
    }
});

// ── Serve available providers + shot catalog to frontend ────────────────────
app.get('/providers', (_req, res) => {
    // Hide providers flagged hidden (e.g. OpenAI is fallback-only).
    const visible = {};
    for (const [k, v] of Object.entries(PROVIDERS)) if (!v.hidden) visible[k] = v;
    res.json(visible);
});
app.get('/shots', (req, res) => {
    const brandId = BRANDS[req.query.brand] ? req.query.brand : DEFAULT_BRAND;
    res.json(buildShotCatalog(brandId));
});
app.get('/brands', (_req, res) => res.json({ brands: listBrands(), defaultBrand: DEFAULT_BRAND }));

// Returns the rendered scene text for a (shotId, brand) pair — used by the
// Catalog Presets modal to load the ACTUAL scene template into the user's
// prompt library, so different catalog presets produce different outputs
// (otherwise customPrompt = boilerplate stub and the server can't tell them
// apart).
app.get('/shot-scene', (req, res) => {
    const shotId = (req.query.shotId || '').trim();
    const brandId = BRANDS[req.query.brand] ? req.query.brand : DEFAULT_BRAND;
    if (!shotId) return res.status(400).json({ error: 'shotId required' });
    const scene = getSceneText(shotId, brandId);
    if (!scene) return res.status(404).json({ error: 'Unknown shotId for this brand' });
    res.json({ shotId, brand: brandId, scene });
});

// Brand-aware shot catalog. Merges the framework's SHOT_CATALOG with any
// extraShots a brand declares (e.g. Taheri's taheri_signature emerald shot).
function buildShotCatalog(brandId = DEFAULT_BRAND) {
    const brand = resolveBrand(brandId);
    const merged = { ...SHOT_CATALOG };
    for (const extra of brand.extraShots || []) {
        // The scene prompt itself is materialized inside buildShotPrompt;
        // here we only register the catalog metadata so the shot is pickable.
        const { scenePrompt, ...meta } = extra;
        merged[extra.id] = meta;
    }
    return merged;
}

// Parse overlay options off a request body. Returns null when the brand
// doesn't support overlays (so callers can short-circuit cheaply).
function parseOverlayOpts(body, brandId) {
    const brand = resolveBrand(brandId);
    const cfg = brand.overlay;
    if (!cfg || !cfg.supported) return null;
    const raw = body && body.overlayEnabled;
    let enabled;
    if (raw === undefined || raw === null || raw === '') {
        enabled = !!cfg.defaultEnabled;
    } else {
        enabled = raw === true || raw === '1' || raw === 1 || raw === 'true';
    }
    const weightText = (body && typeof body.weightText === 'string') ? body.weightText : '';
    return { enabled, weightText };
}
app.get('/usage', (_req, res) => res.json(usageStats));
app.post('/usage/reset', (_req, res) => {
    for (const key of Object.keys(usageStats.session)) {
        usageStats.session[key] = { images: 0, cost: 0 };
    }
    usageStats.history = [];
    res.json({ reset: true });
});
app.get('/cost-rates', (_req, res) => res.json(COST_PER_IMAGE));

// ── Serve generated files from disk ─────────────────────────────────────────
app.get('/file', (req, res) => {
    const filePath = req.query.path;
    if (!filePath || !fs.existsSync(filePath)) return res.status(404).send('Not found');
    res.sendFile(path.resolve(filePath));
});

// ── Batch cancellation state ────────────────────────────────────────────────
let activeBatchId  = null;
let batchCancelled = false;

const cancelHandler = (req, res) => {
    if (activeBatchId) {
        batchCancelled = true;
        res.json({ cancelled: true });
    } else {
        res.json({ cancelled: false, message: 'No active batch.' });
    }
};
app.post('/cancel-batch', cancelHandler);
app.post('/batch/cancel', cancelHandler);

// ── Model capabilities endpoint ────────────────────────────────────────────
app.get('/model-capabilities', (_req, res) => {
    res.json({
        gemini: {
            model: 'gemini-3-pro-image-preview',
            label: 'Gemini 3 Pro',
            aspectRatios: ['1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9'],
            imageSizes: ['1K', '2K', '4K'],
            defaultAspectRatio: '1:1',
            defaultImageSize: '1K',
        },
        nanobana2: {
            model: 'gemini-3.1-flash-image-preview',
            label: 'Gemini 3.1 Flash',
            aspectRatios: ['1:1', '1:4', '1:8', '2:3', '3:2', '3:4', '4:1', '4:3', '4:5', '5:4', '8:1', '9:16', '16:9', '21:9'],
            imageSizes: ['512', '1K', '2K', '4K'],
            defaultAspectRatio: '1:1',
            defaultImageSize: '2K',
        },
        openai: {
            model: 'gpt-image-1.5',
            label: 'GPT Image 1.5',
            aspectRatios: ['1:1'],
            imageSizes: ['1024x1024'],
            defaultAspectRatio: '1:1',
            defaultImageSize: '1024x1024',
        },
    });
});

// ── Single product generation ───────────────────────────────────────────────
app.post('/generate', upload.array('images[]', 10), async (req, res) => {
    if (!req.files || req.files.length === 0) return res.status(400).json({ error: 'No images uploaded.' });

    const shotIds           = JSON.parse(req.body.shots || '[]');
    const customInstruction = (req.body.customInstruction || '').trim() || null;
    const customPrompt      = (req.body.customPrompt || '').trim() || null;
    const provider          = (req.body.provider || 'gemini').trim();
    const aspectRatios      = JSON.parse(req.body.aspectRatios || '["1:1"]');
    const imageSize         = (req.body.imageSize || '').trim() || null;
    const variationCount    = Math.min(Math.max(parseInt(req.body.variationCount) || 1, 1), 5);
    const overlayEnabled    = req.body.overlayEnabled === 'true';
    const weightText        = (req.body.weightText || '').trim();
    const overlayOpts       = { enabled: overlayEnabled, weightText };
    const brandId           = BRANDS[req.body.brand] ? req.body.brand : DEFAULT_BRAND;
    const shotCatalog       = buildShotCatalog(brandId);

    if (shotIds.length === 0) return res.status(400).json({ error: 'No shots selected.' });

    const imageInputs = await Promise.all(req.files.map(async (f) => {
        const buf = await toJpeg(f.originalname || '', f.buffer);
        return { base64: buf.toString('base64'), mimeType: 'image/jpeg' };
    }));

    try {
        const totalImages = shotIds.length * aspectRatios.length * variationCount;
        console.log(`[Generate] brand=${brandId} ${shotIds.length} shot(s) × ${aspectRatios.length} ratio(s) × ${variationCount} var(s) = ${totalImages} images via ${provider}${overlayEnabled ? ` [overlay: ${weightText || 'logo only'}]` : ''}`);

        // ── Anchor-first consistency pipeline ──
        const results = [];
        let anchorRef = null;

        const anchorId = shotIds.includes('ecom_hero') ? 'ecom_hero'
            : shotIds.find(id => id.startsWith('ecom_'))
            || shotIds[0];

        // Generate anchor shot once (first aspect ratio, variation 1)
        console.log(`[Anchor] Generating ${anchorId} as consistency anchor via ${provider}...`);
        const anchorShot = shotCatalog[anchorId];
        const anchorAR = aspectRatios[0];
        const anchorData = await generateShot(anchorId, imageInputs, customInstruction, false, provider, { aspectRatio: anchorAR, imageSize }, overlayOpts, customPrompt, brandId);
        // Store clean (no-overlay) version as anchor reference for consistency
        const anchorClean = await generateWithGemini ? anchorData : anchorData; // already generated
        anchorRef = { base64: anchorData, mimeType: 'image/png' };
        results.push({ id: anchorId, label: anchorShot.label, category: anchorShot.category, data: anchorData, aspectRatio: anchorAR, variation: 1 });

        // Build all remaining tasks: (shot × ratio × variation) minus the anchor we already did
        const tasks = [];
        for (const shotId of shotIds) {
            const shot = shotCatalog[shotId];
            if (!shot) continue;
            for (const ar of aspectRatios) {
                for (let v = 1; v <= variationCount; v++) {
                    // Skip the anchor we already generated
                    if (shotId === anchorId && ar === anchorAR && v === 1) continue;
                    tasks.push({ shotId, shot, ar, v, isAnchorShot: shotId === anchorId });
                }
            }
        }

        if (tasks.length > 0) {
            const refsWithAnchor = [...imageInputs, anchorRef];
            const parallel = await Promise.all(tasks.map(async ({ shotId, shot, ar, v, isAnchorShot }) => {
                const refs = isAnchorShot ? imageInputs : refsWithAnchor;
                const hasAnchor = !isAnchorShot;
                const data = await generateShot(shotId, refs, customInstruction, hasAnchor, provider, { aspectRatio: ar, imageSize }, overlayOpts, customPrompt, brandId);
                return { id: shotId, label: shot.label, category: shot.category, data, aspectRatio: ar, variation: v };
            }));
            results.push(...parallel.filter(Boolean));
        }

        // Sort: by shot order, then aspect ratio order, then variation
        const ordered = [];
        for (const shotId of shotIds) {
            for (const ar of aspectRatios) {
                for (let v = 1; v <= variationCount; v++) {
                    const match = results.find(r => r.id === shotId && r.aspectRatio === ar && r.variation === v);
                    if (match) ordered.push(match);
                }
            }
        }

        res.json({ success: true, results: { shots: ordered }, usage: usageStats });
    } catch (err) {
        console.error('[Generate Error]', err?.message || err);
        const safetyBlocked = err?.message?.toLowerCase().includes('safety');
        res.status(500).json({
            error: safetyBlocked
                ? 'Image blocked by safety filters — try a different photo.'
                : err.message || 'Generation failed.',
        });
    }
});

// ── Retry single shot ───────────────────────────────────────────────────────
app.post('/generate-angle', upload.array('images[]', 10), async (req, res) => {
    if (!req.files || req.files.length === 0) return res.status(400).json({ error: 'No images uploaded.' });

    const shotId            = req.body.angleId;
    const customInstruction = (req.body.customInstruction || '').trim() || null;
    const customPrompt      = (req.body.customPrompt || '').trim() || null;
    const provider          = (req.body.provider || 'gemini').trim();
    const aspectRatio       = (req.body.aspectRatio || '').trim() || null;
    const imageSize         = (req.body.imageSize || '').trim() || null;
    const overlayEnabled    = req.body.overlayEnabled === 'true';
    const weightText        = (req.body.weightText || '').trim();
    const overlayOpts       = { enabled: overlayEnabled, weightText };
    const brandId           = BRANDS[req.body.brand] ? req.body.brand : DEFAULT_BRAND;

    const imageInputs = await Promise.all(req.files.map(async (f) => {
        const buf = await toJpeg(f.originalname || '', f.buffer);
        return { base64: buf.toString('base64'), mimeType: 'image/jpeg' };
    }));

    try {
        const shot = buildShotCatalog(brandId)[shotId];
        if (!shot) return res.status(400).json({ error: 'Unknown shot type.' });

        const imageData = await generateShot(shotId, imageInputs, customInstruction, false, provider, { aspectRatio, imageSize }, overlayOpts, customPrompt, brandId);
        res.json({ success: true, imageData, usage: usageStats });
    } catch (err) {
        console.error('[Retry Error]', err?.message || err);
        res.status(500).json({ error: err.message || 'Generation failed.' });
    }
});

// ── Batch folder endpoint (SSE) ─────────────────────────────────────────────
app.get('/batch', async (req, res) => {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    const send = (payload) => res.write(`data: ${JSON.stringify(payload)}\n\n`);

    const folderPath        = (req.query.folderPath || '').trim().replace(/^['"]|['"]$/g, '');
    const customInstruction = (req.query.customInstruction || '').trim() || null;
    const customPrompt      = (req.query.customPrompt || '').trim() || null;
    const shotIds           = JSON.parse(req.query.shots || '[]');
    const provider          = (req.query.provider || 'gemini').trim();
    const aspectRatios      = JSON.parse(req.query.aspectRatios || '["1:1"]');
    const imageSize         = (req.query.imageSize || '').trim() || null;
    const variationCount    = Math.min(Math.max(parseInt(req.query.variationCount) || 1, 1), 5);
    const brandId           = BRANDS[req.query.brand] ? req.query.brand : DEFAULT_BRAND;
    const shotCatalog       = buildShotCatalog(brandId);

    if (!folderPath) { send({ type: 'error', message: 'No folder path provided.' }); return res.end(); }
    if (!fs.existsSync(folderPath)) { send({ type: 'error', message: `Folder not found: ${folderPath}` }); return res.end(); }
    if (!fs.statSync(folderPath).isDirectory()) { send({ type: 'error', message: 'That path is a file, not a folder.' }); return res.end(); }
    if (shotIds.length === 0) { send({ type: 'error', message: 'No shots selected.' }); return res.end(); }

    const productDirs = fs.readdirSync(folderPath, { withFileTypes: true })
        .filter(d => d.isDirectory() && !d.name.startsWith('.') && d.name !== 'ecommerce' && d.name !== 'output')
        .map(d => ({ name: d.name, fullPath: path.join(folderPath, d.name) }));

    if (productDirs.length === 0) {
        send({ type: 'error', message: 'No product subfolders found.' });
        return res.end();
    }

    activeBatchId = Date.now().toString();
    batchCancelled = false;

    send({ type: 'start', total: productDirs.length, batchId: activeBatchId, shots: shotIds });

    for (const { name: productName, fullPath: productFolder } of productDirs) {
        if (batchCancelled) {
            send({ type: 'cancelled', message: 'Batch cancelled by user.' });
            break;
        }

        send({ type: 'product_start', product: productName, productFolder });

        try {
            const IMAGE_EXTS = /\.(jpe?g|png|webp|gif|heic|heif)$/i;
            const imageFiles = fs.readdirSync(productFolder)
                .filter(f => IMAGE_EXTS.test(f) && !f.startsWith('.'))
                .map(f => path.join(productFolder, f));

            if (imageFiles.length === 0) {
                send({ type: 'product_error', product: productName, message: 'No images found in folder.' });
                continue;
            }

            const imageInputs = await Promise.all(imageFiles.map(async (fp) => {
                const buf = await toJpeg(fp, fs.readFileSync(fp));
                return { base64: buf.toString('base64'), mimeType: 'image/jpeg' };
            }));

            const outDir = path.join(folderPath, 'output', productName);
            fs.mkdirSync(outDir, { recursive: true });

            // ── Anchor-first consistency pipeline ──
            const anchorId = shotIds.includes('ecom_hero') ? 'ecom_hero'
                : shotIds.find(id => id.startsWith('ecom_'))
                || shotIds[0];

            let anchorRef = null;
            const anchorShot = shotCatalog[anchorId];
            const anchorAR = aspectRatios[0];

            send({ type: 'angle_start', product: productName, angle: anchorId, label: `${anchorShot.label} (anchor)` });
            try {
                const b64 = await generateShot(anchorId, imageInputs, customInstruction, false, provider, { aspectRatio: anchorAR, imageSize }, {}, customPrompt, brandId);
                anchorRef = { base64: b64, mimeType: 'image/png' };
                const outPath = path.join(outDir, `${anchorId}_${anchorAR.replace(':', 'x')}_v1.png`);
                fs.writeFileSync(outPath, Buffer.from(b64, 'base64'));
                send({ type: 'angle_done', product: productName, angle: anchorId, label: `${anchorShot.label} · ${anchorAR}`, savedTo: outPath });
                send({ type: 'usage', usage: usageStats });
            } catch (err) {
                send({ type: 'angle_error', product: productName, angle: anchorId, message: err.message });
            }

            if (batchCancelled) {
                send({ type: 'product_done', product: productName });
                send({ type: 'cancelled', message: 'Batch cancelled by user.' });
                break;
            }

            // Build all remaining tasks
            const batchTasks = [];
            for (const shotId of shotIds) {
                const shot = shotCatalog[shotId];
                if (!shot) continue;
                for (const ar of aspectRatios) {
                    for (let v = 1; v <= variationCount; v++) {
                        if (shotId === anchorId && ar === anchorAR && v === 1) continue;
                        batchTasks.push({ shotId, shot, ar, v, isAnchorShot: shotId === anchorId });
                    }
                }
            }

            const refsWithAnchor = anchorRef ? [...imageInputs, anchorRef] : imageInputs;

            for (const t of batchTasks) {
                send({ type: 'angle_start', product: productName, angle: t.shotId, label: `${t.shot.label} · ${t.ar}${t.v > 1 ? ` #${t.v}` : ''}` });
            }

            const parallelTasks = batchTasks.map(({ shotId, shot, ar, v, isAnchorShot }) => {
                const refs = isAnchorShot ? imageInputs : refsWithAnchor;
                const hasAnchor = !isAnchorShot;
                return generateShot(shotId, refs, customInstruction, hasAnchor, provider, { aspectRatio: ar, imageSize }, {}, customPrompt, brandId)
                    .then(b64 => {
                        const suffix = variationCount > 1 ? `_v${v}` : '';
                        const p = path.join(outDir, `${shotId}_${ar.replace(':', 'x')}${suffix}.png`);
                        fs.writeFileSync(p, Buffer.from(b64, 'base64'));
                        send({ type: 'angle_done', product: productName, angle: shotId, label: `${shot.label} · ${ar}${v > 1 ? ` #${v}` : ''}`, savedTo: p });
                        send({ type: 'usage', usage: usageStats });
                    })
                    .catch(err => send({ type: 'angle_error', product: productName, angle: shotId, message: err.message }));
            });

            await Promise.all(parallelTasks);
        } catch (err) {
            console.error(`[Batch] ${productName}:`, err.message);
            send({ type: 'product_error', product: productName, message: err.message });
        }

        send({ type: 'product_done', product: productName });

        if (batchCancelled) {
            send({ type: 'cancelled', message: 'Batch cancelled by user.' });
            break;
        }
    }

    const wasCancelled = batchCancelled;
    activeBatchId = null;
    batchCancelled = false;

    if (!wasCancelled) send({ type: 'done' });
    res.end();
});

// ── Batch retry single shot ─────────────────────────────────────────────────
app.post('/retry-angle', upload.none(), async (req, res) => {
    const { productFolder, angleId, provider: retryProvider, aspectRatio: retryAR, imageSize: retryIS, brand: rawBrand, customPrompt: retryCustom, customInstruction: retryInst } = req.body;
    const provider = (retryProvider || 'gemini').trim();
    const aspectRatio = (retryAR || '').trim() || null;
    const imageSize = (retryIS || '').trim() || null;
    const brandId = BRANDS[rawBrand] ? rawBrand : DEFAULT_BRAND;
    // Honor the prompt + overlay the user had selected for the original
    // batch — without these, retries reset to the canned shot scene and
    // strip overlays. (Audit #1.)
    const customPrompt = (retryCustom || '').trim() || null;
    const customInstruction = (retryInst || '').trim() || null;
    const overlayOpts = parseOverlayOpts(req.body, brandId) || { enabled: false, weightText: '' };
    if (!productFolder || !angleId) return res.status(400).json({ error: 'Missing productFolder or angleId.' });

    const shot = buildShotCatalog(brandId)[angleId];
    if (!shot) return res.status(400).json({ error: 'Unknown shot type.' });

    const IMAGE_EXTS = /\.(jpe?g|png|webp|gif|heic|heif)$/i;
    const imageFiles = fs.readdirSync(productFolder)
        .filter(f => IMAGE_EXTS.test(f) && !f.startsWith('.'))
        .map(f => path.join(productFolder, f));

    if (imageFiles.length === 0) return res.status(400).json({ error: 'No source images in product folder.' });

    try {
        const imageInputs = await Promise.all(imageFiles.map(async (fp) => {
            const buf = await toJpeg(fp, fs.readFileSync(fp));
            return { base64: buf.toString('base64'), mimeType: 'image/jpeg' };
        }));

        const raw = await generateShot(angleId, imageInputs, customInstruction, false, provider, { aspectRatio, imageSize }, overlayOpts, customPrompt, brandId);
        const outPath = path.join(productFolder, '..', 'output', path.basename(productFolder), `${angleId}.png`);
        fs.mkdirSync(path.dirname(outPath), { recursive: true });
        fs.writeFileSync(outPath, Buffer.from(raw, 'base64'));
        res.json({ success: true, base64: raw, usage: usageStats });
    } catch (err) {
        console.error('[Retry Error]', err?.message || err);
        res.status(500).json({ error: err.message || 'Retry failed.' });
    }
});

// ── WhatsApp caption generation ────────────────────────────────────────────
app.post('/generate-caption', upload.array('images[]', 10), async (req, res) => {
    const productName   = (req.body.productName || '').trim() || 'this piece';
    const extraContext  = (req.body.extraContext || '').trim();
    const brandId       = BRANDS[req.body.brand] ? req.body.brand : DEFAULT_BRAND;
    const brand         = resolveBrand(brandId);

    // Build image inputs from uploaded files (if any)
    let imageInputs = [];
    if (req.files && req.files.length > 0) {
        imageInputs = await Promise.all(req.files.map(async (f) => {
            const buf = await toJpeg(f.originalname || '', f.buffer);
            return { base64: buf.toString('base64'), mimeType: 'image/jpeg' };
        }));
    }

    // Also accept base64 images from JSON body (for generated images)
    const jsonImages = req.body.captionImages ? JSON.parse(req.body.captionImages) : [];
    for (const img of jsonImages) {
        if (img.b64) imageInputs.push({ base64: img.b64, mimeType: 'image/png' });
    }

    const captionPrompt = `${brand.captionSystem}

BRAND VOICE & STYLE:
- Sophisticated yet accessible, warm, elegant, aspirational
- Short punchy sentences. Conversational but elevated
- Open with a hook that creates desire (e.g. "Meet your new obsession.", "Some pieces just speak for themselves.", "This one's going to turn heads.")
- Highlight the key visual feature of THIS specific piece (describe what you actually see in the image — the stone color, the design style, the sparkle)
- Use WhatsApp bold formatting with asterisks for key specs: *925 Sterling Silver*, *Gold Plated*, etc.
- End with a soft-launch / urgency line, then the standard CTA

DEFAULT MATERIAL SPECS (use these unless the image clearly shows otherwise or extra context overrides):
- 925 Sterling Silver with White Rhodium / Gold Plating
- Cubic Zirconia stones
- Simulated coloured stones (e.g. *simulated emeralds*, *simulated rubies*) — NOT certified/natural unless specified

WHATSAPP FORMATTING RULES:
- Use *asterisks* for bold (key specs, brand name)
- Use _underscores_ for italic (rare, only for emphasis)
- Line breaks between sections (hook / description / CTA)
- Emojis only at the CTA section at the end
- Keep the whole caption under 500 characters

SAMPLE FOR REFERENCE (match this energy and structure):
"Meet your new obsession. *A certified yellow sapphire. Brilliant zircon accents. 925 sterling silver*. A combination this stunning doesn't come along often — and at ${brand.captionBrandMention}, it's entirely yours. We're celebrating our soft launch with special introductory pricing. These pieces won't wait forever. 📩 DM to order 🇵🇰 Nationwide Delivery"

CTA BLOCK (always end with this exact block):
📩 DM to order
🇵🇰 Nationwide Delivery

${extraContext ? `EXTRA CONTEXT FROM THE USER: ${extraContext}\n` : ''}
Now look at the jewelry image(s) provided and write ONE WhatsApp community caption for ${productName}. Output ONLY the caption text, nothing else — no quotes, no explanation, no markdown code blocks.`;

    try {
        let captionText;

        if (geminiClient && imageInputs.length > 0) {
            const parts = [
                { text: captionPrompt },
                ...imageInputs.map(img => ({ inlineData: { mimeType: img.mimeType, data: img.base64 } })),
            ];
            await acquireGeminiSlot();
            try {
                const _capPick = await pickGeminiClient();
                const response = await withTimeout(
                    (_capPick?.client || geminiClient).models.generateContent({
                        model: 'gemini-3-flash-preview',
                        contents: [{ role: 'user', parts }],
                    }),
                    30000,
                    'caption'
                );
                const resParts = response.candidates?.[0]?.content?.parts || [];
                captionText = resParts.map(p => p.text).filter(Boolean).join('').trim();
            } finally {
                releaseGeminiSlot();
            }
        } else if (geminiClient) {
            // Text-only (no images)
            await acquireGeminiSlot();
            try {
                const _capPick = await pickGeminiClient();
                const response = await withTimeout(
                    (_capPick?.client || geminiClient).models.generateContent({
                        model: 'gemini-3-flash-preview',
                        contents: [{ role: 'user', parts: [{ text: captionPrompt }] }],
                    }),
                    20000,
                    'caption'
                );
                const resParts = response.candidates?.[0]?.content?.parts || [];
                captionText = resParts.map(p => p.text).filter(Boolean).join('').trim();
            } finally {
                releaseGeminiSlot();
            }
        } else {
            return res.status(500).json({ error: 'No AI provider available for caption generation.' });
        }

        // Clean up: remove wrapping quotes or code blocks if the model added them
        captionText = captionText.replace(/^["'`]+|["'`]+$/g, '').replace(/^```[\s\S]*?\n/, '').replace(/\n```$/, '').trim();

        res.json({ success: true, caption: captionText });
    } catch (err) {
        console.error('[Caption Error]', err?.message || err);
        res.status(500).json({ error: err.message || 'Caption generation failed.' });
    }
});

// ── AI prompt naming endpoint ───────────────────────────────────────────────
// Takes a prompt body, returns a short editorial label suitable as the saved-prompt name.
app.post('/name-prompt', async (req, res) => {
    const body = (req.body?.body || '').toString().trim();
    if (!body) return res.status(400).json({ error: 'Prompt body is required.' });
    if (!geminiClient) return res.status(500).json({ error: 'No AI provider available.' });

    const namingPrompt = `You are naming a saved prompt in a luxury jewelry photography pipeline. The user has written a creative direction / photographic prompt, and you must produce ONE short editorial label for it — the kind of name you'd see in a moodboard or shot list.

REQUIREMENTS:
- 2 to 5 words maximum
- Title Case (capitalise each significant word)
- No quotes, no emojis, no punctuation (no periods, commas, dashes, ellipsis)
- Evocative and specific to THIS prompt — capture the mood, setting, lighting, or core visual idea
- Never generic ("Nice Shot", "Jewelry Photo", "Beautiful Image")
- Never copy verbatim phrases from the prompt — distill the essence

EXAMPLES OF GOOD NAMES:
- "Golden Hour Intimacy"
- "Velvet Atelier"
- "Cinematic Close Crop"
- "Marble Pedestal Study"
- "Soft Dawn Portrait"

THE PROMPT TO NAME:
"""
${body}
"""

Output ONLY the name — nothing else. No explanation. No quotes. Just the words.`;

    const t0 = Date.now();
    console.log('[Name Prompt] Start, body length:', body.length);
    try {
        await acquireGeminiSlot();
        let nameText;
        try {
            const response = await withTimeout(
                geminiClient.models.generateContent({
                    model: 'gemini-3-flash-preview',
                    contents: [{ role: 'user', parts: [{ text: namingPrompt }] }],
                    config: { thinkingConfig: { thinkingLevel: 'MINIMAL' } },
                }),
                10000,
                'name-prompt'
            );
            const parts = response.candidates?.[0]?.content?.parts || [];
            nameText = parts.map(p => p.text).filter(Boolean).join('').trim();
        } finally {
            releaseGeminiSlot();
        }

        // Strip quotes, markdown, trailing punctuation, and cap the length
        nameText = (nameText || '')
            .replace(/^["'`*_]+|["'`*_.,!?…\-]+$/g, '')
            .replace(/^```[\s\S]*?\n/, '')
            .replace(/\n```$/, '')
            .replace(/\s+/g, ' ')
            .trim();

        if (!nameText) {
            console.warn('[Name Prompt] Empty result after', Date.now() - t0, 'ms');
            return res.status(500).json({ error: 'AI returned an empty name.' });
        }
        if (nameText.length > 60) nameText = nameText.slice(0, 57).trimEnd() + '…';

        console.log('[Name Prompt] OK in', Date.now() - t0, 'ms ->', nameText);
        res.json({ success: true, name: nameText });
    } catch (err) {
        console.error('[Name Prompt Error] after', Date.now() - t0, 'ms:', err?.message || err);
        res.status(500).json({ error: err.message || 'Name generation failed.' });
    }
});

// ── WhatsApp community post endpoint ────────────────────────────────────────
// Takes a generated jewelry image (base64) and optional weight string.
// Returns a copy-paste-ready WhatsApp post following the Taheri format.
app.post('/whatsapp-post', async (req, res) => {
    const imageB64 = (req.body?.imageB64 || '').toString();
    const mimeType = (req.body?.mimeType || 'image/png').toString();
    const weight = (req.body?.weight || '').toString().trim();

    if (!imageB64) return res.status(400).json({ error: 'Image data is required.' });
    if (!geminiClient) return res.status(500).json({ error: 'No AI provider available.' });

    const userSpecsBlock = weight
        ? `\n\nUSER-PROVIDED SPECS (these override visual guesses):\n- Weight: ${weight}`
        : '';

    const postPrompt = `You are a high-end jewelry branding expert writing a WhatsApp community post for Taheri (taheri.shop). Analyze the attached jewelry image and produce ONE elegant, copy-paste-ready WhatsApp post.

ABSOLUTE RULES:
- Output ONLY the final formatted post. No greetings, no explanations, no preamble, no sign-off.
- Response must be 100% ready to copy and paste — no surrounding code fences, no quotes around it.
- Never use generic names like "gold ring" or "diamond bracelet" — invent a unique poetic name.

STEP 1 — Analyze the image:
- Item type: ring, bangle, bracelet, earring, pendant, necklace, set, etc.
- Metal: Yellow / Rose / White gold, with purity if visible (18K, 21K, 22K, 24K). If unclear, write the colour only without a purity figure.
- Stones: ruby, emerald, zircon, pearl, diamond, CZ — or "No Stones / Pure Gold" if none.
- Design vibe: minimalist, ornate, vintage, geometric, floral, Arabic-inspired, bridal, statement, everyday.
- Occasion suitability: wedding, casual, formal, festive, gifting, everyday.

STEP 2 — Invent a poetic name. Examples of the right register:
- The Celestial Arc (curved bangle)
- The Gilded Reverie (delicate gold ring)
- The Ember Bloom (ruby floral pendant)
- The Quiet Storm (bold geometric bracelet)
Never plain descriptive names. Evoke emotion or imagery.

STEP 3 — Write the post in EXACTLY this format (WhatsApp markdown, asterisks for bold, underscores for italic):

✨ *[Unique Poetic Name] [Item Type]* — _[Metal & Purity] | [Weight]_

[One hook sentence — elegant, punchy, evocative. Captures the design vibe in one breath.]

✦ *Stone:* [Specific stone(s) OR "No Stones / Pure Gold"]
✦ *Where to wear:* [Short occasion detail — 5 to 10 words]
✦ *Style:* [Short aesthetic detail — 5 to 10 words]

*Shop Now:*
💬 WhatsApp: +923352275553, +923262275554
📍 Visit: Najmi Market, Shop #40 & #16
🌐 Browse all designs at *taheri.shop*

FORMATTING RULES:
- Header line: name+type in *bold*, metal/weight in _italic_.
- Bullet labels (Stone, Where to wear, Style) in *bold*.
- Do NOT add extra bullets or sections.
- Do NOT repeat metal or weight inside the bullets — they live in the header.
- Bullet values: short punchy phrases, not full sentences.
- Hook sentence: complete and standalone, poetic but not overwrought.

EDGE CASES:
- If weight is not provided, write "Weight on request" in the header.
- If the image shows multiple items as a set, name it as a set (e.g. *The Dusk Duet Set*) and list all stones together.
- If purity is unclear, write the metal colour only (e.g. "Yellow Gold").${userSpecsBlock}

Output ONLY the formatted post. Begin with the ✨ line.`;

    const t0 = Date.now();
    console.log('[WhatsApp Post] Start, weight:', weight || '(none)');

    try {
        await acquireGeminiSlot();
        let postText;
        try {
            const response = await withTimeout(
                geminiClient.models.generateContent({
                    model: 'gemini-3-flash-preview',
                    contents: [{
                        role: 'user',
                        parts: [
                            { text: postPrompt },
                            { inlineData: { mimeType, data: imageB64 } },
                        ],
                    }],
                }),
                30000,
                'whatsapp-post'
            );
            const parts = response.candidates?.[0]?.content?.parts || [];
            postText = parts.map(p => p.text).filter(Boolean).join('').trim();
        } finally {
            releaseGeminiSlot();
        }

        // Strip wrapping code fences and stray quotes if the model added them
        postText = (postText || '')
            .replace(/^```[a-z]*\s*\n?/i, '')
            .replace(/\n?```\s*$/, '')
            .trim();

        if (!postText) {
            console.warn('[WhatsApp Post] Empty result after', Date.now() - t0, 'ms');
            return res.status(500).json({ error: 'AI returned an empty post.' });
        }

        console.log('[WhatsApp Post] OK in', Date.now() - t0, 'ms, length:', postText.length);
        res.json({ success: true, post: postText });
    } catch (err) {
        console.error('[WhatsApp Post Error] after', Date.now() - t0, 'ms:', err?.message || err);
        res.status(500).json({ error: err.message || 'WhatsApp post generation failed.' });
    }
});

// ── Download ZIP endpoint ───────────────────────────────────────────────────
app.post('/download-zip', async (req, res) => {
    const { images } = req.body;
    if (!images || !Array.isArray(images) || images.length === 0) return res.status(400).json({ error: 'No images.' });

    const entries = images.map((img, i) => ({
        name: img.name || `image-${i + 1}.png`,
        data: Buffer.from(img.data, 'base64'),
    }));

    const zipBuf = buildZip(entries);
    res.setHeader('Content-Type', 'application/zip');
    res.setHeader('Content-Disposition', 'attachment; filename="taheri-shots.zip"');
    res.send(zipBuf);
});

function buildZip(entries) {
    const localHeaders = [];
    const centralHeaders = [];
    let offset = 0;

    for (const { name, data } of entries) {
        const nameBuf = Buffer.from(name, 'utf8');
        const lh = Buffer.alloc(30);
        lh.writeUInt32LE(0x04034b50, 0);
        lh.writeUInt16LE(20, 4);
        lh.writeUInt16LE(0, 8);
        lh.writeUInt32LE(data.length, 18);
        lh.writeUInt32LE(data.length, 22);
        lh.writeUInt16LE(nameBuf.length, 26);
        localHeaders.push(Buffer.concat([lh, nameBuf, data]));

        const ch = Buffer.alloc(46);
        ch.writeUInt32LE(0x02014b50, 0);
        ch.writeUInt16LE(20, 4);
        ch.writeUInt16LE(20, 6);
        ch.writeUInt32LE(data.length, 20);
        ch.writeUInt32LE(data.length, 24);
        ch.writeUInt16LE(nameBuf.length, 28);
        ch.writeUInt32LE(offset, 42);
        centralHeaders.push(Buffer.concat([ch, nameBuf]));

        offset += 30 + nameBuf.length + data.length;
    }

    const centralBuf = Buffer.concat(centralHeaders);
    const eocd = Buffer.alloc(22);
    eocd.writeUInt32LE(0x06054b50, 0);
    eocd.writeUInt16LE(entries.length, 8);
    eocd.writeUInt16LE(entries.length, 10);
    eocd.writeUInt32LE(centralBuf.length, 12);
    eocd.writeUInt32LE(offset, 16);

    return Buffer.concat([...localHeaders, centralBuf, eocd]);
}

// ── Universal shot generator (multi-provider) ───────────────────────────────
async function generateShot(shotId, imageInputs, customInstruction, hasAnchor = false, provider = 'gemini', imageOpts = {}, overlayOpts = {}, customPrompt = null, brandId = DEFAULT_BRAND) {
    const prompt = buildShotPrompt(shotId, customInstruction, hasAnchor, customPrompt, brandId) + buildImageConfigPrompt(imageOpts);

    let result;
    if (provider === 'openai') {
        result = await generateWithOpenAI(prompt, imageInputs, imageOpts);
    } else if (provider === 'nanobana2') {
        result = await generateWithNanoBana2(prompt, imageInputs, imageOpts);
    } else {
        result = await generateWithGemini(prompt, imageInputs, imageOpts);
    }

    trackUsage(provider, shotId, imageOpts.imageSize);

    // Apply overlay if enabled — brand-aware logo + weight composite.
    if (overlayOpts.enabled) {
        result = await applyOverlay(result, overlayOpts.weightText || '', brandId);
    }

    // Log final dimensions
    const finalBuf = Buffer.from(result, 'base64');
    const finalMeta = await sharp(finalBuf).metadata();
    console.log(`[Final] ${shotId}: ${finalMeta.width}x${finalMeta.height} (${(finalBuf.length / 1024 / 1024).toFixed(1)}MB)`);

    return result;
}

async function generateWithGemini(prompt, imageInputs, imageOpts = {}) {
    const parts = [
        { text: prompt },
        ...imageInputs.map(img => ({ inlineData: { mimeType: img.mimeType, data: img.base64 } })),
    ];
    let raw = await callGemini(parts, 0, imageOpts);
    // Only force square if no aspect ratio specified
    if (!imageOpts.aspectRatio || imageOpts.aspectRatio === '1:1') {
        raw = await makeSquareBase64(raw);
    }
    // Upscale if API ignored imageSize
    if (imageOpts.imageSize) raw = await upscaleIfNeeded(raw, imageOpts.imageSize, imageOpts.aspectRatio);
    return raw;
}

async function generateWithOpenAI(prompt, imageInputs) {
    await acquireGeminiSlot(); // reuse the same concurrency limiter
    try {
        // Use gpt-image-1.5 via the Images API with reference images
        const imageFiles = imageInputs.map((img, i) => {
            const buf = Buffer.from(img.base64, 'base64');
            return new File([buf], `ref_${i}.png`, { type: img.mimeType });
        });

        console.log(`[OpenAI] calling gpt-image-1.5... (${imageFiles.length} reference image(s))`);

        const response = await openaiClient.images.edit({
            model: 'gpt-image-1.5',
            image: imageFiles,
            prompt: prompt,
            n: 1,
            size: '1024x1024',
            quality: 'high',
        });

        const b64 = response.data?.[0]?.b64_json;
        if (!b64) {
            throw new Error('OpenAI returned no image data');
        }

        // Validate
        const buf = Buffer.from(b64, 'base64');
        const meta = await sharp(buf).metadata();
        if (!meta.width || !meta.height) throw new Error('OpenAI returned invalid image');

        console.log('[OpenAI] image OK');
        return makeSquareBase64(b64);
    } finally {
        releaseGeminiSlot();
    }
}

async function generateWithNanoBana2(prompt, imageInputs, imageOpts = {}) {
    const parts = [
        { text: prompt },
        ...imageInputs.map(img => ({ inlineData: { mimeType: img.mimeType, data: img.base64 } })),
    ];
    let raw = await callNanoBana2(parts, 0, imageOpts);
    if (!imageOpts.aspectRatio || imageOpts.aspectRatio === '1:1') {
        raw = await makeSquareBase64(raw);
    }
    // Upscale if API ignored imageSize
    if (imageOpts.imageSize) raw = await upscaleIfNeeded(raw, imageOpts.imageSize, imageOpts.aspectRatio);
    return raw;
}

// ── Shared Gemini call with retry + backoff + concurrency ───────────────────
const MAX_RETRIES = 3;
const RETRY_DELAYS = [2000, 5000, 10000];

// Concurrency: scale with the number of Gemini API keys we have. With key
// rotation + 60s 429 cooldowns, ~2 in-flight calls per key is comfortable.
const MAX_CONCURRENT = Math.max(3, geminiClients.length * 2);
let activeGeminiCalls = 0;
const geminiQueue = [];

function acquireGeminiSlot() {
    return new Promise(resolve => {
        if (activeGeminiCalls < MAX_CONCURRENT) {
            activeGeminiCalls++;
            resolve();
        } else {
            geminiQueue.push(resolve);
        }
    });
}

function releaseGeminiSlot() {
    activeGeminiCalls--;
    if (geminiQueue.length > 0) {
        activeGeminiCalls++;
        geminiQueue.shift()();
    }
}

// Race a promise against a timeout so a hung SDK call surfaces as an error
// instead of blocking a slot indefinitely.
function withTimeout(promise, ms, label = 'gemini') {
    return new Promise((resolve, reject) => {
        const t = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
        promise.then(
            v => { clearTimeout(t); resolve(v); },
            e => { clearTimeout(t); reject(e); }
        );
    });
}

// 429 detector — Vertex/Gemini surface RESOURCE_EXHAUSTED, rate limit, or status 429.
function _is429(err) {
    const msg = String(err?.message || err || '');
    return /\b429\b|RESOURCE_EXHAUSTED|Too Many Requests|quota|rate limit/i.test(msg);
}

async function callGemini(parts, attempt = 0, imageOpts = {}) {
    await acquireGeminiSlot();
    let pickedIdx = null;
    try {
        const imgConfig = {};
        if (imageOpts.aspectRatio) imgConfig.aspectRatio = imageOpts.aspectRatio;
        if (imageOpts.imageSize) imgConfig.imageSize = imageOpts.imageSize;

        const picked = await pickGeminiClient();
        if (!picked) throw new Error('No Gemini client configured.');
        pickedIdx = picked.idx;

        console.log(`[Gemini] key#${pickedIdx + 1}/${geminiClients.length} calling... (${parts.filter(p => p.inlineData).length} image(s))${attempt > 0 ? ` [retry ${attempt}]` : ''}${Object.keys(imgConfig).length ? ` [${JSON.stringify(imgConfig)}]` : ''}`);
        const config = { responseModalities: ['TEXT', 'IMAGE'] };
        if (Object.keys(imgConfig).length > 0) config.imageConfig = imgConfig;

        const response = await picked.client.models.generateContent({
            model: 'gemini-3-pro-image-preview',
            contents: [{ role: 'user', parts }],
            config,
        });

        const resParts  = response.candidates?.[0]?.content?.parts || [];
        const imagePart = resParts.find(p => p.inlineData?.data && !p.thought);
        if (!imagePart) {
            const text = resParts.find(p => p.text)?.text || 'none';
            console.error('[Gemini] No image. Response text:', text.slice(0, 300));
            throw new Error('Gemini returned no image — ' + text.slice(0, 120));
        }

        const buf = Buffer.from(imagePart.inlineData.data, 'base64');
        const meta = await sharp(buf).metadata();
        if (!meta.width || !meta.height) throw new Error('Gemini returned invalid image data');

        console.log(`[Gemini] key#${pickedIdx + 1} OK — native ${meta.width}x${meta.height} (${(buf.length / 1024 / 1024).toFixed(1)}MB)${imgConfig.imageSize ? ` [requested ${imgConfig.imageSize}]` : ''}`);
        return imagePart.inlineData.data;
    } catch (err) {
        if (_is429(err) && pickedIdx != null) {
            markClientCooldown(pickedIdx);
            console.log(`[Gemini] key#${pickedIdx + 1} hit 429 — cooled for ${COOLDOWN_MS / 1000}s, will rotate next call`);
        }
        if (attempt < MAX_RETRIES - 1) {
            // Slightly longer backoff for 429 since it's a quota wait
            const delay = _is429(err) ? 3000 : (RETRY_DELAYS[attempt] || 5000);
            console.log(`[Gemini] retry ${attempt + 1}/${MAX_RETRIES} in ${delay}ms...`);
            await new Promise(r => setTimeout(r, delay));
            return callGemini(parts, attempt + 1, imageOpts);
        }
        throw err;
    } finally {
        releaseGeminiSlot();
    }
}

async function callNanoBana2(parts, attempt = 0, imageOpts = {}) {
    await acquireGeminiSlot();
    let pickedIdx = null;
    try {
        const imgConfig = {
            aspectRatio: imageOpts.aspectRatio || '1:1',
            imageSize: imageOpts.imageSize || '2K',
        };
        const picked = await pickGeminiClient();
        if (!picked) throw new Error('No Gemini client configured.');
        pickedIdx = picked.idx;

        console.log(`[NanoBana2] key#${pickedIdx + 1}/${geminiClients.length} calling gemini-3.1-flash-image-preview... (${parts.filter(p => p.inlineData).length} image(s))${attempt > 0 ? ` [retry ${attempt}]` : ''} [${JSON.stringify(imgConfig)}]`);
        const response = await picked.client.models.generateContent({
            model: 'gemini-3.1-flash-image-preview',
            contents: [{ role: 'user', parts }],
            config: {
                responseModalities: ['TEXT', 'IMAGE'],
                imageConfig: imgConfig,
            },
        });

        const resParts  = response.candidates?.[0]?.content?.parts || [];
        const imagePart = resParts.find(p => p.inlineData?.data && !p.thought);
        if (!imagePart) {
            const text = resParts.find(p => p.text)?.text || 'none';
            console.error('[NanoBana2] No image. Response text:', text.slice(0, 300));
            throw new Error('Nano Banana 2 returned no image — ' + text.slice(0, 120));
        }

        const buf = Buffer.from(imagePart.inlineData.data, 'base64');
        const meta = await sharp(buf).metadata();
        if (!meta.width || !meta.height) throw new Error('Nano Banana 2 returned invalid image data');

        console.log(`[NanoBana2] key#${pickedIdx + 1} OK — native ${meta.width}x${meta.height} (${(buf.length / 1024 / 1024).toFixed(1)}MB) [requested ${imgConfig.imageSize}]`);
        return imagePart.inlineData.data;
    } catch (err) {
        if (_is429(err) && pickedIdx != null) {
            markClientCooldown(pickedIdx);
            console.log(`[NanoBana2] key#${pickedIdx + 1} hit 429 — cooled for ${COOLDOWN_MS / 1000}s, will rotate next call`);
        }
        if (attempt < MAX_RETRIES - 1) {
            const delay = _is429(err) ? 3000 : (RETRY_DELAYS[attempt] || 5000);
            console.log(`[NanoBana2] retry ${attempt + 1}/${MAX_RETRIES} in ${delay}ms...`);
            await new Promise(r => setTimeout(r, delay));
            return callNanoBana2(parts, attempt + 1, imageOpts);
        }
        throw err;
    } finally {
        releaseGeminiSlot();
    }
}

// ── Overlay: weight text + logo ────────────────────────────────────────────
const LOGO_PATH = path.join(__dirname, 'public', 'assets', 'taheri-light.png');
// Futura LT Light must be installed in the system/user fonts directory for librsvg to find it
const FUTURA_FONT_FAMILY = 'Futura LT';

async function applyOverlay(base64, weightText, brandId = DEFAULT_BRAND) {
    const brand = resolveBrand(brandId);
    const overlayCfg = brand.overlay;
    if (!overlayCfg || !overlayCfg.supported) return base64;

    const buf = Buffer.from(base64, 'base64');
    const meta = await sharp(buf).metadata();
    const w = meta.width;
    const h = meta.height;

    // Scale overlay relative to the SHORTER dimension so portrait/landscape
    // outputs don't oversize the logo. Reference square: 3000x3000. (Audit #11.)
    const ref = Math.min(w, h);
    const pad = Math.round(ref * (125 / 3000));
    const fontSize = Math.round(ref * (143 / 3000));
    const logoWidth = Math.round(ref * (580 / 3000));

    const composites = [];

    // Weight text (top-left) — SVG, embedded Jost Light (Futura alternative)
    if (weightText && weightText.trim()) {
        const textLeftPad = Math.round(ref * (120 / 3000));
        const textSvg = Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${fontSize * 2}">
            <text x="${textLeftPad}" y="${fontSize * 1.1}" font-family="${FUTURA_FONT_FAMILY}" font-size="${fontSize}" font-weight="300" fill="white" letter-spacing="2">${weightText.trim()}</text>
        </svg>`);
        const textPad = Math.round(ref * (100 / 3000));
        composites.push({ input: textSvg, top: textPad, left: 0, });
    }

    // Brand logo (top-right) — path comes from brands.js so each brand can ship
    // its own asset variant.
    const logoAbsPath = overlayCfg.logoPath && path.isAbsolute(overlayCfg.logoPath)
        ? overlayCfg.logoPath
        : path.join(__dirname, overlayCfg.logoPath || '');
    if (logoAbsPath && fs.existsSync(logoAbsPath)) {
        const logoBuf = await sharp(logoAbsPath)
            .resize({ width: logoWidth, fit: 'inside' })
            .png()
            .toBuffer();
        const logoMeta = await sharp(logoBuf).metadata();
        composites.push({
            input: logoBuf,
            top: pad,
            left: w - logoMeta.width - pad,
        });
    }

    if (composites.length === 0) return base64;

    console.log(`[Overlay] brand=${brand.id} ${weightText ? 'weight "' + weightText.trim() + '"' : 'no weight'} + logo on ${w}x${h}`);
    const result = await sharp(buf)
        .composite(composites)
        .png({ compressionLevel: 6 })
        .toBuffer();
    return result.toString('base64');
}

// ── Image helpers ───────────────────────────────────────────────────────────
async function makeSquareBase64(base64) {
    const buf = Buffer.from(base64, 'base64');
    const meta = await sharp(buf).metadata();
    // If already square (or very close), return as-is
    if (Math.abs(meta.width - meta.height) <= 2) return base64;
    // Crop to square from center instead of padding with white
    const size = Math.min(meta.width, meta.height);
    const out = await sharp(buf)
        .extract({
            left: Math.floor((meta.width - size) / 2),
            top: Math.floor((meta.height - size) / 2),
            width: size,
            height: size,
        })
        .png()
        .toBuffer();
    return out.toString('base64');
}

async function toJpeg(filePathOrName, buffer) {
    const ext = path.extname(filePathOrName).toLowerCase();
    if (ext === '.heic' || ext === '.heif') {
        const tmpIn  = path.join(os.tmpdir(), `heic-in-${Date.now()}.heic`);
        const tmpOut = path.join(os.tmpdir(), `heic-out-${Date.now()}.jpg`);
        try {
            fs.writeFileSync(tmpIn, buffer);
            await new Promise((resolve, reject) => {
                execFile('sips', ['-s', 'format', 'jpeg', tmpIn, '--out', tmpOut], err => err ? reject(err) : resolve());
            });
            return fs.readFileSync(tmpOut);
        } finally {
            if (fs.existsSync(tmpIn)) fs.unlinkSync(tmpIn);
            if (fs.existsSync(tmpOut)) fs.unlinkSync(tmpOut);
        }
    }
    return sharp(buffer).jpeg({ quality: 95 }).toBuffer();
}

// ── List image files in a product folder ──────────────────────────────────
// Used by the Compare-with-original modal on batch result cards. Returns
// absolute paths (the UI loads them via /file?path=...).
app.get('/folder-images', (req, res) => {
    const folderPath = (req.query.path || '').trim();
    if (!folderPath) return res.status(400).json({ error: 'path required' });
    if (!fs.existsSync(folderPath)) return res.status(404).json({ error: 'folder not found' });
    if (!fs.statSync(folderPath).isDirectory()) return res.status(400).json({ error: 'not a directory' });
    try {
        const exts = /\.(jpe?g|png|webp|gif|heic|heif)$/i;
        const files = fs.readdirSync(folderPath)
            .filter(f => !f.startsWith('.') && exts.test(f))
            .map(f => path.join(folderPath, f));
        res.json({ files });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// ── Native folder picker (macOS) ──────────────────────────────────────────
// Spawns Finder's "choose folder" dialog and returns the absolute path. The
// UI's Browse button hits this so users don't have to paste a path.
app.post('/pick-folder', (req, res) => {
    if (process.platform !== 'darwin') {
        return res.status(501).json({ error: 'Folder picker is macOS-only. Paste the path manually.' });
    }
    const script = 'POSIX path of (choose folder with prompt "Pick a product folder")';
    execFile('osascript', ['-e', script], { timeout: 120_000 }, (err, stdout) => {
        if (err) {
            // User cancelled → osascript exits non-zero. Treat as no-op.
            if (/User canceled|cancelled/i.test(String(err))) return res.json({ cancelled: true });
            return res.status(500).json({ error: err.message });
        }
        const folderPath = (stdout || '').trim().replace(/\/$/, '');
        if (!folderPath) return res.json({ cancelled: true });
        res.json({ folderPath });
    });
});

// ── Start ───────────────────────────────────────────────────────────────────
app.listen(PORT, () => console.log(`\nTaheri Pipeline → http://localhost:${PORT}\n`));
