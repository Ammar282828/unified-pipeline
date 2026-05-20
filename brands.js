// Brand registry — drives copy, assets, and brand-specific shot availability.
// The HTTP server reads the active brand from req.body.brand / req.query.brand
// (falls back to DEFAULT_BRAND). All brand-specific prompt text flows from here.

const BRANDS = {
    mina: {
        id: 'mina',
        label: 'House of Mina',
        shortLabel: 'Mina',
        domain: 'houseofmina.store',
        logo: '/assets/brands/mina/logo.png',
        logoDark: '/assets/brands/mina/logo.png',
        tagline: 'luxury South Asian jewelry brand based in Karachi',
        zipFilename: 'house-of-mina-shots.zip',
        zipSelectedFilename: 'house-of-mina-selected.zip',
        scrapePlaceholder: 'https://houseofmina.store/products/...',
        // Skill-aligned brand preset (nanobanana-jewelry/SKILL.md). These
        // fields drop into the master prompt template's lighting / mood /
        // avoid slots. NO metal-naming — the skill warns metal names can
        // override the actual metal in the reference photo.
        voice: 'House of Mina, a luxury South Asian bridal jewelry brand based in Karachi.',
        lighting: 'soft diffused key light from upper left producing clean white fire on stones (not rainbow flare), gentle fill from the right, faint rim light to separate the piece from the background',
        mood: 'romantic, premium, bridal, editorial',
        avoid: 'dull grey metal, yellow cast on metal, BTS or workshop feel',
        // How the brand name appears in the ecom_stand scene.
        ecomStandBrandRef: 'House of Mina brand display presentation',
        // WhatsApp caption copywriter system prompt — brand voice comes from here.
        captionSystem: `You are the copywriter for House of Mina (houseofmina.store), a luxury South Asian jewelry brand based in Karachi. You write WhatsApp community posts to showcase new jewelry pieces.`,
        captionBrandMention: '*House of Mina*',
        // Extra brand-only shot types injected into the shot catalog.
        extraShots: [],
        // Logo+weight overlay composited onto finished ecom shots. Mina has no
        // overlay logo yet, so overlay is opt-in but will render weight-text only.
        overlay: {
            supported: true,
            logoPath: 'public/assets/brands/mina/logo.png',
            defaultEnabled: false,
        },
    },

    taheri: {
        id: 'taheri',
        label: 'Taheri Collections',
        shortLabel: 'Taheri',
        domain: 'tahericollections.com',
        logo: '/assets/brands/taheri/logo.png',
        logoDark: '/assets/brands/taheri/logo-dark.png',
        tagline: 'Taheri Collections — signature jewelry with an editorial, heritage-modern sensibility',
        zipFilename: 'taheri-shots.zip',
        zipSelectedFilename: 'taheri-selected.zip',
        scrapePlaceholder: 'https://tahericollections.com/products/...',
        // Skill-aligned brand preset (nanobanana-jewelry/SKILL.md).
        voice: 'Taheri Collections, a heritage-modern jewelry house with a Bohri community lineage.',
        lighting: 'soft warm key light from upper left with gentle gold reflection, subtle fill from the right, no harsh contrast',
        mood: 'heritage, premium, refined, Bohri elegance',
        avoid: 'black backgrounds, modern minimalist coldness, BTS or workshop, karat call-outs',
        ecomStandBrandRef: 'Taheri Collections brand display presentation',
        captionSystem: `You are the copywriter for Taheri Collections, a signature jewelry house. You write WhatsApp community posts to showcase new jewelry pieces.`,
        captionBrandMention: '*Taheri Collections*',
        // Taheri's signature deliverable — logo top-right, weight-text top-left,
        // scaled against a 3000px reference so it looks right at any output size.
        overlay: {
            supported: true,
            logoPath: 'public/assets/brands/taheri/logo-dark.png',
            defaultEnabled: true,
        },
        extraShots: [
            {
                id: 'taheri_signature',
                label: 'Taheri Signature',
                category: 'taheri',
                description: 'Wooden stand/bust on emerald green — auto-detects jewelry type',
                scenePrompt: `Taheri signature shot. FLAT dark matte emerald-green velvet background, full bleed — no environment beyond. Every piece sits on a dark walnut display (rich grain, polished, never directly on velvet). Pick by piece type: single ring → small walnut block; pendant/chain → walnut neck bust; full set → walnut bust with necklace draped, earrings at ear level, ring on a small walnut platform at the base; bangles → vertical walnut cylinder; earrings only → walnut T-bar; tikka → walnut dome. Warm soft light from upper-left. Jewelry sharp, wood supporting, centered.`,
            },
        ],
    },
};

const DEFAULT_BRAND = 'mina';

function resolveBrand(id) {
    if (id && BRANDS[id]) return BRANDS[id];
    return BRANDS[DEFAULT_BRAND];
}

function listBrands() {
    return Object.values(BRANDS).map(b => ({
        id: b.id,
        label: b.label,
        shortLabel: b.shortLabel,
        domain: b.domain,
        logo: b.logo,
        logoDark: b.logoDark,
        scrapePlaceholder: b.scrapePlaceholder,
        overlay: b.overlay
            ? { supported: !!b.overlay.supported, defaultEnabled: !!b.overlay.defaultEnabled }
            : { supported: false, defaultEnabled: false },
    }));
}

module.exports = { BRANDS, DEFAULT_BRAND, resolveBrand, listBrands };
