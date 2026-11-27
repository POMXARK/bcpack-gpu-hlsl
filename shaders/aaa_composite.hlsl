// Open AAA Showcase Final Composite: TAA HDR + Bloom + Auto-Exposure + AgX Filmic + CAS Sharpen + Interactive HUD
cbuffer AaaCompCB : register(b0) {
    float g_BloomStrength;
    float g_CasStrength;
    float g_ExposureScale;
    float g_Time;
    float g_HudEnable;     // > 0.5 draws bottom-left interactive HUD panel
    float g_FpsNorm;       // 0.0..1.0 thumb position (1.0 = MAX/VSync)
    float g_FpsCapVal;     // 0 = uncapped/VSync, 15..300 = cap
    float g_FpsLive;       // live measured FPS
    float g_SimNorm;       // 0.0..1.0 thumb position (0.40 = 1.00x)
    float g_SimSpeed;      // 0.00..4.00
    float g_ActiveDrag;    // 0 = none, 1 = FPS, 2 = SIM
    float g_VsyncOn;       // > 0.5 when VSync active
    // Bundle 1: Pursuit Zero-Copy (BcPackDecodeBatchToTextures) & VirtualDescriptorPager HUD
    float g_PursuitOn;       // > 0.5 draws top-right Pursuit HUD panel
    float g_ZcThroughputMbs; // Zero-Copy VRAM streaming MB/s
    float g_PursuitBatches;  // Total DecodeBatchToTextures batches
    float g_PagerActiveK;    // Active physical slots in thousands (0..262)
    float g_PagerEvictPerSec;// Evictions/sec in VirtualDescriptorPager
    float g_Fps1PctLow;      // Rolling 1% Low FPS
    float g_WhipPanPulse;    // 0..1 peak during 180-deg whip-pan bursts
    float g_FormatCount;     // Active BCPack format count (6)
    // UMA Emulation Stress & Optimization HUD (top-left panel)
    float g_UmaOn;           // > 0.5 draws top-left UMA HUD panel
    float g_UmaMask;         // bitmask 0..15: [1]=Detile, [2]=Patch, [3]=8T+Hazard, [4]=VRAM
    float g_UmaSub1Ms;
    float g_UmaSub2Ms;
    float g_UmaSub3Ms;
    float g_UmaSub4Ms;
    float g_UmaTotalMs;
    float g_UmaOkMeshes;
};

Texture2D<float4>       g_TaaHDR      : register(t0);
Texture2D<float4>       g_BloomTex    : register(t1);
StructuredBuffer<float> g_ExposureBuf : register(t2);
SamplerState            g_LinClamp    : register(s0);

struct CompPSIn {
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

CompPSIn VSMain(uint vid : SV_VertexID) {
    float2 p = float2((vid == 1u) ? 3.0 : -1.0, (vid == 2u) ? 3.0 : -1.0);
    CompPSIn o;
    o.pos = float4(p, 0.0, 1.0);
    o.uv = float2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return o;
}

// AgX-inspired Filmic Transform (smooth warm highlight roll-off without sunset blowout)
float3 agx_filmic(float3 x) {
    static const float3x3 agxMat = float3x3(
        0.84247906, 0.07843360, 0.07922375,
        0.04232824, 0.87846864, 0.07916613,
        0.04237565, 0.07843360, 0.87914297
    );
    static const float3x3 agxInv = float3x3(
         1.1968790, -0.0980209, -0.0990297,
        -0.0528969,  1.1519031, -0.0989612,
        -0.0529716, -0.0980435,  1.1510737
    );
    float3 v = mul(agxMat, max(x, 1e-6));
    v = clamp((log2(v) + 12.47393) / 16.5, 0.0, 1.0);
    float3 v2 = v * v;
    float3 v4 = v2 * v2;
    v = 15.5 * v4 * v2 - 40.14 * v4 * v + 31.96 * v4 - 6.868 * v2 * v + 0.4298 * v2 + 0.1191 * v - 0.00232;
    v = mul(agxInv, v);
    return saturate(v);
}

float luma(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

#define GLYPH5(r0, r1, r2, r3, r4) (((r0) << 12) | ((r1) << 9) | ((r2) << 6) | ((r3) << 3) | (r4))

uint hud_glyph_mask(uint ch) {
    // Codes: 0..9 = digits, 10=' ', 11=':', 12='.', 13='F', 14='P', 15='S',
    // 16='C', 17='A', 18='M', 19='X', 20='V', 21='Y', 22='N', 23='L', 24='I',
    // 25='E', 26='D', 27='U', 28='H'
    if (ch == 0u)  return GLYPH5(7u, 5u, 5u, 5u, 7u);
    if (ch == 1u)  return GLYPH5(2u, 6u, 2u, 2u, 7u);
    if (ch == 2u)  return GLYPH5(7u, 1u, 7u, 4u, 7u);
    if (ch == 3u)  return GLYPH5(7u, 1u, 7u, 1u, 7u);
    if (ch == 4u)  return GLYPH5(5u, 5u, 7u, 1u, 1u);
    if (ch == 5u)  return GLYPH5(7u, 4u, 7u, 1u, 7u);
    if (ch == 6u)  return GLYPH5(7u, 4u, 7u, 5u, 7u);
    if (ch == 7u)  return GLYPH5(7u, 1u, 1u, 2u, 2u);
    if (ch == 8u)  return GLYPH5(7u, 5u, 7u, 5u, 7u);
    if (ch == 9u)  return GLYPH5(7u, 5u, 7u, 1u, 7u);
    if (ch == 11u) return GLYPH5(0u, 2u, 0u, 2u, 0u); // ':'
    if (ch == 12u) return GLYPH5(0u, 0u, 0u, 0u, 2u); // '.'
    if (ch == 13u) return GLYPH5(7u, 4u, 6u, 4u, 4u); // 'F'
    if (ch == 14u) return GLYPH5(7u, 5u, 7u, 4u, 4u); // 'P'
    if (ch == 15u) return GLYPH5(7u, 4u, 7u, 1u, 7u); // 'S'
    if (ch == 16u) return GLYPH5(7u, 4u, 4u, 4u, 7u); // 'C'
    if (ch == 17u) return GLYPH5(2u, 5u, 7u, 5u, 5u); // 'A'
    if (ch == 18u) return GLYPH5(5u, 7u, 5u, 5u, 5u); // 'M'
    if (ch == 19u) return GLYPH5(5u, 5u, 2u, 5u, 5u); // 'X'
    if (ch == 20u) return GLYPH5(5u, 5u, 5u, 5u, 2u); // 'V'
    if (ch == 21u) return GLYPH5(5u, 5u, 2u, 2u, 2u); // 'Y'
    if (ch == 22u) return GLYPH5(5u, 6u, 5u, 3u, 5u); // 'N'
    if (ch == 23u) return GLYPH5(4u, 4u, 4u, 4u, 7u); // 'L'
    if (ch == 24u) return GLYPH5(7u, 2u, 2u, 2u, 7u); // 'I'
    if (ch == 25u) return GLYPH5(7u, 4u, 6u, 4u, 7u); // 'E'
    if (ch == 26u) return GLYPH5(6u, 5u, 5u, 5u, 6u); // 'D'
    if (ch == 27u) return GLYPH5(5u, 5u, 5u, 5u, 7u); // 'U'
    if (ch == 28u) return GLYPH5(5u, 5u, 7u, 5u, 5u); // 'H'
    if (ch == 29u) return GLYPH5(6u, 5u, 6u, 5u, 6u); // 'B'
    if (ch == 30u) return GLYPH5(7u, 4u, 5u, 5u, 7u); // 'G'
    if (ch == 31u) return GLYPH5(5u, 5u, 6u, 5u, 5u); // 'K'
    if (ch == 32u) return GLYPH5(7u, 5u, 5u, 5u, 7u); // 'O'
    if (ch == 33u) return GLYPH5(6u, 5u, 6u, 5u, 5u); // 'R'
    if (ch == 34u) return GLYPH5(7u, 2u, 2u, 2u, 2u); // 'T'
    if (ch == 35u) return GLYPH5(5u, 5u, 5u, 7u, 5u); // 'W'
    if (ch == 36u) return GLYPH5(7u, 1u, 2u, 4u, 7u); // 'Z'
    if (ch == 37u) return GLYPH5(1u, 1u, 2u, 4u, 4u); // '/'
    if (ch == 38u) return GLYPH5(5u, 1u, 2u, 4u, 5u); // '%'
    if (ch == 39u) return GLYPH5(0u, 0u, 7u, 0u, 0u); // '-'
    return 0u;
}

float sample_hud_char(float2 uv, float2 org, uint ch) {
    static const float CW = 0.0017;
    static const float CH = 0.0030;
    float lx = (uv.x - org.x) / CW;
    float ly = (uv.y - org.y) / CH;
    if (lx < 0.0 || lx >= 3.0 || ly < 0.0 || ly >= 5.0) return 0.0;
    uint gx = (uint)lx;
    uint gy = (uint)ly;
    uint bit = (4u - gy) * 3u + (2u - gx);
    return ((hud_glyph_mask(ch) >> bit) & 1u) != 0u ? 1.0 : 0.0;
}

float draw_uint3(float2 uv, float2 org, float adv, uint val) {
    uint v = min(val, 999u);
    uint h = v / 100u;
    uint t = (v / 10u) % 10u;
    uint o = v % 10u;
    float m = 0.0;
    uint idx = 0u;
    if (h > 0u) { m = max(m, sample_hud_char(uv, org + float2(adv * idx, 0.0), h)); idx++; }
    if (h > 0u || t > 0u) { m = max(m, sample_hud_char(uv, org + float2(adv * idx, 0.0), t)); idx++; }
    m = max(m, sample_hud_char(uv, org + float2(adv * idx, 0.0), o));
    return m;
}

float draw_uint4(float2 uv, float2 org, float adv, uint val) {
    uint v = min(val, 9999u);
    uint k = v / 1000u;
    uint h = (v / 100u) % 10u;
    uint t = (v / 10u) % 10u;
    uint o = v % 10u;
    float m = 0.0;
    uint idx = 0u;
    if (k > 0u) { m = max(m, sample_hud_char(uv, org + float2(adv * idx, 0.0), k)); idx++; }
    if (k > 0u || h > 0u) { m = max(m, sample_hud_char(uv, org + float2(adv * idx, 0.0), h)); idx++; }
    if (k > 0u || h > 0u || t > 0u) { m = max(m, sample_hud_char(uv, org + float2(adv * idx, 0.0), t)); idx++; }
    m = max(m, sample_hud_char(uv, org + float2(adv * idx, 0.0), o));
    return m;
}

float sd_round_box(float2 p, float2 b, float r) {
    float2 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

float3 composite_hud(float3 bg, float2 uv) {
    // Bottom-left translucent frosted panel: x in [0.018, 0.310], y in [0.815, 0.972]
    static const float2 P_CENTER = float2(0.164, 0.8935);
    static const float2 P_HALF   = float2(0.146, 0.0785);
    float2 aspect = float2(1.7777778, 1.0);
    float sd_panel = sd_round_box((uv - P_CENTER) * aspect, P_HALF * aspect, 0.014);
    if (sd_panel > 0.004) {
        return bg;
    }

    float panel_alpha = smoothstep(0.002, -0.002, sd_panel);
    float border_mask = smoothstep(0.004, 0.001, abs(sd_panel));
    float3 col = lerp(bg, bg * 0.25 + float3(0.035, 0.052, 0.078), 0.80 * panel_alpha);
    col = lerp(col, float3(0.24, 0.62, 0.84), 0.45 * border_mask);

    static const float ADV = 0.0068;
    static const float X0  = 0.032;
    static const float X1  = 0.296;

    // --- Row 1 Label: "FPS CAP: <val>" and right-aligned "LIVE:<fps>" ---
    float text1 = 0.0;
    float2 r1 = float2(X0, 0.830);
    // "FPS CAP:"
    text1 = max(text1, sample_hud_char(uv, r1 + float2(ADV * 0.0, 0.0), 13u)); // F
    text1 = max(text1, sample_hud_char(uv, r1 + float2(ADV * 1.0, 0.0), 14u)); // P
    text1 = max(text1, sample_hud_char(uv, r1 + float2(ADV * 2.0, 0.0), 15u)); // S
    text1 = max(text1, sample_hud_char(uv, r1 + float2(ADV * 4.0, 0.0), 16u)); // C
    text1 = max(text1, sample_hud_char(uv, r1 + float2(ADV * 5.0, 0.0), 17u)); // A
    text1 = max(text1, sample_hud_char(uv, r1 + float2(ADV * 6.0, 0.0), 14u)); // P
    text1 = max(text1, sample_hud_char(uv, r1 + float2(ADV * 7.0, 0.0), 11u)); // :

    float2 v1 = r1 + float2(ADV * 9.0, 0.0);
    if (g_FpsCapVal < 0.5) {
        if (g_VsyncOn > 0.5) {
            text1 = max(text1, sample_hud_char(uv, v1 + float2(ADV * 0.0, 0.0), 20u)); // V
            text1 = max(text1, sample_hud_char(uv, v1 + float2(ADV * 1.0, 0.0), 15u)); // S
            text1 = max(text1, sample_hud_char(uv, v1 + float2(ADV * 2.0, 0.0), 21u)); // Y
            text1 = max(text1, sample_hud_char(uv, v1 + float2(ADV * 3.0, 0.0), 22u)); // N
            text1 = max(text1, sample_hud_char(uv, v1 + float2(ADV * 4.0, 0.0), 16u)); // C
        } else {
            text1 = max(text1, sample_hud_char(uv, v1 + float2(ADV * 0.0, 0.0), 18u)); // M
            text1 = max(text1, sample_hud_char(uv, v1 + float2(ADV * 1.0, 0.0), 17u)); // A
            text1 = max(text1, sample_hud_char(uv, v1 + float2(ADV * 2.0, 0.0), 19u)); // X
        }
    } else {
        text1 = max(text1, draw_uint3(uv, v1, ADV, (uint)round(g_FpsCapVal)));
    }

    float2 live_pos = float2(0.228, 0.830);
    float text_live = 0.0;
    text_live = max(text_live, sample_hud_char(uv, live_pos + float2(ADV * 0.0, 0.0), 23u)); // L
    text_live = max(text_live, sample_hud_char(uv, live_pos + float2(ADV * 1.0, 0.0), 24u)); // I
    text_live = max(text_live, sample_hud_char(uv, live_pos + float2(ADV * 2.0, 0.0), 20u)); // V
    text_live = max(text_live, sample_hud_char(uv, live_pos + float2(ADV * 3.0, 0.0), 25u)); // E
    text_live = max(text_live, sample_hud_char(uv, live_pos + float2(ADV * 4.0, 0.0), 11u)); // :
    text_live = max(text_live, draw_uint3(uv, live_pos + float2(ADV * 5.5, 0.0), ADV, (uint)round(max(g_FpsLive, 0.0))));

    // --- Row 1 Slider Track + Thumb (Cyan) ---
    float tr1_y = 0.872;
    float tr1_fill_x = lerp(X0, X1, saturate(g_FpsNorm));
    float sd_tr1 = sd_round_box((uv - float2(0.5 * (X0 + X1), tr1_y)) * aspect, float2(0.5 * (X1 - X0), 0.0045) * aspect, 0.004);
    if (sd_tr1 < 0.002) {
        float m = smoothstep(0.002, -0.001, sd_tr1);
        float3 tcol = (uv.x <= tr1_fill_x) ? float3(0.18, 0.80, 0.96) : float3(0.16, 0.21, 0.28);
        col = lerp(col, tcol, m * 0.92);
    }
    float d_th1 = length((uv - float2(tr1_fill_x, tr1_y)) * aspect);
    float th1_r = (g_ActiveDrag > 0.5 && g_ActiveDrag < 1.5) ? 0.0155 : 0.0130;
    if (d_th1 < th1_r + 0.004) {
        float outer = smoothstep(th1_r + 0.003, th1_r, d_th1);
        float inner = smoothstep(th1_r * 0.58, th1_r * 0.35, d_th1);
        float3 th_col = (g_ActiveDrag > 0.5 && g_ActiveDrag < 1.5) ? float3(0.55, 0.95, 1.0) : float3(0.24, 0.85, 0.98);
        col = lerp(col, float3(0.95, 0.98, 1.0), outer * 0.95);
        col = lerp(col, th_col, inner);
    }

    // --- Row 2 Label: "SIM SPD: <val>" and right-aligned "H:HUD" ---
    float text2 = 0.0;
    float2 r2 = float2(X0, 0.898);
    // "SIM SPD:"
    text2 = max(text2, sample_hud_char(uv, r2 + float2(ADV * 0.0, 0.0), 15u)); // S
    text2 = max(text2, sample_hud_char(uv, r2 + float2(ADV * 1.0, 0.0), 24u)); // I
    text2 = max(text2, sample_hud_char(uv, r2 + float2(ADV * 2.0, 0.0), 18u)); // M
    text2 = max(text2, sample_hud_char(uv, r2 + float2(ADV * 4.0, 0.0), 15u)); // S
    text2 = max(text2, sample_hud_char(uv, r2 + float2(ADV * 5.0, 0.0), 14u)); // P
    text2 = max(text2, sample_hud_char(uv, r2 + float2(ADV * 6.0, 0.0), 26u)); // D
    text2 = max(text2, sample_hud_char(uv, r2 + float2(ADV * 7.0, 0.0), 11u)); // :

    float2 v2 = r2 + float2(ADV * 9.0, 0.0);
    if (g_SimSpeed < 0.005) {
        text2 = max(text2, sample_hud_char(uv, v2 + float2(ADV * 0.0, 0.0), 14u)); // P
        text2 = max(text2, sample_hud_char(uv, v2 + float2(ADV * 1.0, 0.0), 17u)); // A
        text2 = max(text2, sample_hud_char(uv, v2 + float2(ADV * 2.0, 0.0), 27u)); // U
        text2 = max(text2, sample_hud_char(uv, v2 + float2(ADV * 3.0, 0.0), 15u)); // S
        text2 = max(text2, sample_hud_char(uv, v2 + float2(ADV * 4.0, 0.0), 25u)); // E
    } else {
        uint sc = (uint)round(clamp(g_SimSpeed, 0.0, 4.0) * 100.0);
        uint d_one = sc / 100u;
        uint d_ten = (sc / 10u) % 10u;
        uint d_hun = sc % 10u;
        text2 = max(text2, sample_hud_char(uv, v2 + float2(ADV * 0.0, 0.0), d_one));
        text2 = max(text2, sample_hud_char(uv, v2 + float2(ADV * 0.85, 0.0), 12u)); // '.'
        text2 = max(text2, sample_hud_char(uv, v2 + float2(ADV * 1.7, 0.0), d_ten));
        text2 = max(text2, sample_hud_char(uv, v2 + float2(ADV * 2.7, 0.0), d_hun));
        text2 = max(text2, sample_hud_char(uv, v2 + float2(ADV * 3.8, 0.0), 19u)); // 'X'
    }

    float2 hint_pos = float2(0.258, 0.898);
    float text_hint = 0.0;
    text_hint = max(text_hint, sample_hud_char(uv, hint_pos + float2(ADV * 0.0, 0.0), 28u)); // H
    text_hint = max(text_hint, sample_hud_char(uv, hint_pos + float2(ADV * 1.0, 0.0), 11u)); // :
    text_hint = max(text_hint, sample_hud_char(uv, hint_pos + float2(ADV * 2.0, 0.0), 28u)); // H
    text_hint = max(text_hint, sample_hud_char(uv, hint_pos + float2(ADV * 3.0, 0.0), 27u)); // U
    text_hint = max(text_hint, sample_hud_char(uv, hint_pos + float2(ADV * 4.0, 0.0), 26u)); // D

    // --- Row 2 Slider Track + 1.0X Tick + Thumb (Amber) ---
    float tr2_y = 0.942;
    float tr2_fill_x = lerp(X0, X1, saturate(g_SimNorm));
    float sd_tr2 = sd_round_box((uv - float2(0.5 * (X0 + X1), tr2_y)) * aspect, float2(0.5 * (X1 - X0), 0.0045) * aspect, 0.004);
    if (sd_tr2 < 0.002) {
        float m = smoothstep(0.002, -0.001, sd_tr2);
        float3 tcol = (uv.x <= tr2_fill_x) ? float3(0.96, 0.68, 0.20) : float3(0.16, 0.21, 0.28);
        col = lerp(col, tcol, m * 0.92);
    }
    // 1.00x notch at norm = 0.40
    float notch_x = lerp(X0, X1, 0.40);
    if (abs(uv.x - notch_x) < 0.0011 && abs(uv.y - tr2_y) < 0.009) {
        col = lerp(col, float3(0.85, 0.90, 0.95), 0.75);
    }
    float d_th2 = length((uv - float2(tr2_fill_x, tr2_y)) * aspect);
    float th2_r = (g_ActiveDrag > 1.5) ? 0.0155 : 0.0130;
    if (d_th2 < th2_r + 0.004) {
        float outer = smoothstep(th2_r + 0.003, th2_r, d_th2);
        float inner = smoothstep(th2_r * 0.58, th2_r * 0.35, d_th2);
        float3 th_col = (g_ActiveDrag > 1.5) ? float3(1.0, 0.88, 0.45) : float3(0.98, 0.72, 0.22);
        col = lerp(col, float3(1.0, 0.97, 0.92), outer * 0.95);
        col = lerp(col, th_col, inner);
    }

    col = lerp(col, float3(0.86, 0.96, 1.0), text1);
    col = lerp(col, float3(0.52, 0.94, 0.72), text_live);
    col = lerp(col, float3(1.0, 0.90, 0.72), text2);
    col = lerp(col, float3(0.58, 0.68, 0.78), text_hint);
    return col;
}

float3 composite_pursuit_hud(float3 bg, float2 uv) {
    // Top-right frosted HUD panel: x in [0.642, 0.982], y in [0.025, 0.215]
    static const float2 P_CENTER = float2(0.812, 0.120);
    static const float2 P_HALF   = float2(0.170, 0.095);
    float2 aspect = float2(1.7777778, 1.0);
    float sd_panel = sd_round_box((uv - P_CENTER) * aspect, P_HALF * aspect, 0.014);
    if (sd_panel > 0.004) {
        return bg;
    }

    float panel_alpha = smoothstep(0.002, -0.002, sd_panel);
    float border_mask = smoothstep(0.004, 0.001, abs(sd_panel));
    float pulse = saturate(g_WhipPanPulse);
    float3 col = lerp(bg, bg * 0.22 + float3(0.032, 0.048, 0.075), 0.84 * panel_alpha);
    float3 bcol = lerp(float3(0.24, 0.62, 0.84), float3(1.0, 0.68, 0.18), pulse);
    col = lerp(col, bcol, (0.50 + 0.35 * pulse) * border_mask);

    static const float ADV = 0.0068;
    static const float X0  = 0.658;
    static const float X1  = 0.966;

    // Row 1: "PURSUIT: 6-FMT GFX10" (or "180 TURN!" accent on right)
    float t_hdr = 0.0;
    float2 r1 = float2(X0, 0.042);
    t_hdr = max(t_hdr, sample_hud_char(uv, r1 + float2(ADV * 0.0, 0.0), 14u)); // P
    t_hdr = max(t_hdr, sample_hud_char(uv, r1 + float2(ADV * 1.0, 0.0), 27u)); // U
    t_hdr = max(t_hdr, sample_hud_char(uv, r1 + float2(ADV * 2.0, 0.0), 33u)); // R
    t_hdr = max(t_hdr, sample_hud_char(uv, r1 + float2(ADV * 3.0, 0.0), 15u)); // S
    t_hdr = max(t_hdr, sample_hud_char(uv, r1 + float2(ADV * 4.0, 0.0), 27u)); // U
    t_hdr = max(t_hdr, sample_hud_char(uv, r1 + float2(ADV * 5.0, 0.0), 24u)); // I
    t_hdr = max(t_hdr, sample_hud_char(uv, r1 + float2(ADV * 6.0, 0.0), 34u)); // T
    t_hdr = max(t_hdr, sample_hud_char(uv, r1 + float2(ADV * 7.0, 0.0), 11u)); // :
    t_hdr = max(t_hdr, sample_hud_char(uv, r1 + float2(ADV * 9.0, 0.0), (uint)round(clamp(g_FormatCount, 1.0, 9.0))));
    t_hdr = max(t_hdr, sample_hud_char(uv, r1 + float2(ADV * 10.0, 0.0), 39u)); // -
    t_hdr = max(t_hdr, sample_hud_char(uv, r1 + float2(ADV * 11.0, 0.0), 13u)); // F
    t_hdr = max(t_hdr, sample_hud_char(uv, r1 + float2(ADV * 12.0, 0.0), 18u)); // M
    t_hdr = max(t_hdr, sample_hud_char(uv, r1 + float2(ADV * 13.0, 0.0), 34u)); // T
    t_hdr = max(t_hdr, sample_hud_char(uv, r1 + float2(ADV * 15.0, 0.0), 30u)); // G
    t_hdr = max(t_hdr, sample_hud_char(uv, r1 + float2(ADV * 16.0, 0.0), 13u)); // F
    t_hdr = max(t_hdr, sample_hud_char(uv, r1 + float2(ADV * 17.0, 0.0), 19u)); // X
    t_hdr = max(t_hdr, sample_hud_char(uv, r1 + float2(ADV * 18.0, 0.0), 1u));  // 1
    t_hdr = max(t_hdr, sample_hud_char(uv, r1 + float2(ADV * 19.0, 0.0), 0u));  // 0

    float t_b = 0.0;
    float2 bpos = float2(0.922, 0.042);
    t_b = max(t_b, sample_hud_char(uv, bpos + float2(ADV * 0.0, 0.0), 29u)); // B
    t_b = max(t_b, sample_hud_char(uv, bpos + float2(ADV * 1.0, 0.0), 11u)); // :
    t_b = max(t_b, sample_hud_char(uv, bpos + float2(ADV * 2.0, 0.0), 32u)); // O
    t_b = max(t_b, sample_hud_char(uv, bpos + float2(ADV * 3.0, 0.0), 22u)); // N

    // Row 2: "ZC VRAM: <NNNN> MBS" + cyan throughput bar
    float t_zc = 0.0;
    float2 r2 = float2(X0, 0.078);
    t_zc = max(t_zc, sample_hud_char(uv, r2 + float2(ADV * 0.0, 0.0), 36u)); // Z
    t_zc = max(t_zc, sample_hud_char(uv, r2 + float2(ADV * 1.0, 0.0), 16u)); // C
    t_zc = max(t_zc, sample_hud_char(uv, r2 + float2(ADV * 3.0, 0.0), 20u)); // V
    t_zc = max(t_zc, sample_hud_char(uv, r2 + float2(ADV * 4.0, 0.0), 33u)); // R
    t_zc = max(t_zc, sample_hud_char(uv, r2 + float2(ADV * 5.0, 0.0), 17u)); // A
    t_zc = max(t_zc, sample_hud_char(uv, r2 + float2(ADV * 6.0, 0.0), 18u)); // M
    t_zc = max(t_zc, sample_hud_char(uv, r2 + float2(ADV * 7.0, 0.0), 11u)); // :
    t_zc = max(t_zc, draw_uint4(uv, r2 + float2(ADV * 9.0, 0.0), ADV, (uint)round(max(g_ZcThroughputMbs, 0.0))));
    t_zc = max(t_zc, sample_hud_char(uv, r2 + float2(ADV * 14.0, 0.0), 18u)); // M
    t_zc = max(t_zc, sample_hud_char(uv, r2 + float2(ADV * 15.0, 0.0), 29u)); // B
    t_zc = max(t_zc, sample_hud_char(uv, r2 + float2(ADV * 16.0, 0.0), 15u)); // S

    float tr1_y = 0.104;
    float fill_zc = lerp(X0, X1, saturate(g_ZcThroughputMbs / 1500.0));
    float sd_tr1 = sd_round_box((uv - float2(0.5 * (X0 + X1), tr1_y)) * aspect, float2(0.5 * (X1 - X0), 0.004) * aspect, 0.003);
    if (sd_tr1 < 0.002) {
        float m = smoothstep(0.002, -0.001, sd_tr1);
        float3 tcol = (uv.x <= fill_zc) ? lerp(float3(0.18, 0.82, 0.96), float3(1.0, 0.72, 0.22), pulse) : float3(0.15, 0.20, 0.27);
        col = lerp(col, tcol, m * 0.92);
    }

    // Row 3: "PAGER:<NNN>K/262K EV:<NNNN>" + residency gauge bar
    float t_pg = 0.0;
    float2 r3 = float2(X0, 0.122);
    t_pg = max(t_pg, sample_hud_char(uv, r3 + float2(ADV * 0.0, 0.0), 14u)); // P
    t_pg = max(t_pg, sample_hud_char(uv, r3 + float2(ADV * 1.0, 0.0), 17u)); // A
    t_pg = max(t_pg, sample_hud_char(uv, r3 + float2(ADV * 2.0, 0.0), 30u)); // G
    t_pg = max(t_pg, sample_hud_char(uv, r3 + float2(ADV * 3.0, 0.0), 25u)); // E
    t_pg = max(t_pg, sample_hud_char(uv, r3 + float2(ADV * 4.0, 0.0), 33u)); // R
    t_pg = max(t_pg, sample_hud_char(uv, r3 + float2(ADV * 5.0, 0.0), 11u)); // :
    t_pg = max(t_pg, draw_uint3(uv, r3 + float2(ADV * 6.5, 0.0), ADV, (uint)round(clamp(g_PagerActiveK, 0.0, 262.0))));
    t_pg = max(t_pg, sample_hud_char(uv, r3 + float2(ADV * 10.0, 0.0), 31u)); // K
    t_pg = max(t_pg, sample_hud_char(uv, r3 + float2(ADV * 11.0, 0.0), 37u)); // /
    t_pg = max(t_pg, sample_hud_char(uv, r3 + float2(ADV * 12.0, 0.0), 2u));  // 2
    t_pg = max(t_pg, sample_hud_char(uv, r3 + float2(ADV * 13.0, 0.0), 6u));  // 6
    t_pg = max(t_pg, sample_hud_char(uv, r3 + float2(ADV * 14.0, 0.0), 2u));  // 2
    t_pg = max(t_pg, sample_hud_char(uv, r3 + float2(ADV * 15.0, 0.0), 31u)); // K
    t_pg = max(t_pg, sample_hud_char(uv, r3 + float2(ADV * 17.5, 0.0), 25u)); // E
    t_pg = max(t_pg, sample_hud_char(uv, r3 + float2(ADV * 18.5, 0.0), 20u)); // V
    t_pg = max(t_pg, sample_hud_char(uv, r3 + float2(ADV * 19.5, 0.0), 11u)); // :
    float ev_s = max(g_PagerEvictPerSec, 0.0);
    if (ev_s >= 1000.0) {
        uint ev_tenths_k = min((uint)round(ev_s / 100.0), 9999u);
        uint ev_whole = ev_tenths_k / 10u;
        uint ev_frac = ev_tenths_k % 10u;
        float ev_x = 20.8;
        if (ev_whole >= 100u) {
            t_pg = max(t_pg, draw_uint3(uv, r3 + float2(ADV * ev_x, 0.0), ADV, ev_whole));
            t_pg = max(t_pg, sample_hud_char(uv, r3 + float2(ADV * (ev_x + 3.2), 0.0), 31u)); // K
        } else {
            uint d_off = (ev_whole >= 10u) ? 2u : 1u;
            t_pg = max(t_pg, draw_uint3(uv, r3 + float2(ADV * ev_x, 0.0), ADV, ev_whole));
            t_pg = max(t_pg, sample_hud_char(uv, r3 + float2(ADV * (ev_x + (float)d_off), 0.0), 12u)); // '.'
            t_pg = max(t_pg, sample_hud_char(uv, r3 + float2(ADV * (ev_x + (float)d_off + 0.85), 0.0), ev_frac));
            t_pg = max(t_pg, sample_hud_char(uv, r3 + float2(ADV * (ev_x + (float)d_off + 1.95), 0.0), 31u)); // K
        }
    } else {
        t_pg = max(t_pg, draw_uint3(uv, r3 + float2(ADV * 21.0, 0.0), ADV, (uint)round(ev_s)));
    }

    float tr2_y = 0.148;
    float fill_pg = lerp(X0, X1, saturate(g_PagerActiveK / 262.0));
    float sd_tr2 = sd_round_box((uv - float2(0.5 * (X0 + X1), tr2_y)) * aspect, float2(0.5 * (X1 - X0), 0.004) * aspect, 0.003);
    if (sd_tr2 < 0.002) {
        float m = smoothstep(0.002, -0.001, sd_tr2);
        float3 tcol = (uv.x <= fill_pg) ? lerp(float3(0.32, 0.92, 0.58), float3(0.96, 0.64, 0.18), pulse) : float3(0.15, 0.20, 0.27);
        col = lerp(col, tcol, m * 0.92);
    }

    // Row 4: "FPS:<NNN>  1%LOW:<NNN>"
    float t_fps = 0.0;
    float2 r4 = float2(X0, 0.172);
    t_fps = max(t_fps, sample_hud_char(uv, r4 + float2(ADV * 0.0, 0.0), 13u)); // F
    t_fps = max(t_fps, sample_hud_char(uv, r4 + float2(ADV * 1.0, 0.0), 14u)); // P
    t_fps = max(t_fps, sample_hud_char(uv, r4 + float2(ADV * 2.0, 0.0), 15u)); // S
    t_fps = max(t_fps, sample_hud_char(uv, r4 + float2(ADV * 3.0, 0.0), 11u)); // :
    t_fps = max(t_fps, draw_uint3(uv, r4 + float2(ADV * 4.5, 0.0), ADV, (uint)round(max(g_FpsLive, 0.0))));

    t_fps = max(t_fps, sample_hud_char(uv, r4 + float2(ADV * 10.5, 0.0), 1u));  // 1
    t_fps = max(t_fps, sample_hud_char(uv, r4 + float2(ADV * 11.5, 0.0), 38u)); // %
    t_fps = max(t_fps, sample_hud_char(uv, r4 + float2(ADV * 12.5, 0.0), 23u)); // L
    t_fps = max(t_fps, sample_hud_char(uv, r4 + float2(ADV * 13.5, 0.0), 32u)); // O
    t_fps = max(t_fps, sample_hud_char(uv, r4 + float2(ADV * 14.5, 0.0), 35u)); // W
    t_fps = max(t_fps, sample_hud_char(uv, r4 + float2(ADV * 15.5, 0.0), 11u)); // :
    t_fps = max(t_fps, draw_uint3(uv, r4 + float2(ADV * 17.0, 0.0), ADV, (uint)round(max(g_Fps1PctLow, 0.0))));

    col = lerp(col, lerp(float3(0.86, 0.96, 1.0), float3(1.0, 0.85, 0.35), pulse), t_hdr);
    col = lerp(col, float3(0.58, 0.68, 0.78), t_b);
    col = lerp(col, float3(0.35, 0.92, 1.0), t_zc);
    col = lerp(col, float3(0.55, 0.96, 0.74), t_pg);
    col = lerp(col, float3(1.0, 0.92, 0.72), t_fps);
    return col;
}

float draw_fixed3_1dp(float2 uv, float2 org, float adv, float val_ms) {
    uint tt = (uint)round(clamp(val_ms, 0.0, 999.9) * 10.0);
    uint whole = tt / 10u;
    uint frac = tt % 10u;
    uint h = (whole / 100u) % 10u;
    uint t = (whole / 10u) % 10u;
    uint o = whole % 10u;
    float m = 0.0;
    if (h > 0u) {
        m = max(m, sample_hud_char(uv, org + float2(adv * 0.0, 0.0), h));
    }
    if (h > 0u || t > 0u) {
        m = max(m, sample_hud_char(uv, org + float2(adv * 1.0, 0.0), t));
    }
    m = max(m, sample_hud_char(uv, org + float2(adv * 2.0, 0.0), o));
    m = max(m, sample_hud_char(uv, org + float2(adv * 2.9, 0.0), 12u)); // '.'
    m = max(m, sample_hud_char(uv, org + float2(adv * 3.75, 0.0), frac));
    m = max(m, sample_hud_char(uv, org + float2(adv * 5.0, 0.0), 18u)); // 'M'
    m = max(m, sample_hud_char(uv, org + float2(adv * 6.0, 0.0), 15u)); // 'S'
    return m;
}

float draw_uma_row(float2 uv, float2 org, float adv, uint num_ch, uint c0, uint c1, uint c2, uint c3, uint c4, uint c5, bool is_opt, float val_ms) {
    float m = 0.0;
    m = max(m, sample_hud_char(uv, org + float2(adv * 0.0, 0.0), num_ch));
    m = max(m, sample_hud_char(uv, org + float2(adv * 1.0, 0.0), 11u)); // ':'
    m = max(m, sample_hud_char(uv, org + float2(adv * 2.0, 0.0), c0));
    m = max(m, sample_hud_char(uv, org + float2(adv * 3.0, 0.0), c1));
    m = max(m, sample_hud_char(uv, org + float2(adv * 4.0, 0.0), c2));
    m = max(m, sample_hud_char(uv, org + float2(adv * 5.0, 0.0), c3));
    m = max(m, sample_hud_char(uv, org + float2(adv * 6.0, 0.0), c4));
    m = max(m, sample_hud_char(uv, org + float2(adv * 7.0, 0.0), c5));
    if (is_opt) {
        m = max(m, sample_hud_char(uv, org + float2(adv * 9.0, 0.0), 32u));  // O
        m = max(m, sample_hud_char(uv, org + float2(adv * 10.0, 0.0), 14u)); // P
        m = max(m, sample_hud_char(uv, org + float2(adv * 11.0, 0.0), 34u)); // T
    } else {
        m = max(m, sample_hud_char(uv, org + float2(adv * 9.0, 0.0), 22u));  // N
        m = max(m, sample_hud_char(uv, org + float2(adv * 10.0, 0.0), 17u)); // A
        m = max(m, sample_hud_char(uv, org + float2(adv * 11.0, 0.0), 24u)); // I
        m = max(m, sample_hud_char(uv, org + float2(adv * 12.0, 0.0), 20u)); // V
    }
    m = max(m, draw_fixed3_1dp(uv, org + float2(adv * 14.5, 0.0), adv, val_ms));
    return m;
}

float3 composite_uma_hud(float3 bg, float2 uv) {
    // Top-left frosted HUD panel: x in [0.018, 0.300], y in [0.025, 0.255]
    static const float2 P_CENTER = float2(0.159, 0.140);
    static const float2 P_HALF   = float2(0.141, 0.115);
    float2 aspect = float2(1.7777778, 1.0);
    float sd_panel = sd_round_box((uv - P_CENTER) * aspect, P_HALF * aspect, 0.014);
    if (sd_panel > 0.004) {
        return bg;
    }

    uint mask = (uint)round(clamp(g_UmaMask, 0.0, 15.0));
    bool opt1 = (mask & 1u) != 0u;
    bool opt2 = (mask & 2u) != 0u;
    bool opt3 = (mask & 4u) != 0u;
    bool opt4 = (mask & 8u) != 0u;
    bool all_opt = (mask == 15u);

    float panel_alpha = smoothstep(0.002, -0.002, sd_panel);
    float border_mask = smoothstep(0.004, 0.001, abs(sd_panel));
    float3 col = lerp(bg, bg * 0.20 + float3(0.032, 0.045, 0.072), 0.86 * panel_alpha);
    float3 bcol = all_opt ? float3(0.22, 0.88, 0.62) : float3(0.96, 0.56, 0.22);
    col = lerp(col, bcol, 0.58 * border_mask);

    static const float ADV = 0.0068;
    static const float X0  = 0.032;
    bool is_pso = (g_UmaOn > 1.5);

    // Header: "UMA STRESS   U:ON" or "PSO STRESS   O:ON"
    float t_hdr = 0.0;
    float2 rh = float2(X0, 0.042);
    if (is_pso) {
        t_hdr = max(t_hdr, sample_hud_char(uv, rh + float2(ADV * 0.0, 0.0), 14u)); // P
        t_hdr = max(t_hdr, sample_hud_char(uv, rh + float2(ADV * 1.0, 0.0), 15u)); // S
        t_hdr = max(t_hdr, sample_hud_char(uv, rh + float2(ADV * 2.0, 0.0), 32u)); // O
    } else {
        t_hdr = max(t_hdr, sample_hud_char(uv, rh + float2(ADV * 0.0, 0.0), 27u)); // U
        t_hdr = max(t_hdr, sample_hud_char(uv, rh + float2(ADV * 1.0, 0.0), 18u)); // M
        t_hdr = max(t_hdr, sample_hud_char(uv, rh + float2(ADV * 2.0, 0.0), 17u)); // A
    }
    t_hdr = max(t_hdr, sample_hud_char(uv, rh + float2(ADV * 4.0, 0.0), 15u)); // S
    t_hdr = max(t_hdr, sample_hud_char(uv, rh + float2(ADV * 5.0, 0.0), 34u)); // T
    t_hdr = max(t_hdr, sample_hud_char(uv, rh + float2(ADV * 6.0, 0.0), 33u)); // R
    t_hdr = max(t_hdr, sample_hud_char(uv, rh + float2(ADV * 7.0, 0.0), 25u)); // E
    t_hdr = max(t_hdr, sample_hud_char(uv, rh + float2(ADV * 8.0, 0.0), 15u)); // S
    t_hdr = max(t_hdr, sample_hud_char(uv, rh + float2(ADV * 9.0, 0.0), 15u)); // S
    t_hdr = max(t_hdr, sample_hud_char(uv, rh + float2(ADV * 16.0, 0.0), is_pso ? 32u : 27u)); // O or U
    t_hdr = max(t_hdr, sample_hud_char(uv, rh + float2(ADV * 17.0, 0.0), 11u)); // :
    t_hdr = max(t_hdr, sample_hud_char(uv, rh + float2(ADV * 18.0, 0.0), 32u)); // O
    t_hdr = max(t_hdr, sample_hud_char(uv, rh + float2(ADV * 19.0, 0.0), 22u)); // N

    float t1 = is_pso
        ? draw_uma_row(uv, float2(X0, 0.076), ADV, 1u, 26u, 25u, 26u, 27u, 14u, 10u, opt1, g_UmaSub1Ms) // 1:DEDUP
        : draw_uma_row(uv, float2(X0, 0.076), ADV, 1u, 26u, 25u, 34u, 24u, 23u, 25u, opt1, g_UmaSub1Ms); // 1:DETILE
    float t2 = is_pso
        ? draw_uma_row(uv, float2(X0, 0.106), ADV, 2u, 32u, 14u, 34u, 39u, 32u, 1u,  opt2, g_UmaSub2Ms) // 2:OPT-O1
        : draw_uma_row(uv, float2(X0, 0.106), ADV, 2u, 14u, 17u, 34u, 16u, 28u, 10u, opt2, g_UmaSub2Ms); // 2:PATCH
    float t3 = is_pso
        ? draw_uma_row(uv, float2(X0, 0.136), ADV, 3u, 23u, 24u, 29u, 39u, 8u,  34u, opt3, g_UmaSub3Ms) // 3:LIB-8T
        : draw_uma_row(uv, float2(X0, 0.136), ADV, 3u, 16u, 18u, 26u, 28u, 36u, 26u, opt3, g_UmaSub3Ms); // 3:CMDHZD
    float t4 = is_pso
        ? draw_uma_row(uv, float2(X0, 0.166), ADV, 4u, 27u, 29u, 25u, 33u, 10u, 10u, opt4, g_UmaSub4Ms) // 4:UBER
        : draw_uma_row(uv, float2(X0, 0.166), ADV, 4u, 20u, 33u, 17u, 18u, 35u, 15u, opt4, g_UmaSub4Ms); // 4:VRAMWS

    // Summary Row: "TOT: NNN.N MS   OK:64/64" or "TOT: NNN.N MS  DRP:NN%"
    float t_sum = 0.0;
    float2 rs = float2(X0, 0.206);
    t_sum = max(t_sum, sample_hud_char(uv, rs + float2(ADV * 0.0, 0.0), 34u)); // T
    t_sum = max(t_sum, sample_hud_char(uv, rs + float2(ADV * 1.0, 0.0), 32u)); // O
    t_sum = max(t_sum, sample_hud_char(uv, rs + float2(ADV * 2.0, 0.0), 34u)); // T
    t_sum = max(t_sum, sample_hud_char(uv, rs + float2(ADV * 3.0, 0.0), 11u)); // :
    t_sum = max(t_sum, draw_fixed3_1dp(uv, rs + float2(ADV * 4.5, 0.0), ADV, g_UmaTotalMs));
    if (is_pso) {
        t_sum = max(t_sum, sample_hud_char(uv, rs + float2(ADV * 12.8, 0.0), 26u)); // D
        t_sum = max(t_sum, sample_hud_char(uv, rs + float2(ADV * 13.8, 0.0), 33u)); // R
        t_sum = max(t_sum, sample_hud_char(uv, rs + float2(ADV * 14.8, 0.0), 14u)); // P
        t_sum = max(t_sum, sample_hud_char(uv, rs + float2(ADV * 15.8, 0.0), 11u)); // :
        t_sum = max(t_sum, draw_uint3(uv, rs + float2(ADV * 16.8, 0.0), ADV, (uint)round(clamp(g_UmaOkMeshes, 0.0, 100.0))));
        t_sum = max(t_sum, sample_hud_char(uv, rs + float2(ADV * 19.6, 0.0), 38u)); // %
    } else {
        t_sum = max(t_sum, sample_hud_char(uv, rs + float2(ADV * 13.0, 0.0), 32u)); // O
        t_sum = max(t_sum, sample_hud_char(uv, rs + float2(ADV * 14.0, 0.0), 31u)); // K
        t_sum = max(t_sum, sample_hud_char(uv, rs + float2(ADV * 15.0, 0.0), 11u)); // :
        t_sum = max(t_sum, draw_uint3(uv, rs + float2(ADV * 16.0, 0.0), ADV, (uint)round(clamp(g_UmaOkMeshes, 0.0, 99.0))));
        t_sum = max(t_sum, sample_hud_char(uv, rs + float2(ADV * 18.2, 0.0), 37u)); // /
        t_sum = max(t_sum, sample_hud_char(uv, rs + float2(ADV * 19.2, 0.0), 6u));  // 6
        t_sum = max(t_sum, sample_hud_char(uv, rs + float2(ADV * 20.2, 0.0), 4u));  // 4
    }

    float3 opt_c = float3(0.32, 0.96, 0.68);
    float3 naiv_c = float3(1.0, 0.56, 0.28);
    col = lerp(col, float3(0.86, 0.96, 1.0), t_hdr);
    col = lerp(col, opt1 ? opt_c : naiv_c, t1);
    col = lerp(col, opt2 ? opt_c : naiv_c, t2);
    col = lerp(col, opt3 ? opt_c : naiv_c, t3);
    col = lerp(col, opt4 ? opt_c : naiv_c, t4);
    col = lerp(col, all_opt ? float3(0.45, 0.98, 0.82) : float3(1.0, 0.84, 0.36), t_sum);
    return col;
}

float3 eval_tonemapped(float2 uv, float autoExp) {
    float3 taa = g_TaaHDR.SampleLevel(g_LinClamp, uv, 0.0).rgb;
    float3 bloom = g_BloomTex.SampleLevel(g_LinClamp, uv, 0.0).rgb * g_BloomStrength;
    float3 hdr = (taa + bloom) * autoExp * g_ExposureScale;
    float3 mapped = agx_filmic(hdr);
    float2 d = uv - 0.5;
    float vig = 1.0 - dot(d, d) * 0.22;
    return saturate(mapped * vig);
}

float4 PSMain(CompPSIn input) : SV_Target {
    uint tw = 1u, th = 1u;
    g_TaaHDR.GetDimensions(tw, th);
    float2 inv = 1.0 / float2((float)tw, (float)th);
    float autoExp = clamp(g_ExposureBuf[0], 0.5, 1.5);

    float3 cC = eval_tonemapped(input.uv, autoExp);
    float3 cN = eval_tonemapped(input.uv + float2(0.0, -inv.y), autoExp);
    float3 cS = eval_tonemapped(input.uv + float2(0.0,  inv.y), autoExp);
    float3 cW = eval_tonemapped(input.uv + float2(-inv.x, 0.0), autoExp);
    float3 cE = eval_tonemapped(input.uv + float2( inv.x, 0.0), autoExp);

    // 5-Tap Contrast-Adaptive Sharpening (CAS) in perceptual space
    float lC = luma(cC);
    float lN = luma(cN);
    float lS = luma(cS);
    float lW = luma(cW);
    float lE = luma(cE);

    float minL = min(lC, min(min(lN, lS), min(lW, lE)));
    float maxL = max(lC, max(max(lN, lS), max(lW, lE)));
    float amp = sqrt(saturate(min(minL, 1.0 - maxL) / max(maxL, 1e-4)));
    float w = amp * (-0.125 * saturate(g_CasStrength));
    float3 sharpened = saturate((cC + (cN + cS + cW + cE) * w) / (1.0 + 4.0 * w));

    if (g_HudEnable > 0.5) {
        sharpened = composite_hud(sharpened, input.uv);
        if (g_PursuitOn > 0.5) {
            sharpened = composite_pursuit_hud(sharpened, input.uv);
        }
        if (g_UmaOn > 0.5) {
            sharpened = composite_uma_hud(sharpened, input.uv);
        }
    }

    return float4(sharpened, 1.0);
}
