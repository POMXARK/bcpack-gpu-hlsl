// Open AAA Showcase PBR + Normal Mapping + HDRI IBL + DXR 1.1 Inline RT/RTGI + BCPack Clay/X-Ray Heatmap
// Shader Model 6.5 Pixel Shader (DirectX Raytracing 1.1 RayQuery + NonUniformResourceIndex)

cbuffer AaaDrawCB : register(b0) {
    float4x4 g_MVP;
    float4x4 g_Model;
    float4   g_CameraPos;    // xyz = camera world pos, w = bitflags (1=RT, 2=RTGI, 4=XRay)
    float4   g_BaseColor;    // rgba
    float4   g_MatParams;    // x = metallic, y = roughness, z = uvScale, w = mode (0=glTF, 1=glass, 2=PolyHaven, 3=wetAsphalt)
    float4   g_EmissiveTime; // xyz = emissive, w = time
    uint4    g_TexIndices;   // x = albedoIdx, y = normalIdx, z = ormIdx, w = emissiveIdx (0xFFFFFFFF = none)
    float4   g_TileStates;   // x = albedoState, y = normalState, z = ormState (<0 clay, 0..1 fresh wave), w = stripCountY
};

Texture2D<float4>               g_Textures[32] : register(t0);
Texture2D<float4>               g_EnvHdr       : register(t32);
RaytracingAccelerationStructure g_SceneTlas    : register(t33);
StructuredBuffer<float4>        g_InstColors   : register(t34);

SamplerState g_LinearWrap : register(s0);
SamplerState g_EnvSampler : register(s1);

struct VSIn {
    float3 pos     : POSITION;
    float3 normal  : NORMAL;
    float4 tangent : TANGENT;
    float2 uv      : TEXCOORD0;
};

struct PSIn {
    float4 pos      : SV_Position;
    float3 worldN   : NORMAL;
    float4 worldT   : TANGENT;
    float2 uv       : TEXCOORD0;
    float3 worldPos : TEXCOORD1;
};

struct GBufferOut {
    float4 directHDR   : SV_Target0;
    float4 rawRTGI     : SV_Target1;
    float4 normalDepth : SV_Target2;
};

static const float PI = 3.14159265359;
static const uint NO_TEX = 0xFFFFFFFFu;

PSIn VSMain(VSIn input) {
    PSIn o;
    float4 wp = mul(g_Model, float4(input.pos, 1.0));
    o.pos = mul(g_MVP, float4(input.pos, 1.0));

    float3 wn = mul((float3x3)g_Model, input.normal);
    o.worldN = (dot(wn, wn) > 1e-6) ? normalize(wn) : float3(0.0, 1.0, 0.0);

    float3 wt = mul((float3x3)g_Model, input.tangent.xyz);
    float tLen = dot(wt, wt);
    o.worldT = (tLen > 1e-6) ? float4(wt / sqrt(tLen), input.tangent.w) : float4(1.0, 0.0, 0.0, 1.0);

    o.uv = input.uv * g_MatParams.z;
    o.worldPos = wp.xyz;
    return o;
}

float2 dir_to_equirect_uv(float3 d) {
    float phi = atan2(d.z, d.x);
    float u = frac(phi / 6.28318530718 + 0.5 - 0.115);
    float v = clamp(acos(clamp(d.y, -1.0, 1.0)) / PI, 0.002, 0.998);
    return float2(u, v);
}

float3 sample_env_lod(float3 dir, float roughness) {
    float3 d = (dot(dir, dir) > 1e-6) ? normalize(dir) : float3(0.0, 1.0, 0.0);
    float2 uv = dir_to_equirect_uv(d);
    if (roughness < 0.08) {
        return g_EnvHdr.SampleLevel(g_EnvSampler, uv, 0.0).rgb * 0.62;
    }
    // 5-tap directional blur across equirectangular HDRI approximating specular mip lobe
    float spread = roughness * roughness * 0.055;
    float3 c = g_EnvHdr.SampleLevel(g_EnvSampler, uv, 0.0).rgb * 0.36;
    c += g_EnvHdr.SampleLevel(g_EnvSampler, float2(frac(uv.x + spread), clamp(uv.y + spread * 0.5, 0.002, 0.998)), 0.0).rgb * 0.16;
    c += g_EnvHdr.SampleLevel(g_EnvSampler, float2(frac(uv.x - spread), clamp(uv.y + spread * 0.5, 0.002, 0.998)), 0.0).rgb * 0.16;
    c += g_EnvHdr.SampleLevel(g_EnvSampler, float2(frac(uv.x + spread), clamp(uv.y - spread * 0.5, 0.002, 0.998)), 0.0).rgb * 0.16;
    c += g_EnvHdr.SampleLevel(g_EnvSampler, float2(frac(uv.x - spread), clamp(uv.y - spread * 0.5, 0.002, 0.998)), 0.0).rgb * 0.16;
    return c * 0.62;
}

float3 sample_env_diffuse(float3 N) {
    float3 up = (abs(N.y) < 0.98) ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);
    float3 irr = sample_env_lod(N, 0.85) * 0.40;
    irr += sample_env_lod(normalize(N + T * 0.65), 0.95) * 0.15;
    irr += sample_env_lod(normalize(N - T * 0.65), 0.95) * 0.15;
    irr += sample_env_lod(normalize(N + B * 0.65), 0.95) * 0.15;
    irr += sample_env_lod(normalize(N - B * 0.65), 0.95) * 0.15;
    return irr;
}

float distribution_ggx(float ndh, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denom = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, 1e-5);
}

float geometry_schlick_ggx(float ndv, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return ndv / max(ndv * (1.0 - k) + k, 1e-5);
}

float geometry_smith(float ndv, float ndl, float roughness) {
    return geometry_schlick_ggx(ndv, roughness) * geometry_schlick_ggx(ndl, roughness);
}

float3 fresnel_schlick_roughness(float cosTheta, float3 F0, float roughness) {
    float3 maxVal = max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0);
    return F0 + (maxVal - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

float ign(float2 xy, float frame) {
    float2 p = xy + 5.588238 * fmod(frame, 64.0);
    return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
}

float hash21(float2 p) {
    p = frac(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return frac(p.x * p.y);
}

float noise2(float2 p) {
    float2 i = floor(p);
    float2 f = frac(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + float2(1.0, 0.0));
    float c = hash21(i + float2(0.0, 1.0));
    float d = hash21(i + float2(1.0, 1.0));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

float3 sample_rtgi_ray(float3 worldPos, float3 N, float3 T, float3 B, float r1, float r2) {
    float phi = 2.0 * PI * r1;
    float cosTheta = sqrt(r2);
    float sinTheta = sqrt(max(1.0 - r2, 0.0));
    float3 giRaw = T * (cos(phi) * sinTheta) + B * (sin(phi) * sinTheta) + N * cosTheta;
    float3 giDir = (dot(giRaw, giRaw) > 1e-6) ? normalize(giRaw) : float3(0.0, 1.0, 0.0);

    RayDesc gRay;
    gRay.Origin = worldPos + N * 0.04;
    gRay.Direction = giDir;
    gRay.TMin = 0.04;
    gRay.TMax = 45.0;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> gq;
    gq.TraceRayInline(g_SceneTlas, RAY_FLAG_NONE, 0xFFu, gRay);
    while (gq.Proceed()) {}

    if (gq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        float4 hd = g_InstColors[gq.CommittedInstanceID()];
        float fade = saturate(1.0 - gq.CommittedRayT() / 45.0);
        return hd.rgb * (0.40 + hd.a * 1.8) * fade;
    }
    return sample_env_lod(giDir, 0.85) * 0.48;
}

float3 trace_diffuse_rtgi(float3 worldPos, float3 N, float2 screenXY) {
    float3 upVec = (abs(N.z) < 0.999) ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 T = normalize(cross(upVec, N));
    float3 B = cross(N, T);
    // Static IGN + 2 stratified rays eliminate 60 Hz Monte Carlo boiling across walls & asphalt
    float r1a = ign(screenXY, 0.0);
    float r2a = ign(screenXY + float2(17.0, 31.0), 0.0) * 0.5;
    float r1b = frac(r1a + 0.5);
    float r2b = r2a + 0.5;
    return 0.5 * (sample_rtgi_ray(worldPos, N, T, B, r1a, r2a) + sample_rtgi_ray(worldPos, N, T, B, r1b, r2b));
}

GBufferOut PSMain(PSIn input) {
    float3 geomN = (dot(input.worldN, input.worldN) > 1e-6) ? normalize(input.worldN) : float3(0.0, 1.0, 0.0);
    float3 toCam = g_CameraPos.xyz - input.worldPos;
    float camDist = length(toCam);
    float3 V = (camDist > 1e-4) ? (toCam / camDist) : float3(0.0, 0.0, -1.0);

    uint flags = (uint)(g_CameraPos.w + 0.25);
    bool rtEnabled = (flags & 1u) != 0u;
    bool rtgiEnabled = (flags & 2u) != 0u;
    bool xrayEnabled = (flags & 4u) != 0u;

    uint matMode = (uint)(g_MatParams.w + 0.25); // 0=glTF, 1=glass, 2=PolyHaven, 3=wetAsphalt
    bool isGlass = (matMode == 1u);
    bool isPolyHaven = (matMode >= 2u);
    bool isWetAsphalt = (matMode == 3u);

    // 1. Sample 2K PBR textures gated by async GPU BCPack streaming residency
    bool hasAlbedoTex = (g_TexIndices.x != NO_TEX);
    bool albedoResident = !hasAlbedoTex || (g_TileStates.x >= 0.0);

    float3 albedo = g_BaseColor.rgb;
    if (hasAlbedoTex) {
        if (albedoResident) {
            float3 srgbTex = g_Textures[NonUniformResourceIndex(min(g_TexIndices.x, 31u))].Sample(g_LinearWrap, input.uv).rgb;
            albedo *= pow(max(srgbTex, 0.001), 2.2);
        } else {
            // Unstreamed clay fallback while VRAM is flushed / waving in
            albedo = float3(0.46, 0.45, 0.44);
        }
    }

    float metallic = albedoResident ? saturate(g_MatParams.x) : 0.0;
    float roughness = albedoResident ? clamp(g_MatParams.y, 0.04, 0.99) : 0.86;
    float ao = 1.0;

    if (albedoResident && g_TexIndices.z != NO_TEX && g_TileStates.z >= 0.0) {
        float3 orm = g_Textures[NonUniformResourceIndex(min(g_TexIndices.z, 31u))].Sample(g_LinearWrap, input.uv).rgb;
        if (isPolyHaven) {
            ao = lerp(0.35, 1.0, orm.r);
            roughness = clamp(orm.g * max(g_MatParams.y, 0.65), 0.05, 0.99);
            metallic = saturate(orm.b);
        } else {
            roughness = clamp(orm.g * g_MatParams.y, 0.04, 0.99);
            metallic = saturate(orm.b * g_MatParams.x);
        }
    }

    // 2. Tangent-space Normal Mapping
    float3 N = geomN;
    if (albedoResident && g_TexIndices.y != NO_TEX && g_TileStates.y >= 0.0) {
        float3 mapN = g_Textures[NonUniformResourceIndex(min(g_TexIndices.y, 31u))].Sample(g_LinearWrap, input.uv).xyz;
        float2 xy = mapN.xy * 2.0 - 1.0;
        if (!isPolyHaven) {
            // glTF normal maps use +Y green; flip Y and attenuate subtle metallic flake / interior bump
            xy.y = -xy.y;
            if (!hasAlbedoTex) {
                xy *= 0.06;
            } else {
                xy *= 0.65;
            }
        }
        float z = sqrt(saturate(1.0 - dot(xy, xy)));
        float3 tN = float3(xy, z);

        float3 T = input.worldT.xyz - geomN * dot(geomN, input.worldT.xyz);
        if (dot(T, T) > 1e-6) {
            T = normalize(T);
            float3 B = cross(geomN, T) * ((input.worldT.w < 0.0) ? -1.0 : 1.0);
            N = normalize(T * tN.x + B * tN.y + geomN * tN.z);
        }
    }

    // Wet coastal asphalt puddles (low-frequency puddles + crisp normal flattening)
    float puddle = 0.0;
    if (albedoResident && isWetAsphalt) {
        float n1 = noise2(input.worldPos.xz * 0.14);
        float n2 = noise2(input.worldPos.xz * 0.42 + 7.3);
        puddle = smoothstep(0.46, 0.66, n1 * 0.65 + n2 * 0.35);
        float laneWet = smoothstep(14.0, 1.5, abs(input.worldPos.x));
        puddle = saturate(puddle * (0.72 + 0.45 * laneWet));
        N = normalize(lerp(N, geomN, puddle * 0.94));
        roughness = lerp(roughness, 0.065, puddle);
        albedo *= lerp(1.0, 0.50, puddle);
    }

    // Geometric / Normal Specular Anti-Aliasing (Kaplanyan/Toksvig screen-space derivative filter)
    float3 dNdx = ddx(N);
    float3 dNdy = ddy(N);
    float nVar = dot(dNdx, dNdx) + dot(dNdy, dNdy);
    roughness = sqrt(saturate(roughness * roughness + min(nVar * 0.75, 0.25)));

    // 3. HDR Sunset Directional Light matching 2K Sunset HDRI peak azimuth
    float3 sunDir = normalize(float3(0.28, 0.16, 0.94));
    float ndl = max(dot(N, sunDir), 0.0);
    float ndv = max(dot(N, V), 1e-4);

    // Hardware DXR 1.1 Inline Ray Traced Sun Shadow
    float sunVis = 1.0;
    if (rtEnabled && ndl > 0.01) {
        RayDesc sRay;
        sRay.Origin = input.worldPos + geomN * 0.035;
        sRay.Direction = sunDir;
        sRay.TMin = 0.03;
        sRay.TMax = 140.0;
        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
        sq.TraceRayInline(g_SceneTlas, RAY_FLAG_NONE, 0xFFu, sRay);
        while (sq.Proceed()) {}
        if (sq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            sunVis = 0.16;
        }
    }

    float3 sunRadiance = float3(1.85, 1.12, 0.60) * sunVis;

    // 4. Cook-Torrance GGX Direct BRDF
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
    if (isGlass) {
        albedo = float3(0.015, 0.018, 0.022);
        metallic = 0.0;
        roughness = 0.04;
        F0 = float3(0.10, 0.11, 0.12);
    }

    float3 H = normalize(V + sunDir);
    float ndh = max(dot(N, H), 0.0);
    float hdv = max(dot(H, V), 0.0);
    float D = distribution_ggx(ndh, roughness);
    float G = geometry_smith(ndv, ndl, roughness);
    float3 F = fresnel_schlick_roughness(hdv, F0, roughness);

    float3 specular = min((D * G * F) / max(4.0 * ndv * ndl, 0.001), float3(6.0, 6.0, 6.0));
    float3 kD = (1.0 - F) * (1.0 - metallic);
    float3 directLit = (kD * albedo / PI + specular) * sunRadiance * ndl;

    // Key warm rim / bounce fill from coastal promenade lights
    float3 fillDir = normalize(float3(-0.65, 0.45, -0.55));
    float fillNdl = max(dot(N, fillDir), 0.0);
    directLit += (kD * albedo / PI) * float3(0.32, 0.42, 0.64) * fillNdl;

    // 5. HDRI Equirectangular IBL + Hardware DXR 1.1 Glossy Reflections
    float3 F_ibl = fresnel_schlick_roughness(ndv, F0, roughness);
    float3 kD_ibl = (1.0 - F_ibl) * (1.0 - metallic);
    float3 diffuseIBL = sample_env_diffuse(N) * albedo * kD_ibl * ao;

    float3 reflDir = reflect(-V, N);
    float3 specEnv = min(sample_env_lod(reflDir, roughness), float3(5.0, 5.0, 5.0));

    if (rtEnabled && (roughness < 0.58 || isGlass || puddle > 0.25)) {
        RayDesc rRay;
        rRay.Origin = input.worldPos + geomN * 0.035;
        rRay.Direction = (dot(reflDir, reflDir) > 1e-6) ? normalize(reflDir) : float3(0.0, 1.0, 0.0);
        rRay.TMin = 0.03;
        rRay.TMax = 100.0;
        RayQuery<RAY_FLAG_FORCE_OPAQUE> rq;
        rq.TraceRayInline(g_SceneTlas, RAY_FLAG_NONE, 0xFFu, rRay);
        while (rq.Proceed()) {}

        if (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            float4 hitData = g_InstColors[rq.CommittedInstanceID()];
            float hitFade = saturate(1.0 - rq.CommittedRayT() / 90.0);
            float glossWeight = saturate(1.0 - roughness * 1.45);
            float3 hitColor = hitData.rgb * (0.75 + hitData.a * 1.65);
            specEnv = lerp(specEnv, hitColor, hitFade * glossWeight * 0.85);
        }
    }

    float3 specularIBL = specEnv * F_ibl * lerp(0.55, 1.0, ao);
    if (isGlass) {
        // Automotive smoked glass: deep obsidian interior + crisp HDRI & DXR Fresnel mirror reflections
        float glassFresnel = 0.12 + 0.86 * pow(1.0 - ndv, 4.0);
        directLit = directLit * 0.25 + float3(0.012, 0.014, 0.018) + specEnv * glassFresnel;
        diffuseIBL = 0.0;
        specularIBL = 0.0;
    }

    float3 emissiveCol = g_EmissiveTime.xyz;
    if (albedoResident && g_TexIndices.w != NO_TEX) {
        float3 emTex = g_Textures[NonUniformResourceIndex(min(g_TexIndices.w, 31u))].Sample(g_LinearWrap, input.uv).rgb;
        emissiveCol *= pow(max(emTex, 0.001), 2.2);
    }

    float3 hdr = directLit + diffuseIBL + specularIBL + emissiveCol * 1.8;

    // Subtle warm exponential distance fog near seawall horizon
    float fog = 1.0 - exp(-max(camDist - 55.0, 0.0) * 0.0065);
    float3 fogCol = float3(0.82, 0.44, 0.30);
    hdr = lerp(hdr, fogCol, saturate(fog * 0.65));

    // 6. 1-spp Diffuse RTGI Bounce
    float3 gi = float3(0.0, 0.0, 0.0);
    if (rtgiEnabled && camDist < 120.0 && !isGlass) {
        gi = trace_diffuse_rtgi(input.worldPos, N, input.pos.xy) * albedo * (1.0 - metallic * 0.7) * 0.45;
    }

    // 7. [V] X-Ray BCPack 4K-Tile Heatmap Overlay
    if (xrayEnabled) {
        float strips = max(g_TileStates.w, 16.0);
        float vCoord = frac(input.uv.y);
        float2 tileCoord = float2(frac(input.uv.x) * 8.0, vCoord * strips);
        float2 tileFrac = frac(tileCoord);
        float edge = (tileFrac.x < 0.04 || tileFrac.y < 0.08) ? 1.0 : 0.0;
        float effState = g_TileStates.x;
        if (!hasAlbedoTex) {
            effState = max(g_TileStates.y, g_TileStates.z);
        }
        if (effState < 0.0) {
            // Unstreamed Clay zone
            float3 clayHeat = float3(0.95, 0.16, 0.48);
            hdr = lerp(hdr * 0.22, clayHeat, 0.42 + edge * 0.48);
        } else if (effState < 1.0) {
            // Partially streamed texture: reveal streamed top strips vs active cyan wavefront vs bottom clay strips
            if (vCoord > effState) {
                float3 clayHeat = float3(0.95, 0.16, 0.48);
                hdr = lerp(hdr * 0.22, clayHeat, 0.42 + edge * 0.48);
            } else if (abs(vCoord - effState) < 0.10) {
                float3 waveCol = float3(0.15, 1.55, 1.75);
                hdr = lerp(hdr * 0.30, waveCol, 0.65 + edge * 0.35);
            } else {
                float3 resCol = float3(0.06, 0.65, 0.28);
                hdr = lerp(hdr * 0.35, resCol, 0.36 + edge * 0.24);
            }
        } else {
            // Fully streamed texture (fresh flash > 1.0 fades down to calm emerald resident grid)
            float fresh = saturate(effState - 1.0);
            float3 resCol = float3(0.06, 0.65, 0.28);
            float3 freshCol = float3(0.15, 1.45, 1.65);
            float3 heat = lerp(resCol, freshCol, fresh * 0.65);
            hdr = lerp(hdr * 0.38, heat, 0.32 + fresh * 0.22 + edge * 0.25);
        }
    }

    GBufferOut o;
    o.directHDR = float4(hdr, 1.0);
    o.rawRTGI = float4(gi, 1.0);
    o.normalDepth = float4(geomN, camDist);
    return o;
}
