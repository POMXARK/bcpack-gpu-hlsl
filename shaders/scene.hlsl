cbuffer DrawCB : register(b0) {
    float4x4 g_MVP;
    float4x4 g_Model;
    uint     g_TexIndex;
    float    g_Emissive;
    float    g_Time;
    float    g_UvScale;
    float4   g_CameraPos;
    float4   g_PointLights[4]; // xyz = world pos, w = intensity
    float4   g_LightColors[4]; // rgb = color, a = specular power
};

Texture2D    g_Textures[12] : register(t0);
SamplerState g_Sampler      : register(s0);

struct VSIn {
    float3 pos    : POSITION;
    float3 normal : NORMAL;
    float2 uv     : TEXCOORD0;
};

struct PSIn {
    float4 pos      : SV_Position;
    float3 worldN   : NORMAL;
    float2 uv       : TEXCOORD0;
    float3 worldPos : TEXCOORD1;
};

PSIn VSMain(VSIn input) {
    PSIn o;
    float4 wp = mul(g_Model, float4(input.pos, 1.0));
    o.pos = mul(g_MVP, float4(input.pos, 1.0));
    o.worldN = normalize(mul((float3x3)g_Model, input.normal));
    o.uv = input.uv * g_UvScale;
    if (g_TexIndex == 4u) {
        // Animated ocean wave scroll
        o.uv += float2(g_Time * 0.08, g_Time * 0.04);
    }
    o.worldPos = wp.xyz;
    return o;
}

float4 PSMain(PSIn input) : SV_Target {
    float3 albedo = g_Textures[NonUniformResourceIndex(g_TexIndex)].Sample(g_Sampler, input.uv).rgb;
    float3 N = normalize(input.worldN);
    float3 V = normalize(g_CameraPos.xyz - input.worldPos);

    // Warm Vice-City sunset directional light + twilight hemisphere ambient
    float3 sunDir = normalize(float3(0.08, 0.24, 0.96));
    float ndl = max(dot(N, sunDir), 0.0);
    float3 sunColor = float3(1.15, 0.68, 0.44);
    float3 ambient = lerp(float3(0.14, 0.10, 0.26), float3(0.32, 0.26, 0.48), N.y * 0.5 + 0.5);

    // Blinn-Phong specular highlight (wet highway sheen, ocean sun path, glossy sports cars)
    float specPower = g_LightColors[0].a;
    float3 H = normalize(sunDir + V);
    float spec = pow(max(dot(N, H), 0.0), specPower) * (g_TexIndex == 0u || g_TexIndex == 4u || g_TexIndex == 8u || g_TexIndex == 9u ? 0.65 : 0.15);

    float3 lit = albedo * (ambient + sunColor * ndl) + sunColor * spec;

    // 4 dynamic point lights (streetlamps + hero car headlights/taillights)
    [unroll]
    for (int i = 0; i < 4; ++i) {
        float3 toL = g_PointLights[i].xyz - input.worldPos;
        float d2 = dot(toL, toL) + 1.0;
        float att = g_PointLights[i].w / (1.0 + 0.055 * d2);
        float3 L = normalize(toL);
        float diff = max(dot(N, L), 0.0);
        float3 HL = normalize(L + V);
        float pspec = pow(max(dot(N, HL), 0.0), 32.0) * 0.35;
        lit += (albedo * diff + pspec) * g_LightColors[i].rgb * att;
    }

    lit += albedo * g_Emissive;

    // Warm sunset distance haze blending smoothly into sky horizon
    float dist = length(input.worldPos.xz - g_CameraPos.xz);
    float fog = saturate((dist - 40.0) / 115.0);
    float3 fogColor = float3(0.68, 0.28, 0.42);
    float3 finalCol = lerp(lit, fogColor, fog * 0.72);
    return float4(finalCol, 1.0);
}
