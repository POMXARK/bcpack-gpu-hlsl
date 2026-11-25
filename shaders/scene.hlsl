cbuffer DrawCB : register(b0) {
    float4x4 g_MVP;
    float4x4 g_Model;
    uint     g_TexIndex;
    float    g_Emissive;
    float    g_Time;
    float    g_UvScale;
};

Texture2D    g_Textures[8] : register(t0);
SamplerState g_Sampler     : register(s0);

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
    o.worldPos = wp.xyz;
    return o;
}

float4 PSMain(PSIn input) : SV_Target {
    float3 albedo = g_Textures[NonUniformResourceIndex(g_TexIndex)].Sample(g_Sampler, input.uv).rgb;
    float3 N = normalize(input.worldN);

    // Warm Vice-City sunset directional light + cool cyan sky ambient
    float3 sunDir = normalize(float3(0.45, 0.35, -0.82));
    float ndl = max(dot(N, sunDir), 0.0);
    float3 sunColor = float3(1.05, 0.72, 0.55);
    float3 ambient = lerp(float3(0.16, 0.12, 0.28), float3(0.32, 0.38, 0.58), N.y * 0.5 + 0.5);

    // Subtle distance fog blending into sunset horizon
    float dist = length(input.worldPos.xz);
    float fog = saturate((dist - 22.0) / 55.0);
    float3 fogColor = float3(0.48, 0.22, 0.44);

    float3 lit = albedo * (ambient + sunColor * ndl) + albedo * g_Emissive;
    float3 finalCol = lerp(lit, fogColor, fog * 0.55);
    return float4(finalCol, 1.0);
}
