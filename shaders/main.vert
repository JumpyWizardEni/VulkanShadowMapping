#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inTexCoord;

layout (binding = 0) uniform UBO
{
    mat4 model;
    mat4 view;
    mat4 projection;
    mat4 lightSpace;
    vec3 lightPos;
} ubo;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec2 outTexCoord;
layout (location = 2) out vec3 outViewVec;
layout (location = 3) out vec3 outLightVec;
layout (location = 4) out vec4 outShadowCoord;

out gl_PerVertex
{
    vec4 gl_Position;
};

const mat4 biasMat = mat4(
0.5, 0.0, 0.0, 0.0,
0.0, 0.5, 0.0, 0.0,
0.0, 0.0, 1.0, 0.0,
0.5, 0.5, 0.0, 1.0 );

void main()
{
    outTexCoord = inTexCoord;
    outNormal = inNormal;

    gl_Position = ubo.projection * ubo.view * ubo.model * vec4(inPos.xyz, 1.0);

    vec4 pos = ubo.model * vec4(inPos, 1.0);
    outNormal = mat3(ubo.model) * inNormal;
    outLightVec = normalize(ubo.lightPos - inPos);
    outViewVec = -pos.xyz;

    outShadowCoord = ( biasMat * ubo.lightSpace * ubo.model ) * vec4(inPos, 1.0);
}

