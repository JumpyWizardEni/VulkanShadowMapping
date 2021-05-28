#version 450

layout (binding = 1) uniform sampler2D texSampler;
layout (binding = 2) uniform sampler2D shadowMap;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inTexCoord;
layout (location = 2) in vec3 inViewVec;
layout (location = 3) in vec3 inLightVec;
layout (location = 4) in vec4 inShadowCoord;


layout (location = 0) out vec4 outFragColor;

#define ambient 0.1

float textureProj(vec4 shadowCoord, vec2 off)
{
    float shadow = 1.0;
    if ( shadowCoord.z > -1.0 && shadowCoord.z < 1.0 )
    {
        float dist = texture( shadowMap, shadowCoord.st + off ).r;
        if ( shadowCoord.w > 0.0 && dist < shadowCoord.z )
        {
            shadow = ambient;
        }
    }
    return shadow;
}

float filterPCF(vec4 sc)
{
    ivec2 texDim = textureSize(shadowMap, 0);
    float scale = 1.5;
    float dx = scale * 1.0 / float(texDim.x);
    float dy = scale * 1.0 / float(texDim.y);

    float shadowFactor = 0.0;
    int count = 0;
    int range = 1;

    for (int x = -range; x <= range; x++)
    {
        for (int y = -range; y <= range; y++)
        {
            shadowFactor += textureProj(sc, vec2(dx*x, dy*y));
            count++;
        }

    }
    return shadowFactor / count;
}

void main()
{
    vec3 lightColor = vec3(1, 1, 1);
    float shadow = filterPCF(inShadowCoord / inShadowCoord.w);
    vec3 N = normalize(inNormal);
    vec3 L = normalize(inLightVec);
    vec3 V = normalize(inViewVec);
    vec3 R = normalize(-reflect(L, N));

    vec3 halfwayDir = normalize(L + V);
    float spec = pow(max(dot(N, halfwayDir), 0.0), 64);
    vec3 specular = lightColor * spec;


    vec3 ambientLight = lightColor * ambient;

    vec3 diffuse = max(dot(N, L), ambient) * lightColor;


    vec3 result = ((diffuse + spec) * shadow + ambientLight) * vec3(texture(texSampler, inTexCoord));
    outFragColor = vec4(result, 1.0);


}
