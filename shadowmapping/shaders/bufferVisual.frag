#version 450
layout (binding = 2) uniform sampler2D samplerColor;
layout (location = 0) in vec2 inUV;

float LinearizeDepth(float depth)
{
    float n = 1.0; // camera z near
    float f = 128.0; // camera z far
    float z = depth;
    return (2.0 * n) / (f + n - z * (f - n));
}

layout(location = 0) out vec4 outColor;

void main() {
    float depth = texture(samplerColor, inUV).r;
    outColor = vec4(vec3(1.0-depth), 1.0);
}
