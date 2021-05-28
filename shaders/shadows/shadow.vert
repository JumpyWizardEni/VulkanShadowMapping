#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 lightMatrix;
    vec3 lightPos;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;

layout (location = 0) out vec4 color;

out gl_PerVertex
{
    vec4 gl_Position;
};

void main()
{
    gl_Position = ubo.lightMatrix * vec4(inPosition, 1.0);
    vec4 position = ubo.lightMatrix * vec4(inPosition, 1.0);
    float z = position.z;
    color = vec4(0.25, 0.53, 0.1, 1.0);
}