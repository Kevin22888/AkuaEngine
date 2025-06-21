#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in float aSize;
layout (location = 2) in vec4 aColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 cameraPosition;

out vec4 fragColor;

void main() {
    vec4 worldPosition = model * vec4(aPos, 1.0);
    float distance = length(worldPosition.xyz - cameraPosition);
    gl_PointSize = aSize / distance;

    gl_Position = projection * view * worldPosition;
    fragColor = aColor;
}