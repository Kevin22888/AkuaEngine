#version 330 core

in vec4 fragColor;

out vec4 FragColor;

void main() {
    vec2 offset = gl_PointCoord - vec2(0.5);
    float dist = length(offset);

    if (dist > 0.5) {
        discard;
    }

    float alpha = exp(-dist * dist * 3.0);

    FragColor = fragColor * alpha;
}