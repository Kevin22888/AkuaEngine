#version 330 core

in vec2 TexCoord;
out vec4 FragColor;

uniform vec3 color1; // Light gray
uniform vec3 color2; // Dark gray
uniform float scale; // Checker size

void main() {
    // Scale the texture coordinates for the checker pattern
    vec2 scaledCoords = TexCoord * scale;

    // Determine which color to use based on the integer part of the coordinates
    float check = mod(floor(scaledCoords.x) + floor(scaledCoords.y), 2.0);
    vec3 color = mix(color1, color2, check);

    FragColor = vec4(color, 1.0);
}