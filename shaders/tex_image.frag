#version 450

layout (location = 0) vec3 inColor;
layout (location = 1) vec2 inUV;

layout (location = 0) vec4 outColor;

layout (binding = 0, set = 0) uniform sampler2D displayTexture;

void main() {
	outColor = texture(displayTexture, inUV);
}
