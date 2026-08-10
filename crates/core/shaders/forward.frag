#version 460

// Opaque shading. Everything this draws with lives in `shading.glsl`, which
// `oit.frag` includes too — see there for why one definition and not two.
#include "shading.glsl"

layout(location = 0) out vec4 f_color;

void main() {
    // The opacity a material carries is what the transparent pass weights with.
    // Nothing blends here, so it is dropped rather than written into an alpha
    // channel the resolve would ignore anyway.
    f_color = vec4(shade_surface().rgb, 1.0);
}
