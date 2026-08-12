#version 460

// Opaque shading. Everything this draws with lives in `shading.glsl`, which
// `oit.frag` includes too — see there for why one definition and not two.
#include "shading.glsl"

layout(location = 0) out vec4 f_color;

void main() {
    Shaded shaded = shade_surface();
    // Summed, because this variant of the pass has one target: with no diffusion
    // downstream there is nothing to keep the two halves apart for, and
    // `shading.glsl` has already put the wrapped diffuse in the second one to
    // stand in for the pass that is not running.
    //
    // The opacity a material carries is what the transparent pass weights with.
    // Nothing blends here, so it is dropped rather than written into an alpha
    // channel the resolve would ignore anyway.
    f_color = vec4(shaded.color + shaded.diffusible, 1.0);
}
