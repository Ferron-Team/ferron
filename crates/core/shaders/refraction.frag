#version 460

// Screen-space refraction, the draw half.
//
// A refractive surface is not a blended one. Weighted-blended transparency mixes
// with whatever the framebuffer holds under a commutative weight, which is what
// lets it skip the sort; this pass *fetches* what is behind the surface, bends
// the lookup by the surface's own normal and index of refraction, and returns
// the result already composited. Two surfaces doing that over the same pixel are
// not interchangeable — the far one has to be resolved before the near one can
// stand in front of it — so this queue is sorted back to front and the
// transparent one still must not be.
//
// The output is premultiplied, so the blend that gathers overlapping surfaces
// here and the composite that puts the result over the lit frame are the same
// `over` operator applied twice rather than two different ones.

#define ORRIN_REFRACTIVE 1
#include "shading.glsl"

layout(location = 0) out vec4 f_color;

void main() {
    vec4 surface = shade_surface();

    // Contributes nothing, and a zero-coverage fragment leaves the premultiplied
    // blend below an exact no-op anyway — the same early-out `oit.frag` makes.
    if (surface.a <= 0.0) {
        discard;
    }

    f_color = vec4(surface.rgb * surface.a, surface.a);
}
