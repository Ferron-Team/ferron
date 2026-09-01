#version 460

// Opaque shading, in the frame that diffuses subsurface light across the image.
//
// A separate file rather than a branch, for the reason `oit.frag` and
// `refraction.frag` are separate files: the difference is one `#define` and what
// the outputs are, and everything about what a surface *looks like* stays in the
// one place all four include. Which of the two opaque variants runs is a property
// of the frame's structure, so the pipeline is chosen once when the graph is
// compiled rather than tested per fragment.
#define ORRIN_SUBSURFACE 1
#include "shading.glsl"

layout(location = 0) out vec4 f_color;
// `rgb` = the radiance that entered the medium and came back out somewhere else,
// `a` = how far across the surface, in metres, it is allowed to have moved.
//
// Both in one target because the width is exactly a per-pixel property of the
// radiance beside it, and because a zero there is the only mask the blur and the
// composite need: this pass is the only writer, the attachment is cleared, and
// every other pipeline sharing this render pass masks the channel off. So a pixel
// nothing scattering covered reads zero and is left alone.
layout(location = 1) out vec4 f_subsurface;

void main() {
    Shaded shaded = shade_surface();
    // Coverage, as in `forward.frag`. Alpha to coverage reads the *first*
    // attachment's alpha and applies the result to every attachment, so a leaf's
    // scattered light is carved out by the same samples its colour is — which is
    // what keeps the diffusion from spreading radiance out of a texel the cutout
    // removed.
    float coverage = mask_coverage(shaded.alpha);
#ifdef ORRIN_ALPHA_TEST
    // The hard cut `forward.frag` documents. Discarding rather than masking the
    // coverage keeps the two targets in step for free: a discarded fragment
    // writes neither, which is what alpha to coverage was doing for both at four
    // samples.
    if (coverage < 0.5) {
        discard;
    }
#endif
    f_color = vec4(shaded.color, coverage);
    f_subsurface = vec4(shaded.diffusible, shaded.scatter);
}
