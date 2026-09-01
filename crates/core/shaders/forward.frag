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
    // The alpha channel is coverage, not opacity. Nothing blends in this pass,
    // so on the plain pipeline the value is ignored and one is what a fully
    // covered pixel means; on the `Masked` pipeline alpha to coverage is on and
    // this is what carves the cutout out of the four samples.
    float coverage = mask_coverage(shaded.alpha);
#ifdef ORRIN_ALPHA_TEST
    // One sample, so there is no coverage to spend and the ramp above collapses
    // to the hard test it is a smoothing of. Half coverage is alpha exactly at
    // the material's cutoff, so cutting here cuts along the same line
    // `prepass.frag` cut along -- and that is not a nicety but the requirement:
    // this pipeline depth-tests `EQUAL` against the depth that pass wrote, and a
    // fragment the prepass discarded has no depth here for anything to match.
    if (coverage < 0.5) {
        discard;
    }
#endif
    f_color = vec4(shaded.color + shaded.diffusible, coverage);
}
