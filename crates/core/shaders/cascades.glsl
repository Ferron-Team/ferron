// The cascaded shadow lookup, once.
//
// Included by `shading.glsl`, which asks it what a *surface* can see, and by
// `fog_scatter.comp`, which asks it what a point of *air* can see. The file
// exists for the reason `brdf.glsl` and `parallax.glsl` do: a shaft of light in
// the air and the ground it lands on are the same visibility question, and two
// copies that drifted would read as a lighting bug rather than as a mismatch —
// the shaft would miss the shadow it belongs to by a bias.
//
// Nothing here declares a binding. The cascade constants arrive as a [`Cascades`]
// value out of whichever uniform block the includer owns, and the maps and their
// comparison sampler arrive as parameters, so the two callers can keep them at
// different sets and bindings.

#ifndef ORRIN_CASCADES_GLSL
#define ORRIN_CASCADES_GLSL

// Keep in sync with MAX_CASCADES in gfx/shadows.rs.
const int MAX_CASCADES = 4;

// Laid out to match the same four fields in `GpuLighting` and `GpuFog`, in that
// order. std140 gives a struct of `mat4[4]` and three `vec4`s the same offsets it
// gives those fields written out flat, which is what lets one Rust struct feed a
// shader that nests them and another that does not.
struct Cascades {
    mat4 view_proj[MAX_CASCADES];
    // Per-cascade far distance, radial from the camera.
    vec4 splits;
    // World size of one shadow texel, per cascade. It differs per cascade
    // because each fits a different-sized box to the same number of texels.
    vec4 texel_sizes;
    // x = count, y = blend overlap fraction, z = strength, w = debug tint.
    vec4 params;
};

// The cascade a point is routed to: the first whose far distance it is nearer
// than. Distance is radial rather than view-space depth, which costs a little
// over-coverage at the frustum corners and buys rotation invariance — the same
// property the sphere fit on the CPU side is built around.
int select_cascade(Cascades c, float view_dist) {
    int count = int(c.params.x);
    for (int i = 0; i < count; ++i) {
        if (view_dist < c.splits[i]) {
            return i;
        }
    }
    return count - 1;
}

// Percentage-closer filtered visibility from one cascade. 1.0 is lit.
//
// `N` is the direction the lookup is biased along, which is the surface normal
// for a surface and the light direction itself for a point of air — where there
// is no surface to be shadow-acned against, and pushing toward the light is
// exactly the offset that keeps a froxel out of its own occluder.
float cascade_shadow(
    Cascades c,
    texture2DArray maps,
    samplerShadow cmp,
    int cascade,
    vec3 world_pos,
    vec3 N,
    vec3 L
) {
    // Normal-offset bias: move the lookup along `N` by about a texel's worth of
    // world space, more at grazing angles where a texel covers the most depth.
    // Offsetting in texture space rather than in depth is what removes acne
    // without the peter-panning a depth offset causes.
    float texel = c.texel_sizes[cascade];
    float slope = 1.0 - max(dot(N, L), 0.0);
    vec3 p = world_pos + N * texel * 1.4142136 * (1.0 + slope);

    vec4 clip = c.view_proj[cascade] * vec4(p, 1.0);
    vec3 ndc = clip.xyz / clip.w;
    // Past the cascade's far plane there is nothing to occlude against.
    if (ndc.z > 1.0) {
        return 1.0;
    }

    vec2 uv = ndc.xy * 0.5 + 0.5;
    vec2 step = 1.0 / vec2(textureSize(sampler2DArrayShadow(maps, cmp), 0).xy);

    // 3x3 taps. Each is itself a hardware 2x2 comparison — the compare happens
    // before the bilinear filter — so this is effectively a 4x4 kernel.
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 offset = vec2(x, y) * step;
            sum += texture(
                sampler2DArrayShadow(maps, cmp),
                vec4(uv + offset, float(cascade), ndc.z)
            );
        }
    }
    return sum / 9.0;
}

// Sun visibility, blended across the cascade seam and scaled by the strength
// slider. The early-outs a *surface* can take — facing away from the sun, or the
// slider at zero — belong to the caller, because only one of the two callers has
// a normal to test.
float cascade_sun_shadow(
    Cascades c,
    texture2DArray maps,
    samplerShadow cmp,
    vec3 world_pos,
    vec3 N,
    vec3 L,
    float view_dist
) {
    int count = int(c.params.x);
    if (count <= 0) {
        return 1.0;
    }

    int cascade = select_cascade(c, view_dist);
    float shadow = cascade_shadow(c, maps, cmp, cascade, world_pos, N, L);

    // The next cascade only has depth slightly before its own near plane, and
    // that overlap is what `cascades()` widened each slice by — so the blend
    // band has to be the same fraction or it fades into a region with no data.
    float split = c.splits[cascade];
    float band = split * c.params.y;
    if (cascade + 1 < count && view_dist > split - band) {
        float t = clamp((view_dist - (split - band)) / max(band, 1e-4), 0.0, 1.0);
        shadow = mix(shadow, cascade_shadow(c, maps, cmp, cascade + 1, world_pos, N, L), t);
    }

    return mix(1.0, shadow, c.params.z);
}

#endif // ORRIN_CASCADES_GLSL
