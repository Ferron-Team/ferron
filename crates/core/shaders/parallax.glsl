// Parallax occlusion mapping: the texture coordinate a view ray actually lands
// on once the surface is treated as a height field rather than as a plane.
//
// Shared by `shading.glsl` and `prepass.frag` for the reason `brdf.glsl` is
// shared. Both passes sample the same maps at the same pixel, and the prepass's
// normal, f0 and roughness are what the reflection resolve reads back; two
// copies of this march that drifted would put the G-buffer one brick edge away
// from the surface the forward pass shaded, which reads as a reflection bug
// rather than as a mismatch.
//
// The includer supplies the sampler, because the two passes bind their texture
// array to different sets under different names. Define, *before* including
// this file:
//
//     float parallax_height(uint index, vec2 uv, vec2 dx, vec2 dy)
//
// returning the map's height in [0, 1] with 1 at the top of the field. It takes
// explicit gradients because the march below is a loop with a per-fragment trip
// count: an implicit derivative inside non-uniform control flow is undefined,
// and picking mip 0 instead would alias a tiled wall into noise at the exact
// distance the effect is meant to be cheap at.
//
// What this does not do: move the depth buffer. The silhouette of a
// parallaxed surface is still the silhouette of its geometry, and everything
// screen-space — occlusion, reflections, contact shadows, the shadow maps —
// sees the flat surface. That is the whole reason it costs one loop in two
// fragment shaders and nothing anywhere else in the frame.

#ifndef ORRIN_PARALLAX_GLSL
#define ORRIN_PARALLAX_GLSL

// Hard ceiling on the loop, so the compiler can bound it. A material's own step
// counts are clamped to this on the CPU — keep in sync with `PARALLAX_STEP_LIMIT`
// in forward.rs.
const int PARALLAX_STEP_LIMIT = 64;

// Where the field fades out, as cosines of the angle between the ray and the
// surface normal: full relief by `PARALLAX_FADE_FULL` (about 73 degrees off the
// normal), gone by `PARALLAX_FADE_NONE` (about 84).
//
// A fade rather than a clamp on the quotient, and the difference is the whole
// quality of the effect edge-on. The in-plane travel per unit depth is
// `view.xy / view.z`, which runs away as the surface turns edge-on; clamping the
// divisor bounds it but still applies a near-maximum offset to a surface that is
// almost in the view ray, and the result is courses that shear into diagonal
// smears. The offset is not what is wrong there — at 83 degrees a five-centimetre
// groove really does displace most of a brick — it is that displacement without
// *occlusion* is only half of what a real groove does, and the missing half is
// exactly the silhouette a height field has no way to cut. So the honest answer
// where the march cannot be right is to hand back the flat surface, which is what
// this does. Fading also halves the worst-case offset the march ever produces,
// since the peak of `fade / cos` sits at the top of the ramp rather than at its
// end.
const float PARALLAX_FADE_FULL = 0.3;
const float PARALLAX_FADE_NONE = 0.1;

/// How far the texture coordinate moves per metre travelled across the surface,
/// as a 2x2 map from the tangent plane's (T, B) axes into (u, v).
///
/// The depth is authored in metres because that is what the number *is* — a
/// 15 mm mortar groove — and because the UV answer is not a property of the
/// material at all: the same brick map over a wall tiled once and a wall tiled
/// eight times would need two different UV depths for one physical groove. So
/// the conversion is measured here, per fragment, from the pixel's own
/// footprint, and retiling the wall or scaling the mesh leaves the groove the
/// depth it was authored at.
///
/// A full Jacobian rather than one ratio, because the common cases are not
/// square: a wall UV-mapped 0..1 across six metres and three carries twice the
/// texels per metre vertically, and a single scale factor would split the
/// difference and be wrong on both axes. This also carries any rotation the UV
/// layout has relative to the tangent, which is why the march can be handed a
/// direction in metres and get back the offset in the coordinates the maps are
/// actually in.
mat2 parallax_uv_per_metre(vec3 T, vec3 B, vec2 duv_dx, vec2 duv_dy, vec3 dpos_dx, vec3 dpos_dy) {
    // Both derivatives resolved into the tangent plane, so the two matrices are
    // expressed over the same pair of screen axes and one inverts against the
    // other. The out-of-plane component is dropped, which is exact: the height
    // field lives in this plane.
    mat2 metres = mat2(
        vec2(dot(dpos_dx, T), dot(dpos_dx, B)),
        vec2(dot(dpos_dy, T), dot(dpos_dy, B))
    );
    // Near-zero at a silhouette, where the neighbouring fragment belongs to
    // another triangle, and on geometry seen exactly edge-on. Both give an
    // inverse that means nothing, and a zero map is the "no relief here" answer
    // the march already handles.
    if (abs(determinant(metres)) < 1e-12) {
        return mat2(0.0);
    }
    return mat2(duv_dx, duv_dy) * inverse(metres);
}

/// March the view ray through the height field and return where it first goes
/// below the surface.
///
/// `view_ts` points *toward* the eye in tangent space — the same basis the
/// normal map is decoded through, so a map authored for one is authored for the
/// other. The ray enters at the top of the field, which is why every sample is
/// `1 - height`: the map says how high a texel stands, and the march is
/// measuring how far down it has come.
///
/// Nothing is clipped at the UV edges. A tiling detail map is the entire point
/// of this, and it must be free to walk into the next repeat.
vec2 parallax_occlusion(
    uint height_tex,
    vec2 uv,
    vec3 view_ts,
    float depth_metres,
    mat2 uv_per_metre,
    float min_steps,
    float max_steps,
    vec2 dx,
    vec2 dy
) {
    float cosine = clamp(view_ts.z, 0.0, 1.0);
    // Flat where the march would smear rather than occlude; see the constants.
    float depth = depth_metres * smoothstep(PARALLAX_FADE_NONE, PARALLAX_FADE_FULL, cosine);

    // Where the ray comes out the bottom of the field, relative to where it went
    // in. Negative because it travels away from the eye as it descends, and
    // divided by the cosine because a shallower entry crosses more ground for
    // the same depth.
    vec2 full = uv_per_metre
        * (-view_ts.xy / max(cosine, PARALLAX_FADE_NONE) * depth);

    // A quarter of the map in one crossing is already past what the march can
    // resolve at any step count; beyond it the surface reads as a smear rather
    // than as relief, so the offset is capped instead of allowed to run. Zero
    // here covers a degenerate footprint, an unauthored depth and a surface
    // exactly edge-on, all of which want the coordinate left alone.
    float span = length(full);
    if (span < 1e-9) {
        return uv;
    }
    full *= min(span, 0.25) / span;

    // Steps scale with how steeply the ray enters: head-on it crosses the field
    // in almost no lateral distance and a handful of layers resolve it, while at
    // a glancing angle the same field is smeared over many texels and a coarse
    // march reports the first wall it happens to land on. Spending the samples
    // where the geometry is stretched is what keeps the cost off the pixels that
    // do not need it.
    float steps = clamp(mix(max_steps, min_steps, cosine), 1.0, float(PARALLAX_STEP_LIMIT));

    float layer = 1.0 / steps;
    vec2 delta = full * layer;

    vec2 current = uv;
    float ray = 0.0;
    float surface = 1.0 - parallax_height(height_tex, current, dx, dy);
    float previous = surface;

    for (int i = 0; i < PARALLAX_STEP_LIMIT; ++i) {
        if (ray >= surface) {
            break;
        }
        previous = surface;
        current += delta;
        ray += layer;
        surface = 1.0 - parallax_height(height_tex, current, dx, dy);
    }

    // One secant step between the last two samples. The march brackets the
    // intersection; without this the result snaps to a layer boundary, and a
    // sloped brick face turns into the staircase the layers were.
    //
    // `previous` is carried out of the loop rather than resampled: the height at
    // the step before is already known, and a texture fetch to learn it again is
    // a sixteenth of this function's cost.
    float after = surface - ray;
    float before = previous - (ray - layer);
    float denominator = after - before;
    float weight = abs(denominator) < 1e-6 ? 0.0 : after / denominator;
    return mix(current, current - delta, clamp(weight, 0.0, 1.0));
}

#endif
