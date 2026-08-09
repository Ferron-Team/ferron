// Split-sum pieces shared by the forward pass and the screen-space reflection
// composite.
//
// They are in a header rather than duplicated because the composite's whole job
// is to *replace* the environment term the forward pass applied: it subtracts
// what that pass added and adds the traced radiance in its place, weighted
// identically. Two copies of this arithmetic that drifted by a factor would
// show up as reflective surfaces getting brighter or darker the moment a ray
// hits, which reads as a lighting bug rather than as a mismatch.

#ifndef ORRIN_BRDF_GLSL
#define ORRIN_BRDF_GLSL

// Split-sum environment BRDF (scale, bias), Karis' analytic fit from the 2014
// mobile PBR notes. A stand-in for the DFG lookup table image-based lighting
// will bring; swap both callers to the table when it exists.
vec2 env_brdf_approx(float perceptual_roughness, float n_dot_v) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = perceptual_roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    return vec2(-1.04, 1.04) * a004 + r.zw;
}

// Multiple-scattering compensation (Kulla & Conty). Single-scatter GGX drops
// the energy that would have bounced between microfacets, so rough metals go
// dark and desaturated; scaling the specular lobe by 1 + f0*(1/Ess - 1) puts it
// back, where Ess is the single-scatter directional albedo.
vec3 energy_compensation(vec3 f0, float perceptual_roughness, float n_dot_v) {
    vec2 dfg = env_brdf_approx(perceptual_roughness, n_dot_v);
    // Ess is the split-sum result for a fully reflective surface (f0 = 1). It
    // tends to 1 as roughness falls, so this stays near 1 for smooth materials.
    float ess = max(dfg.x + dfg.y, 1e-3);
    return 1.0 + f0 * (1.0 / ess - 1.0);
}

#endif
