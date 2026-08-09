#version 460

layout(location = 0) in vec3 v_world_pos;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec3 v_tangent;
layout(location = 3) in vec3 v_bitangent;
layout(location = 4) in vec2 v_uv;
layout(location = 5) in vec3 v_color;

layout(location = 0) out vec4 f_color;

// Keep in sync with MAX_POINT_LIGHTS / MAX_SPOT_LIGHTS / MAX_TEXTURES in
// forward.rs.
const int MAX_POINT_LIGHTS = 16;
const int MAX_SPOT_LIGHTS = 8;
const int MAX_TEXTURES = 64;
// Keep in sync with MAX_CASCADES in gfx/shadows.rs.
const int MAX_CASCADES = 4;
const float PI = 3.14159265359;

struct PointLight {
    vec4 position; // xyz = world position, w = range
    vec4 color;    // rgb = color,         w = intensity
    vec4 shadow;   // x = first atlas face (< 0 = none), y = near plane
};

struct SpotLight {
    vec4 position;  // xyz = world position, w = range
    vec4 direction; // xyz = cone axis,      w = cos(outer half angle)
    vec4 color;     // rgb = color,          w = intensity
    vec4 params;    // x = cos(inner), y = atlas face (< 0 = none), z = near
};

layout(set = 0, binding = 0) uniform Lighting {
    vec4 camera_pos;    // xyz = camera world position
    vec4 ambient;       // rgb = color, w = intensity
    vec4 sun_direction; // xyz = direction toward the sun (normalized)
    vec4 sun_color;     // rgb = color, w = intensity
    vec4 params;        // x = point light count (y,z legacy), w = spot count
    vec4 viewport;      // x=w, y=h, z=1/w, w=1/h
    vec4 fog_color;     // rgb = color, w = density at the reference height
    vec4 fog_params;    // x = height falloff, y = reference height
    mat4 cascade_view_proj[MAX_CASCADES];
    vec4 cascade_splits;      // per-cascade far distance, radial from the camera
    vec4 cascade_texel_sizes; // world size of one shadow texel, per cascade
    vec4 shadow_params;       // x = count, y = blend overlap, z = strength, w = debug
    PointLight point_lights[MAX_POINT_LIGHTS];
    SpotLight spot_lights[MAX_SPOT_LIGHTS];
    vec4 environment;    // x = sin(env yaw), y = cos(env yaw)
    vec4 env_specular;   // rgb = multiplier on sampled environment radiance
    vec4 irradiance[9];  // rgb = SH coefficient; see gfx/sh.rs
} lighting;

// Mirrors GpuMaterial in forward.rs. std430 packs this exactly like
// the Rust #[repr(C)] struct because every field is 16 bytes.
struct GpuMaterial {
    vec4 base_color;   // rgb = albedo
    vec4 emissive;     // rgb = emissive
    vec4 params;       // x = metallic, y = roughness, z = reflectance
    uvec4 tex_indices; // x=albedo, y=normal, z=metal-rough, w=emissive
};

// Material table indexed by the per-draw material_index. A storage
// buffer so the array can be sized at runtime (one entry per material).
layout(set = 1, binding = 0, std430) readonly buffer Materials {
    GpuMaterial materials[];
};

// Textures are kept separate from the sampler: Metal/MoltenVK allows
// only 16 sampler states per stage but many sampled images, so a
// combined sampler2D[64] would blow the sampler limit. One shared
// sampler + an array of texture2D stays well under it.
layout(set = 2, binding = 0) uniform texture2D textures[MAX_TEXTURES];
layout(set = 2, binding = 1) uniform sampler tex_sampler;

// Screen-space ambient occlusion (blurred), sampled by screen-space UV.
layout(set = 3, binding = 0) uniform sampler2D u_ao;

// The cascade depth maps, one array layer each, and the comparison sampler
// they are read through. Separated for the same reason the texture array above
// is: Metal allows far fewer samplers per stage than sampled images.
layout(set = 3, binding = 1) uniform texture2DArray u_shadow_maps;
layout(set = 3, binding = 2) uniform samplerShadow u_shadow_cmp;

// Prefiltered specular environment: roughness increases with mip level. When no
// environment is loaded this is a 1x1 white cube and `env_specular` carries the
// scene's flat ambient, so there is one path here rather than two.
// Keep in sync with SPECULAR_MIPS in gfx/vulkan/environment.rs.
const float SPECULAR_MIPS = 6.0;
layout(set = 3, binding = 3) uniform textureCube u_environment;
layout(set = 3, binding = 4) uniform sampler u_environment_sampler;

// The sun's visibility over the short range a cascade texel cannot resolve,
// sampled by screen-space UV like `u_ao`. A 1x1 white view when the pass is off,
// so there is one path here rather than two.
layout(set = 3, binding = 5) uniform sampler2D u_contact_shadow;

// The punctual shadow atlas, and the table saying where in it each light's
// faces landed. Read through `u_shadow_cmp`, the cascades' comparison sampler:
// same `Less` against a map cleared to 1.0, so one sampler serves both and the
// per-stage sampler budget Metal imposes stays untouched.
layout(set = 3, binding = 6) uniform texture2D u_shadow_atlas;

struct ShadowFace {
    mat4 view_proj;
    vec4 rect; // xy = atlas uv offset, zw = atlas uv scale
};
layout(set = 3, binding = 7, std430) readonly buffer ShadowFaces {
    ShadowFace shadow_faces[];
};

// Index is dynamically uniform (from the material), so plain indexing
// is legal without the nonuniform qualifier.
vec4 sample_tex(uint index, vec2 uv) {
    return texture(sampler2D(textures[index], tex_sampler), uv);
}

// Declared identically to the vertex shader so the stages share one
// push-constant range; only material_index is read here.
layout(push_constant) uniform Push {
    mat4 view_proj;
    uint material_index;
    uint object_base;
} push;

// --- Cook-Torrance terms (metallic-roughness workflow) ---

// GGX / Trowbridge-Reitz normal distribution.
float distribution_ggx(float n_dot_h, float a) {
    float a2 = a * a;
    float d = (n_dot_h * n_dot_h) * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

// Smith height-correlated visibility (already folds in the 1/(4 NoL NoV) denom).
float visibility_smith_ggx(float n_dot_v, float n_dot_l, float a) {
    float a2 = a * a;
    float gv = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    float gl = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-5);
}

// Fresnel-Schlick reflectance.
vec3 fresnel_schlick(float v_dot_h, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - v_dot_h, 0.0, 1.0), 5.0);
}

// Shared with ssr_resolve.comp, which subtracts exactly the environment term
// this pass adds. See brdf.glsl for why that has to be one definition.
#include "brdf.glsl"

// Specular antialiasing: widen linear roughness `a` to cover the normal
// variance inside this pixel, which the single-sample BRDF below cannot see.
// MSAA cannot do this — it supersamples coverage, not shader inputs.
// sigma = 0.5 px, the standard deviation of the pixel filter kernel in image
// space; the shader wants its square.
const float SPEC_AA_SIGMA2 = 0.25;
// Diffuse irradiance from the environment, divided by pi so the result is
// already the diffuse response for unit albedo.
//
// The nine terms are the real spherical-harmonic basis for bands 0..=2, and
// they must match `basis` in gfx/sh.rs term for term and component for
// component — the coefficients were projected against that one. A mismatch is
// not a compile error, it is lighting that is quietly rotated or mirrored.
//
// With no environment loaded these coefficients carry the scene's flat ambient
// in band 0 alone, which evaluates to that ambient for every normal. So there
// is one path here, not two.
// World space into the environment's own frame. The same rotation the skybox
// samples through, so what is drawn behind the scene and what lights it cannot
// disagree.
vec3 to_environment(vec3 v) {
    float s = lighting.environment.x;
    float c = lighting.environment.y;
    return vec3(c * v.x - s * v.z, v.y, s * v.x + c * v.z);
}

vec3 sh_irradiance(vec3 n) {
    n = to_environment(n);

    vec3 e = lighting.irradiance[0].rgb * 0.282095
           + lighting.irradiance[1].rgb * (0.488603 * n.y)
           + lighting.irradiance[2].rgb * (0.488603 * n.z)
           + lighting.irradiance[3].rgb * (0.488603 * n.x)
           + lighting.irradiance[4].rgb * (1.092548 * n.x * n.y)
           + lighting.irradiance[5].rgb * (1.092548 * n.y * n.z)
           + lighting.irradiance[6].rgb * (0.315392 * (3.0 * n.z * n.z - 1.0))
           + lighting.irradiance[7].rgb * (1.092548 * n.x * n.z)
           + lighting.irradiance[8].rgb * (0.546274 * (n.x * n.x - n.y * n.y));

    // Backstop for the ringing the band window in gfx/sh.rs is sized to
    // contain: a truncated series overshoots at a sun disc and undershoots
    // opposite it, and the undershoot can cross zero.
    return max(e, vec3(0.0));
}

// Specular image-based lighting, Karis' split sum: the prefiltered chain is the
// light integral, `env_brdf_approx` the BRDF integral. This is the term that
// gives a metal anything to be — its diffuse response is zero by definition, so
// without it a metal is lit only where it happens to mirror a light.
//
// `energy` is the same multi-scatter compensation the analytic lights get, and
// it belongs here for the same reason: single-scatter GGX loses energy with
// roughness whether the light is a sun or a sky.
vec3 specular_ibl(vec3 N, vec3 V, vec3 f0, float perceptual_roughness, vec3 energy) {
    vec3 R = to_environment(reflect(-V, N));
    float lod = perceptual_roughness * (SPECULAR_MIPS - 1.0);
    vec3 radiance = textureLod(samplerCube(u_environment, u_environment_sampler), R, lod).rgb;

    vec2 dfg = env_brdf_approx(perceptual_roughness, max(dot(N, V), 1e-4));
    return radiance * lighting.env_specular.rgb * (f0 * dfg.x + dfg.y) * energy;
}

// Clamping threshold, from Kaplanyan et al. 2016.
const float SPEC_AA_KAPPA = 0.18;

// Specular antialiasing, Tokuyoshi & Kaplanyan 2019 listing 2: widen the linear
// roughness to cover the normal variance inside this pixel, which the
// single-sample BRDF below cannot see. MSAA does not help — it supersamples
// coverage, not shader inputs. Filtering off the normal rather than the
// halfvector costs the same regardless of light count and needs no tangent
// frame, so it holds up on meshes with degenerate tangents.
float filter_roughness(float a, vec3 N) {
    vec3 dndu = dFdx(N);
    vec3 dndv = dFdy(N);
    float variance = SPEC_AA_SIGMA2 * (dot(dndu, dndu) + dot(dndv, dndv));
    // Dropping the 2.0 gives the paper's less conservative variant: less
    // overfiltering, at the risk of underfiltering.
    float kernel_roughness2 = min(2.0 * variance, SPEC_AA_KAPPA);
    // The paper works in squared roughness, so square, widen, and root back.
    return sqrt(clamp(a * a + kernel_roughness2, 0.0, 1.0));
}

// Outgoing radiance toward the camera from one light direction L.
// Takes linear roughness `a` rather than perceptual: the caller filters it once
// for specular antialiasing, and this runs once per light.
vec3 brdf(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo,
          float metallic, float a, vec3 f0, vec3 energy) {
    float n_dot_l = max(dot(N, L), 0.0);
    if (n_dot_l <= 0.0) {
        return vec3(0.0);
    }
    vec3 H = normalize(L + V);
    float n_dot_v = max(dot(N, V), 1e-4);
    float n_dot_h = max(dot(N, H), 0.0);
    float v_dot_h = max(dot(V, H), 0.0);

    float D = distribution_ggx(n_dot_h, a);
    float Vis = visibility_smith_ggx(n_dot_v, n_dot_l, a);
    vec3 F = fresnel_schlick(v_dot_h, f0);

    vec3 specular = D * Vis * F * energy;

    // Diffuse keeps the energy not reflected (1 - F) and not metallic.
    vec3 kd = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kd * albedo / PI;

    return (diffuse + specular) * radiance * n_dot_l;
}

// Exponential height fog. Density decays with altitude, so the amount along a
// view ray is the integral of that decay rather than a function of distance
// alone — which is what keeps a ray climbing out of the layer from fogging as
// heavily as one running through it.
vec3 apply_fog(vec3 color, vec3 world_pos, vec3 camera_pos) {
    float density = lighting.fog_color.w;
    if (density <= 0.0) {
        return color;
    }

    float falloff = lighting.fog_params.x;
    vec3 ray = world_pos - camera_pos;
    float dist = length(ray);
    float dir_y = ray.y / max(dist, 1e-4);

    float at_camera = exp(-falloff * (camera_pos.y - lighting.fog_params.y));

    // The quotient has a removable singularity for rays with no vertical
    // component, where the integral is just the ray length.
    float t = falloff * dir_y * dist;
    float integral = abs(t) > 1e-4 ? (1.0 - exp(-t)) / (falloff * dir_y) : dist;

    float amount = clamp(1.0 - exp(-density * at_camera * integral), 0.0, 1.0);
    return mix(color, lighting.fog_color.rgb, amount);
}

// --- Cascaded shadow maps ---

// The cascade this fragment is routed to: the first whose far distance it is
// nearer than. Distance is radial rather than view-space depth, which costs a
// little over-coverage at the frustum corners and buys rotation invariance —
// the same property the sphere fit on the CPU side is built around.
int select_cascade(float view_dist) {
    int count = int(lighting.shadow_params.x);
    for (int i = 0; i < count; ++i) {
        if (view_dist < lighting.cascade_splits[i]) {
            return i;
        }
    }
    return count - 1;
}

// Percentage-closer filtered visibility from one cascade. 1.0 is lit.
float cascade_shadow(int cascade, vec3 world_pos, vec3 N, vec3 L) {
    // Normal-offset bias: move the lookup along the surface normal by about a
    // texel's worth of world space, more at grazing angles where a texel covers
    // the most depth. Offsetting in texture space rather than in depth is what
    // removes acne without the peter-panning a depth offset causes.
    float texel = lighting.cascade_texel_sizes[cascade];
    float slope = 1.0 - max(dot(N, L), 0.0);
    vec3 p = world_pos + N * texel * 1.4142136 * (1.0 + slope);

    vec4 clip = lighting.cascade_view_proj[cascade] * vec4(p, 1.0);
    vec3 ndc = clip.xyz / clip.w;
    // Past the cascade's far plane there is nothing to occlude against.
    if (ndc.z > 1.0) {
        return 1.0;
    }

    vec2 uv = ndc.xy * 0.5 + 0.5;
    vec2 step = 1.0 / vec2(textureSize(sampler2DArrayShadow(u_shadow_maps, u_shadow_cmp), 0).xy);

    // 3x3 taps. Each is itself a hardware 2x2 comparison — the compare happens
    // before the bilinear filter — so this is effectively a 4x4 kernel.
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 offset = vec2(x, y) * step;
            sum += texture(
                sampler2DArrayShadow(u_shadow_maps, u_shadow_cmp),
                vec4(uv + offset, float(cascade), ndc.z)
            );
        }
    }
    return sum / 9.0;
}

// Sun visibility, blended across the cascade seam.
float sun_shadow(vec3 world_pos, vec3 N, vec3 L, float view_dist) {
    int count = int(lighting.shadow_params.x);
    if (count <= 0) {
        return 1.0;
    }

    // A surface facing away from the sun is already unlit by it — `brdf`
    // returns zero for n_dot_l <= 0, so whatever the maps say gets multiplied
    // into nothing. Leaving before the lookup saves the nine taps this cascade
    // would cost, and the eighteen a fragment in the blend band would.
    if (dot(N, L) <= 0.0) {
        return 1.0;
    }

    // Strength zero is the "shadows off" slider, and the blend below collapses
    // to a constant 1.0 at it. Same taps saved, for a setting rather than for
    // the geometry.
    if (lighting.shadow_params.z <= 0.0) {
        return 1.0;
    }

    int cascade = select_cascade(view_dist);
    float shadow = cascade_shadow(cascade, world_pos, N, L);

    // The next cascade only has depth slightly before its own near plane, and
    // that overlap is what `cascades()` widened each slice by — so the blend
    // band has to be the same fraction or it fades into a region with no data.
    float split = lighting.cascade_splits[cascade];
    float band = split * lighting.shadow_params.y;
    if (cascade + 1 < count && view_dist > split - band) {
        float t = clamp((view_dist - (split - band)) / max(band, 1e-4), 0.0, 1.0);
        shadow = mix(shadow, cascade_shadow(cascade + 1, world_pos, N, L), t);
    }

    return mix(1.0, shadow, lighting.shadow_params.z);
}

// Distinct tint per cascade, for checking that the splits land where intended.
vec3 cascade_debug_tint(int cascade) {
    if (cascade == 0) return vec3(1.0, 0.4, 0.4);
    if (cascade == 1) return vec3(0.4, 1.0, 0.4);
    if (cascade == 2) return vec3(0.4, 0.6, 1.0);
    return vec3(1.0, 1.0, 0.4);
}

// Smooth, range-limited falloff (windowed inverse-square).
float attenuate(float dist, float range) {
    float s = dist / max(range, 1e-4);
    if (s >= 1.0) return 0.0;
    float window = 1.0 - s * s;
    return (window * window) / max(dist * dist, 1e-4);
}

// --- Punctual shadows: one atlas, a tile per face ---

// Which cube face a direction belongs to, in the order `gfx/punctual.rs` builds
// them: +X, -X, +Y, -Y, +Z, -Z. That order is the contract between the two — the
// matrices carry every other convention, so this is the only thing that has to
// agree.
int cube_face(vec3 d) {
    vec3 a = abs(d);
    if (a.x >= a.y && a.x >= a.z) return d.x > 0.0 ? 0 : 1;
    if (a.y >= a.z)               return d.y > 0.0 ? 2 : 3;
    return d.z > 0.0 ? 4 : 5;
}

// Percentage-closer visibility from one atlas face. 1.0 is lit.
//
// The taps are clamped inside the tile, which the cascades do not have to do:
// there, a tap past the edge hits the sampler's white border and reads as lit,
// while here it would land in a *neighbouring light's* depth and read as
// whatever that light happened to see. The border the faces are rendered with
// is what keeps that clamp reading real geometry rather than a repeated edge.
float atlas_shadow(int face_index, vec3 world_pos, vec3 N, vec3 L, float dist) {
    ShadowFace face = shadow_faces[face_index];

    // Normal-offset bias, as the cascades do it, but scaled by how much world
    // space a texel covers *here*: a perspective face's texel grows with
    // distance from the light, so a constant offset is either useless near it or
    // peter-panning far from it. `rect.z` is the tile's share of the atlas, so
    // the divisor is that tile's texel count along an edge.
    vec2 atlas_size = vec2(textureSize(sampler2DShadow(u_shadow_atlas, u_shadow_cmp), 0));
    float texel_world = 2.0 * dist / max(face.rect.z * atlas_size.x, 1.0);
    float slope = 1.0 - max(dot(N, L), 0.0);
    vec4 clip = face.view_proj * vec4(world_pos + N * texel_world * 1.4142136 * (1.0 + slope), 1.0);
    if (clip.w <= 0.0) {
        return 1.0;
    }

    vec3 ndc = clip.xyz / clip.w;
    // Past the light's own far plane there is nothing recorded to occlude
    // against, and the attenuation has already taken the light to zero there.
    if (ndc.z > 1.0) {
        return 1.0;
    }

    vec2 tile_uv = ndc.xy * 0.5 + 0.5;
    vec2 texel = 1.0 / atlas_size;
    // Half a texel in from the tile's edge, so a bilinear fetch cannot reach
    // past it either.
    vec2 lo = face.rect.xy + texel * 0.5;
    vec2 hi = face.rect.xy + face.rect.zw - texel * 0.5;

    float sum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 uv = face.rect.xy + (tile_uv + vec2(x, y) * texel / face.rect.zw) * face.rect.zw;
            sum += texture(
                sampler2DShadow(u_shadow_atlas, u_shadow_cmp),
                vec3(clamp(uv, lo, hi), ndc.z)
            );
        }
    }
    return sum / 9.0;
}

// Visibility for a point light: pick the face its direction falls on, then the
// lookup above. `first_face` is negative for a light the atlas had no room for,
// which is the unshadowed path.
float point_shadow(PointLight light, vec3 world_pos, vec3 N, vec3 L, float dist) {
    int first_face = int(light.shadow.x);
    if (first_face < 0) {
        return 1.0;
    }
    // The direction *from* the light, which is what the faces were built around.
    int face = cube_face(world_pos - light.position.xyz);
    return atlas_shadow(first_face + face, world_pos, N, L, dist);
}

// The cone's own falloff: full inside the inner angle, out to nothing at the
// outer one. Smoothstepped rather than linear so the edge of a spot does not
// read as a hard ring on a flat floor.
float cone_falloff(SpotLight light, vec3 L) {
    // `L` points at the light and the axis points away from it, hence the
    // negation — the same relationship `sun_direction` has to the sun.
    float cosine = dot(-L, light.direction.xyz);
    float outer = light.direction.w;
    float inner = light.params.x;
    return smoothstep(outer, max(inner, outer + 1e-4), cosine);
}

void main() {
    GpuMaterial m = materials[push.material_index];

    // Sample the maps. Missing maps point at the default textures, so
    // these multiplies become no-ops. Albedo/emissive images are sRGB
    // (decoded to linear on sample); metal-rough is linear data.
    vec3 albedo_tex = sample_tex(m.tex_indices.x, v_uv).rgb;
    vec4 mr_tex     = sample_tex(m.tex_indices.z, v_uv);
    vec3 emis_tex   = sample_tex(m.tex_indices.w, v_uv).rgb;

    // Vertex color tints the material albedo; drop `* v_color` for a
    // pure material/texture color.
    vec3  albedo      = m.base_color.rgb * v_color * albedo_tex;
    // glTF metallic-roughness convention: G = roughness, B = metallic.
    float metallic    = clamp(m.params.x * mr_tex.b, 0.0, 1.0);
    float roughness   = clamp(m.params.y * mr_tex.g, 0.04, 1.0); // floor avoids a singular highlight
    float reflectance = m.params.z;

    // Dielectric F0 from reflectance (0.5 -> ~4%); metals use albedo as F0.
    vec3 f0 = mix(vec3(0.16 * reflectance * reflectance), albedo, metallic);

    // Tangent-space normal map -> world space via the TBN basis.
    vec3 n_tangent = sample_tex(m.tex_indices.y, v_uv).xyz * 2.0 - 1.0;
    mat3 TBN = mat3(normalize(v_tangent), normalize(v_bitangent), normalize(v_normal));
    vec3 N = normalize(TBN * n_tangent);

    float a = filter_roughness(roughness * roughness, N); // perceptual -> linear

    vec3 V = normalize(lighting.camera_pos.xyz - v_world_pos);

    // Both are constant across lights, so they are computed once here rather
    // than once per light inside brdf().
    vec3 energy = energy_compensation(f0, roughness, max(dot(N, V), 1e-4));

    // Diffuse image-based lighting, attenuated by screen-space ambient
    // occlusion. The (1 - metallic) is the same factor `brdf` applies to its
    // own diffuse lobe: a metal has no diffuse response, and until the
    // prefiltered specular chain lands it has no environment response at all.
    float ao = texture(u_ao, gl_FragCoord.xy * lighting.viewport.zw).r;
    vec3 color = sh_irradiance(N) * albedo * (1.0 - metallic) * ao;
    color += specular_ibl(N, V, f0, roughness, energy) * ao;

    // Directional sun, shadowed by the cascades and the screen-space march.
    // The environment term above is what `u_ao` attenuates instead; the
    // punctual lights below carry their own atlas lookups.
    float view_dist = length(v_world_pos - lighting.camera_pos.xyz);
    {
        vec3 L = normalize(lighting.sun_direction.xyz);
        vec3 radiance = lighting.sun_color.rgb * lighting.sun_color.w;
        // The cascades and the screen-space march answer the same question at
        // two scales, so their answers multiply: the maps carry everything past
        // a texel's width of the contact, and the march carries the band inside
        // it that a map's own bias reports lit.
        float shadow = sun_shadow(v_world_pos, N, L, view_dist)
                     * texture(u_contact_shadow, gl_FragCoord.xy * lighting.viewport.zw).r;
        color += brdf(N, V, L, radiance, albedo, metallic, a, f0, energy) * shadow;
    }

    // Point lights.
    int count = int(lighting.params.x);
    for (int i = 0; i < count; ++i) {
        PointLight light = lighting.point_lights[i];
        vec3 to_light = light.position.xyz - v_world_pos;
        float dist = length(to_light);
        float atten = attenuate(dist, light.position.w);
        if (atten <= 0.0) continue;
        vec3 L = to_light / max(dist, 1e-4);
        // Before the atlas lookup, not after: a surface facing away is already
        // unlit by this light, and the nine taps would be multiplied into a
        // zero. The same early-out `sun_shadow` makes for the cascades.
        if (dot(N, L) <= 0.0) continue;
        vec3 radiance = light.color.rgb * light.color.w * atten;
        float shadow = point_shadow(light, v_world_pos, N, L, dist);
        color += brdf(N, V, L, radiance, albedo, metallic, a, f0, energy) * shadow;
    }

    // Spot lights: a point light with a cone over it, and one atlas face rather
    // than six because a cone is a single frustum.
    int spot_count = int(lighting.params.w);
    for (int i = 0; i < spot_count; ++i) {
        SpotLight light = lighting.spot_lights[i];
        vec3 to_light = light.position.xyz - v_world_pos;
        float dist = length(to_light);
        float atten = attenuate(dist, light.position.w);
        if (atten <= 0.0) continue;
        vec3 L = to_light / max(dist, 1e-4);
        if (dot(N, L) <= 0.0) continue;
        atten *= cone_falloff(light, L);
        if (atten <= 0.0) continue;

        vec3 radiance = light.color.rgb * light.color.w * atten;
        int face = int(light.params.y);
        float shadow = face < 0 ? 1.0 : atlas_shadow(face, v_world_pos, N, L, dist);
        color += brdf(N, V, L, radiance, albedo, metallic, a, f0, energy) * shadow;
    }

    // Emissive adds on top, unaffected by scene lighting.
    color += m.emissive.rgb * emis_tex;

    color = apply_fog(color, v_world_pos, lighting.camera_pos.xyz);

    if (lighting.shadow_params.w > 0.5) {
        color *= cascade_debug_tint(select_cascade(view_dist));
    }

    f_color = vec4(color, 1.0);
}
