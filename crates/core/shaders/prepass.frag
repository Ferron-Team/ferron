#version 460

layout(location = 0) in vec3 v_view_normal;
layout(location = 1) in vec3 v_view_tangent;
layout(location = 2) in vec3 v_view_bitangent;
layout(location = 3) in vec2 v_uv;
layout(location = 4) in vec3 v_color;
layout(location = 5) in vec4 v_clip;
layout(location = 6) in vec4 v_previous_clip;
layout(location = 7) in vec3 v_view_pos;

layout(location = 0) out vec4 f_normal;
layout(location = 1) out vec2 f_velocity;
layout(location = 2) out vec4 f_material;

layout(push_constant) uniform Push {
    uint object_base;
    uint material_index;
} push;

// Mirrors GpuMaterial in forward.rs, field for field: the two passes read the
// same buffer through their own layouts, so a change here is a change there.
//
// The lobe blocks are declared but unread, and both halves of that are
// deliberate. Declared, because std430 derives the array stride from the struct
// and a short mirror would step through the table at the wrong pitch — every
// material past the first would be read out of the middle of its neighbour.
// Unread, because this pass writes the *base* layer's f0 and roughness and
// `ssr_resolve.comp` subtracts exactly that: the clear coat's own reflection
// stays image-based, so nothing here has a second lobe to describe.
//
// Parallax is the one that is read, because it is not a lobe: it changes *where*
// the maps are sampled rather than what is done with them, so this pass and the
// forward pass have to march the same ray to be describing one pixel.
struct GpuMaterial {
    vec4 base_color;
    vec4 emissive;
    vec4 params;      // x = metallic, y = roughness, z = reflectance, w = ior
    uvec4 tex_indices; // [albedo, normal, metal-rough, emissive]
    vec4 clearcoat;
    vec4 sheen;
    vec4 anisotropy;
    vec4 transmission;
    vec4 attenuation;
    vec4 subsurface;
    vec4 subsurface_radius;
    // Read here, unlike the blocks around it: the march moves the coordinate
    // every one of the three outputs below is sampled at, so a prepass that
    // skipped it would report the flat wall's normal for the pixel the forward
    // pass shaded one brick along.
    vec4 parallax;    // x = height field depth in metres, y = min steps, z = max steps
    uvec4 tex_indices_ext;
    uvec4 tex_flags;  // x = transmission map, y = feature flags, z = subsurface, w = height
};
layout(set = 2, binding = 0, std430) readonly buffer Materials {
    GpuMaterial materials[];
};

// Keep in sync with MAX_TEXTURES in gfx/mod.rs, as forward.frag does. One
// shared sampler beside an array of textures rather than 64 combined ones, for
// the sampler-limit reason that shader documents.
const int MAX_TEXTURES = 64;
layout(set = 3, binding = 0) uniform texture2D u_textures[MAX_TEXTURES];
layout(set = 3, binding = 1) uniform sampler u_sampler;

vec4 sample_tex(uint index, vec2 uv) {
    return texture(sampler2D(u_textures[index], u_sampler), uv);
}

// Feature bits in `tex_flags.y`; only the one this pass acts on is named here.
// Mirrors `material_flags` in forward.rs, as `shading.glsl` does.
const uint MATERIAL_PARALLAX = 1u << 5;

float parallax_height(uint index, vec2 uv, vec2 dx, vec2 dy) {
    return textureGrad(sampler2D(u_textures[index], u_sampler), uv, dx, dy).r;
}
#include "parallax.glsl"

void main() {
    GpuMaterial m = materials[push.material_index];

    mat3 TBN = mat3(normalize(v_view_tangent), normalize(v_view_bitangent), normalize(v_view_normal));

    // The same march `shading.glsl` runs, one space along. `view` is rigid, so
    // the tangent-space ray and the metres the depth is authored in survive the
    // change of basis unchanged — which is exactly why the two passes can agree.
    vec2 uv = v_uv;
    if ((m.tex_flags.y & MATERIAL_PARALLAX) != 0u) {
        vec3 V = normalize(-v_view_pos); // the camera is the origin of view space
        vec2 dx = dFdx(v_uv);
        vec2 dy = dFdy(v_uv);
        uv = parallax_occlusion(
            m.tex_flags.w,
            uv,
            normalize(vec3(dot(V, TBN[0]), dot(V, TBN[1]), dot(V, TBN[2]))),
            m.parallax.x,
            parallax_uv_per_metre(
                TBN[0], TBN[1], dx, dy, dFdx(v_view_pos), dFdy(v_view_pos)
            ),
            m.parallax.y,
            m.parallax.z,
            dx,
            dy
        );
    }

    // Same reads and the same conventions as forward.frag: glTF packs roughness
    // in G and metallic in B, and the roughness floor keeps the highlight from
    // going singular.
    vec4  mr_tex     = sample_tex(m.tex_indices.z, uv);
    vec3  albedo     = m.base_color.rgb * v_color * sample_tex(m.tex_indices.x, uv).rgb;
    float metallic   = clamp(m.params.x * mr_tex.b, 0.0, 1.0);
    float roughness  = clamp(m.params.y * mr_tex.g, 0.04, 1.0);
    float reflectance = m.params.z;

    // Dielectric F0 from reflectance (0.5 -> ~4%); metals use albedo as F0.
    vec3 f0 = mix(vec3(0.16 * reflectance * reflectance), albedo, metallic);
    f_material = vec4(f0, roughness);

    vec3 n_tangent = sample_tex(m.tex_indices.y, uv).xyz * 2.0 - 1.0;
    f_normal = vec4(normalize(TBN * n_tangent) * 0.5 + 0.5, 1.0);

    if (v_previous_clip.w <= 0.0) {
        // Behind last frame's camera, so there is no history for this surface at
        // all. A vector this large lands the reprojection off-screen, which is
        // exactly the "reject the history" path the resolve already has.
        f_velocity = vec2(1e3);
        return;
    }

    // UV space rather than NDC, so the resolve can subtract it from a texture
    // coordinate directly. The projection already flips Y for Vulkan's clip
    // space, so this maps to the same orientation the render targets are in.
    vec2 now = (v_clip.xy / v_clip.w) * 0.5 + 0.5;
    vec2 before = (v_previous_clip.xy / v_previous_clip.w) * 0.5 + 0.5;
    f_velocity = now - before;
}
