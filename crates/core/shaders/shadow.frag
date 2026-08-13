#version 460

// Depth-only: the render pass has no color attachment and this stage writes
// nothing. A vertex-only pipeline is legal Vulkan for this, but MoltenVK's
// handling of a nil Metal fragment function has been inconsistent, and an empty
// shader costs nothing measurable.
//
// With ORRIN_MASKED it stops being empty and becomes the reason a leaf casts the
// shadow of a leaf rather than of the quad it is painted on. A cutout is opaque
// where it survives its own alpha test, so it is in every caster list — and a
// shadow map is nothing but depth, which means the test has to be repeated here
// or the map records the card.
//
// The map is one sample, so this is the hard comparison the prepass takes and
// not the coverage ramp the forward pass writes. Nothing samples a shadow map
// expecting a fractional occluder.

#ifdef ORRIN_MASKED
layout(location = 0) in vec2 v_uv;

layout(push_constant) uniform Push {
    mat4 light_view_proj;
    uint object_base;
    uint material_index;
} push;

// Mirrors GpuMaterial in forward.rs, as `prepass.frag` does and for the same
// std430 reason: the stride comes from the struct, so a short mirror would read
// every material past the first out of its neighbour. Only `base_color.a`, the
// albedo index and the cutoff are used.
struct GpuMaterial {
    vec4 base_color;
    vec4 emissive;
    vec4 params;
    uvec4 tex_indices;
    vec4 clearcoat;
    vec4 sheen;
    vec4 anisotropy;
    vec4 transmission;
    vec4 attenuation;
    vec4 subsurface;
    vec4 subsurface_radius;
    vec4 parallax;
    vec4 alpha;       // x = the alpha a MASKED fragment must reach to survive
    uvec4 tex_indices_ext;
    uvec4 tex_flags;
};
layout(set = 1, binding = 0, std430) readonly buffer Materials {
    GpuMaterial materials[];
};

// Keep in sync with MAX_TEXTURES in gfx/mod.rs. One shared sampler beside an
// array of textures, for the per-stage sampler limit Metal imposes.
const int MAX_TEXTURES = 64;
layout(set = 2, binding = 0) uniform texture2D u_textures[MAX_TEXTURES];
layout(set = 2, binding = 1) uniform sampler u_sampler;
#endif

void main() {
#ifdef ORRIN_MASKED
    GpuMaterial m = materials[push.material_index];
    // Deliberately not the parallax-marched coordinate the other two passes use.
    // The march needs a view direction, and the view here is the light's, so a
    // relief-mapped cutout would be tested along the wrong ray. A height field
    // moves texels by centimetres and a cutout's alpha is flat across those
    // centimetres in every atlas anyone authors, so the unmarched coordinate is
    // the right answer rather than a cheaper one.
    float alpha = m.base_color.a
        * texture(sampler2D(u_textures[m.tex_indices.x], u_sampler), v_uv).a;
    if (alpha < m.alpha.x) {
        discard;
    }
#endif
}
