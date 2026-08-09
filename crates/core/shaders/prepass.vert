#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
layout(location = 3) in vec2 uv;
layout(location = 4) in vec4 tangent; // xyz = tangent, w = handedness

// The TBN basis is built in *view* space, not world: the normal target feeds
// SSAO and the reflection trace, both of which reconstruct positions from this
// frame's depth and therefore work in the camera's frame. `view` is rigid, so
// rotating the world-space basis into it is exact.
layout(location = 0) out vec3 v_view_normal;
layout(location = 1) out vec3 v_view_tangent;
layout(location = 2) out vec3 v_view_bitangent;
layout(location = 3) out vec2 v_uv;
// Carried because the forward pass tints albedo with it, and a metal's `f0` is
// its albedo: dropping it here would give the reflection a different colour
// from the surface reflecting it.
layout(location = 4) out vec3 v_color;
// Both unjittered. The rasterised position is jittered — that is what TAA
// samples the pixel with — but a motion vector carrying the jitter would report
// a subpixel shake as scene motion, and the resolve would then reproject away
// the very offsets it exists to accumulate.
layout(location = 5) out vec4 v_clip;
layout(location = 6) out vec4 v_previous_clip;

layout(push_constant) uniform Push {
    // First object row of this instanced run; gl_InstanceIndex counts from it.
    uint object_base;
    // Row of the material table this run draws with; read by the fragment stage.
    uint material_index;
} push;

layout(set = 0, binding = 0) uniform Frame {
    mat4 view;
    mat4 proj;
    mat4 inv_proj;
    mat4 prev_view_proj;
    // xy = the NDC offset baked into `proj`.
    vec4 jitter;
} frame;

// The same per-object buffer the forward pass reads, uploaded once per frame.
// Mirrors GpuObject in forward.rs.
struct Object {
    mat4 model;
    mat4 normal_matrix;
    mat4 prev_model;
};
layout(set = 1, binding = 0, std430) readonly buffer Objects {
    Object objects[];
};

void main() {
    uint object = push.object_base + uint(gl_InstanceIndex);
    mat4 model = objects[object].model;

    // Same construction as forward.vert, one space along: Gram-Schmidt against
    // the normal, handedness from the tangent's w.
    vec3 world_n = mat3(objects[object].normal_matrix) * normal;
    vec3 world_t = mat3(model) * tangent.xyz;
    vec3 N = normalize(mat3(frame.view) * world_n); // view is rigid → pure rotation
    vec3 T = normalize(mat3(frame.view) * world_t);
    T = normalize(T - dot(T, N) * N);
    v_view_normal = N;
    v_view_tangent = T;
    v_view_bitangent = cross(N, T) * tangent.w;
    v_uv = uv;
    v_color = color;

    // `proj * view` is the same view-projection the forward pass pushes; taking
    // it from the frame UBO keeps the prepass push to the one instance offset.
    vec4 clip = frame.proj * frame.view * model * vec4(position, 1.0);
    gl_Position = clip;

    // `proj` carries the jitter as a translation of clip.xy by jitter * w, so
    // subtracting exactly that recovers the unjittered position.
    v_clip = vec4(clip.xy - frame.jitter.xy * clip.w, clip.zw);
    v_previous_clip = frame.prev_view_proj * objects[object].prev_model * vec4(position, 1.0);
}
