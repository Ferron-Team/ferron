#version 460

layout(location = 0) in vec3 position;
#ifdef ORRIN_MASKED
// Only the cutout variant reads the maps, so only it declares the attribute.
// `Vertex::per_vertex().definition(&vs)` derives the vertex input from what the
// shader asked for, so the plain pipeline still fetches position alone.
layout(location = 3) in vec2 uv;
layout(location = 0) out vec2 v_uv;
#endif

// The cascade's light view-projection, pushed per pass, plus where this
// instanced run starts in `instances`; gl_InstanceIndex counts from it, and the
// entry found there is the object's row.
layout(push_constant) uniform Push {
    mat4 light_view_proj;
    uint object_base;
#ifdef ORRIN_MASKED
    // Which row of the material table this run alpha-tests against. Declared in
    // the cutout variant alone, so the plain pipeline's push range stays the
    // shape it always was and the two Rust structs mirror one variant each.
    uint material_index;
#endif
} push;

// The same per-object buffer the forward pass reads, uploaded once per frame.
// Mirrors GpuObject in forward.rs: both fields stay declared even though only
// `model` is used here, because dropping one would halve every index.
struct Object {
    mat4 model;
    mat4 normal_matrix;
    mat4 prev_model; // read only by the prepass; here to keep the stride in step
};
layout(set = 0, binding = 0, std430) readonly buffer Objects {
    Object objects[];
};

// This pass's slice of the frame's draw order, as row numbers into `objects`.
// The indirection is what lets a row be shared: an object the camera sees and
// four cascades also draw occupies one row named five times, rather than five
// copies of the same three matrices. `push.object_base` is where this run's
// slice starts. See `vulkan::instances`.
layout(set = 0, binding = 1, std430) readonly buffer Instances {
    uint instances[];
};

void main() {
    uint object = instances[push.object_base + uint(gl_InstanceIndex)];
    gl_Position = push.light_view_proj * objects[object].model * vec4(position, 1.0);
#ifdef ORRIN_MASKED
    v_uv = uv;
#endif
}
