#version 460

// Brings the froxel mapping and `fog_term`, the same file `shading.glsl` and
// `fog_scatter.comp` include. The sky has to be fogged out of the identical
// volume the geometry is, or the horizon is a seam: the last wall in the scene
// would sit under a hundred metres of air and the sky right beside it under
// none.
#include "fog.glsl"

layout(location = 0) in vec3 v_dir;
layout(location = 0) out vec4 f_color;

// Separated for the reason forward.frag documents: Metal allows far fewer
// sampler states per stage than sampled images, so samplers are shared rather
// than combined into the binding.
layout(set = 0, binding = 0) uniform textureCube u_environment;
layout(set = 0, binding = 1) uniform sampler u_environment_sampler;

// The fog, in this pass's own private set rather than in the five the geometry
// passes share. It costs two bindings and buys not having to declare the whole
// `Lighting` block — point lights, spot lights, irradiance — in a shader that
// draws a cube.
layout(set = 0, binding = 2) uniform sampler3D u_fog_volume;
layout(set = 0, binding = 3) uniform Fog {
    GpuFog f;
} u_fog;

layout(push_constant) uniform Push {
    mat4 inv_view_rot_proj;
    vec4 params; // x = intensity, yz = 1/viewport
} push;

void main() {
    vec3 dir = normalize(v_dir);
    vec3 sky = textureLod(samplerCube(u_environment, u_environment_sampler), dir, 0.0).rgb;
    sky *= push.params.x;

    // A point far enough away to be past every froxel and past anything the
    // height integral still has density at. `fog_term` clamps the volume lookup
    // to its last slice, which already holds the whole volume's integral, and
    // integrates the analytic tail from the volume's far plane out to here — so
    // the sky is attenuated by exactly the air in front of it.
    //
    // Finite rather than a true infinity because the tail's optical depth is an
    // exponential in the ray's height: a ray climbing out of the layer converges
    // long before this, and one running along it is opaque long before it too.
    const float SKY_DISTANCE = 1.0e5;
    vec3 world_pos = u_fog.f.fog.camera_pos.xyz + dir * SKY_DISTANCE;

    FogTerm fog = fog_term(u_fog.f.fog, u_fog_volume, world_pos, gl_FragCoord.xy * push.params.yz);
    f_color = vec4(sky * fog.transmittance + fog.inscatter, 1.0);
}
