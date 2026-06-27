use crate::gfx::Vertex;

#[derive(Clone, Default)]
pub struct CpuMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl CpuMesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    pub fn cube() -> Self {
        // Per-face vertices so each face has a flat normal.
        const FACES: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
            ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),   // +Z
            ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), // -Z
            ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),  // +X
            ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),  // -X
            ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),  // +Y
            ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),  // -Y
        ];

        let mut vertices = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(36);

        for (normal, u, v) in FACES {
            let n = glam::Vec3::from(normal);
            let u = glam::Vec3::from(u);
            let v = glam::Vec3::from(v);
            let base = vertices.len() as u32;
            let color = (n * 0.5 + 0.5).to_array();

            // The tangent follows the +U texture axis (= u). The bitangent the
            // shader rebuilds is `cross(N, T) * w`; pick w so that lands on +v.
            let w = if n.cross(u).dot(v) >= 0.0 { 1.0 } else { -1.0 };
            let tangent = [u.x, u.y, u.z, w];

            for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let pos = (n + u * su + v * sv) * 0.5;
                vertices.push(Vertex {
                    position: pos.to_array(),
                    normal,
                    color,
                    // Map the [-1,1] quad corners to [0,1] UVs along u and v.
                    uv: [su * 0.5 + 0.5, sv * 0.5 + 0.5],
                    tangent,
                });
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        Self { vertices, indices }
    }
}
