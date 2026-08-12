use glam::Vec3;

pub fn load_rgba(bytes: &[u8]) -> (Vec<u8>, u32, u32) {
    let img = image::load_from_memory(bytes)
        .expect("failed to decode texture")
        .to_rgba8();
    let (width, height) = img.dimensions();
    (img.into_raw(), width, height)
}

pub fn checkerboard(size: u32, checks: u32, a: [u8; 3], b: [u8; 3]) -> Vec<u8> {
    let cell = (size / checks).max(1);
    let mut data = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let c = if ((x / cell) + (y / cell)).is_multiple_of(2) {
                a
            } else {
                b
            };
            data.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
    data
}

pub fn bump_normals(size: u32, freq: f32, strength: f32) -> Vec<u8> {
    let mut data = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let u = x as f32 / size as f32 * std::f32::consts::TAU * freq;
            let v = y as f32 / size as f32 * std::f32::consts::TAU * freq;
            // Slope is the gradient of the height field h = sin(u) * sin(v).
            let dx = strength * u.cos() * v.sin();
            let dy = strength * u.sin() * v.cos();
            let n = Vec3::new(-dx, -dy, 1.0).normalize();
            let e = (n * 0.5 + 0.5) * 255.0;
            data.extend_from_slice(&[e.x as u8, e.y as u8, e.z as u8, 255]);
        }
    }
    data
}

/// A running-bond masonry wall, as the three maps that have to agree about it.
pub struct BrickMaps {
    /// sRGB, for a colour map.
    pub albedo: Vec<u8>,
    pub normal: Vec<u8>,
    /// Greyscale height, white at the face of a block and black at the back of
    /// the mortar — the convention every displacement map ships in, and the one
    /// the parallax march reads.
    pub height: Vec<u8>,
}

/// `columns` x `rows` blocks over one tile, offset half a block per course, with
/// the mortar recessed by `depth` metres.
///
/// All three maps come out of one height field rather than being authored
/// separately, and that is the point of generating them here: the normal map is
/// that field's own gradient, so the shading and the parallax describe the same
/// blocks. Two maps drawn independently is the classic way to get a wall whose
/// mortar is lit as though it were raised.
///
/// Which is also why this is told `tile`, the metres of wall the map is stretched
/// over. A slope is a depth over a width, so the normals cannot be derived
/// without it — and a wall stretched twice as far across as up needs two
/// different slopes out of one field. Passing the same `depth` on to
/// [`Material::parallax_depth`](crate::gfx::Material::parallax_depth) is then all
/// that keeps the marched field and the shaded one the same field.
pub fn brick(size: u32, columns: u32, rows: u32, tile: [f32; 2], depth: f32) -> BrickMaps {
    let cell_w = (size / columns).max(2);
    let cell_h = (size / rows).max(2);
    // Everything below measures across the wall in metres rather than in texels,
    // because a texel is not square on a wall wider than it is tall: a joint
    // given as a texel count comes out twice as wide as it is deep, and on this
    // demo's 4 x 2 m panel that was enough to close the blocks up entirely.
    let metres = [tile[0] / size as f32, tile[1] / size as f32];
    let block = (cell_w as f32 * metres[0]).min(cell_h as f32 * metres[1]);
    // A joint a twentieth of a block, and a bevel as wide again: a block with a
    // vertical wall reads as a hole punched in a plane, because the march
    // resolves that wall over one step and the normal map has nothing to shade.
    let joint = block * 0.05;
    let bevel = joint;

    let texel = |x: u32, y: u32| -> (f32, f32) {
        let row = y / cell_h;
        // Half-block offset on alternate courses. Whole texels, or the shift
        // lands mid-texel and the course above meets the one below off by one.
        let shift = if row % 2 == 1 { cell_w / 2 } else { 0 };
        let bx = (x + shift) % cell_w;
        let by = y % cell_h;

        // Distance to the nearest joint, whichever way it runs.
        let edge = (bx.min(cell_w - 1 - bx) as f32 * metres[0])
            .min(by.min(cell_h - 1 - by) as f32 * metres[1]);
        let t = ((edge - joint) / bevel).clamp(0.0, 1.0);
        let face = t * t * (3.0 - 2.0 * t);

        // One value per block, so no two are quite the same height or colour.
        // Read from the block's own index rather than from its position, so the
        // tile still meets itself at the seam.
        let column = (x + shift) / cell_w;
        let id = row.wrapping_mul(0x9e37).wrapping_add(column);
        (face, hash01(id))
    };

    let mut height = vec![0.0f32; (size * size) as usize];
    for y in 0..size {
        for x in 0..size {
            let (face, jitter) = texel(x, y);
            // The jitter rides on the face and not on the joint: a block sits
            // slightly proud of its neighbour, but the mortar between them is
            // one continuous surface at the back of the wall.
            height[(y * size + x) as usize] = face * (0.86 + 0.14 * jitter);
        }
    }

    let mut albedo = Vec::with_capacity((size * size * 4) as usize);
    let mut normal = Vec::with_capacity((size * size * 4) as usize);
    let mut height_map = Vec::with_capacity((size * size * 4) as usize);
    let at = |x: u32, y: u32| height[((y % size) * size + (x % size)) as usize];
    for y in 0..size {
        for x in 0..size {
            let (face, jitter) = texel(x, y);

            let block = [
                0.40 + 0.22 * jitter,
                0.16 + 0.13 * jitter,
                0.13 + 0.07 * jitter,
            ];
            let mortar = [0.55, 0.53, 0.49];
            // Per-texel grain, in the colour only. In the height field it would
            // become a normal map of noise, since the normals below are that
            // field's derivative and a derivative amplifies exactly the
            // frequencies a grain is made of.
            let grain = 0.90 + 0.20 * hash01(y.wrapping_mul(size).wrapping_add(x));
            for channel in 0..3 {
                let c = (mortar[channel] + (block[channel] - mortar[channel]) * face) * grain;
                albedo.push((c.clamp(0.0, 1.0) * 255.0) as u8);
            }
            albedo.push(255);

            // Central differences, wrapped, so the tile's normals meet across
            // the seam as its heights do. Each axis scaled by the depth over the
            // metres that axis spans, because that is what turns a height in
            // [0, 1] into a slope — and the two are different numbers on any
            // wall that is not square.
            let du = (at(x + 1, y) - at(x + size - 1, y)) * 0.5 * size as f32 * depth / tile[0];
            let dv = (at(x, y + 1) - at(x, y + size - 1)) * 0.5 * size as f32 * depth / tile[1];
            let n = Vec3::new(-du, -dv, 1.0).normalize();
            let e = (n * 0.5 + 0.5) * 255.0;
            normal.extend_from_slice(&[e.x as u8, e.y as u8, e.z as u8, 255]);

            let h = (at(x, y) * 255.0) as u8;
            height_map.extend_from_slice(&[h, h, h, 255]);
        }
    }

    BrickMaps {
        albedo,
        normal,
        height: height_map,
    }
}

/// PCG's output hash, which is enough of a random number for a wall: what it has
/// to be is repeatable, since three maps ask it the same questions.
fn hash01(seed: u32) -> f32 {
    let state = seed.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277_803_737);
    ((word >> 22) ^ word) as f32 / u32::MAX as f32
}

// glTF convention: G = roughness, B = metallic.
/// A stand-in equirectangular sky: a zenith-to-horizon gradient over dull
/// ground, with a sun disc along `to_sun`. RGBA f32, linear, unbounded — the
/// disc is two orders of magnitude above the sky, which is the dynamic range a
/// real HDRI has and the thing a prefilter has to survive.
///
/// Generated rather than shipped so the environment path has content without a
/// multi-megabyte file in the repo. The bake consumes an equirect whatever
/// produced it, so a decoded `.hdr` is a different source for these pixels, not
/// a different path.
pub fn sky_equirect(width: u32, height: u32, to_sun: Vec3) -> Vec<f32> {
    use std::f32::consts::{PI, TAU};

    // Relative radiance, like the `.hdr` files this stands in for, because
    // calibration is what supplies the absolute level: `EnvironmentSettings`
    // measures whatever sky it is given and scales it to the cd/m² the scene
    // asked for. So only the *ratios* in here matter, and a second scale factor
    // at this end would be one the calibration divides straight back out.
    let to_sun = to_sun.normalize();
    let zenith = Vec3::new(0.04, 0.08, 0.18);
    let horizon = Vec3::new(0.20, 0.24, 0.30);
    let ground = Vec3::new(0.02, 0.018, 0.016);

    let mut data = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        // The inverse of the mapping in equirect_to_cube.frag: v = acos(y)/PI
        // and u = atan2(z, x)/TAU + 0.5. Sampling at texel centres keeps the
        // poles off the exact singularity.
        let theta = (y as f32 + 0.5) / height as f32 * PI;
        let (sin_theta, cos_theta) = theta.sin_cos();
        for x in 0..width {
            let phi = ((x as f32 + 0.5) / width as f32 - 0.5) * TAU;
            let (sin_phi, cos_phi) = phi.sin_cos();
            let dir = Vec3::new(cos_phi * sin_theta, cos_theta, sin_phi * sin_theta);

            let mut color = if dir.y >= 0.0 {
                horizon.lerp(zenith, dir.y.powf(0.45))
            } else {
                ground.lerp(horizon, (1.0 + dir.y * 6.0).clamp(0.0, 1.0) * 0.35)
            };

            // ~1.5 degrees across, with a wide soft halo. The disc stays two
            // orders of magnitude above the sky: it is what a smooth metal
            // reflects and what the prefilter's firefly handling exists to
            // survive, so flattening it would hide the case that matters.
            //
            // Two orders rather than the six a real sun sits above a real sky,
            // and deliberately: a placeholder carrying that ratio would spend
            // every prefilter tap fighting one texel. The ratio a reflection
            // needs to look like a reflection is the point here; a real capture
            // is what brings the rest of it.
            let cosine = dir.dot(to_sun);
            color += Vec3::splat(40.0) * ((cosine - 0.9996) / 0.0004).clamp(0.0, 1.0);
            color += Vec3::new(1.0, 0.85, 0.6) * cosine.max(0.0).powf(64.0) * 0.8;

            data.extend_from_slice(&[color.x, color.y, color.z, 1.0]);
        }
    }
    data
}

pub fn metallic_roughness(size: u32) -> Vec<u8> {
    let band = (size / 8).max(1);
    let mut data = Vec::with_capacity((size * size * 4) as usize);
    for _ in 0..size {
        for x in 0..size {
            let roughness = (x * 255 / size.max(1)) as u8;
            let metallic = if (x / band).is_multiple_of(2) { 255 } else { 0 };
            data.extend_from_slice(&[0, roughness, metallic, 255]);
        }
    }
    data
}
