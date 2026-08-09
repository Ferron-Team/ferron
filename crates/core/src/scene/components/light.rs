use glam::Vec3;
use orrin_registry::Reflect;

/// Variant names are part of the on-disk format, exactly like a component's id:
/// renaming `Directional` orphans every saved light that used it, and nothing
/// catches that at compile time. Add variants freely; rename them never.
///
/// Field *names* are the same promise, and adding one to a variant that already
/// ships is the same hazard from the other direction: a scene saved before it
/// existed has no such field, and the derive reports that as a broken entity
/// rather than an old one. `casts_shadows` therefore carries
/// `#[reflect(default = true)]`, which is what lets a point light written by an
/// earlier build load as the shadow caster it would have been.
#[derive(Clone, Copy, Debug, Reflect)]
pub enum Light {
    Directional {
        color: Vec3,
        intensity: f32,
    },
    Point {
        color: Vec3,
        intensity: f32,
        range: f32,
        /// Whether this light writes into the shadow atlas. The atlas holds a
        /// fixed number of tiles and a point light spends six of them, so this
        /// is the switch for a fill light that should never cost that — and
        /// what the importance sort spends its budget on when more lights ask
        /// than fit.
        #[reflect(default = true)]
        casts_shadows: bool,
    },
    /// A cone. `inner_angle` is where it is still at full brightness and
    /// `outer_angle` where it has fallen to nothing, both as the *half* angle
    /// from the axis in degrees — the same convention glTF uses, and the one
    /// that makes the cone's field of view `2 * outer_angle`.
    ///
    /// Its axis is the entity's forward, `-Z`, matching `Directional`.
    Spot {
        color: Vec3,
        intensity: f32,
        range: f32,
        inner_angle: f32,
        outer_angle: f32,
        /// One tile rather than a point light's six, because a cone is a single
        /// frustum — which makes a spot the cheapest thing in the atlas by a
        /// factor of six.
        #[reflect(default = true)]
        casts_shadows: bool,
    },
}

impl Light {
    #[inline]
    pub fn directional(color: Vec3, intensity: f32) -> Self {
        Self::Directional { color, intensity }
    }

    #[inline]
    pub fn point(color: Vec3, intensity: f32, range: f32) -> Self {
        Self::Point {
            color,
            intensity,
            range,
            casts_shadows: true,
        }
    }

    #[inline]
    pub fn spot(
        color: Vec3,
        intensity: f32,
        range: f32,
        inner_angle: f32,
        outer_angle: f32,
    ) -> Self {
        Self::Spot {
            color,
            intensity,
            range,
            inner_angle,
            outer_angle,
            casts_shadows: true,
        }
    }
}

impl Default for Light {
    fn default() -> Self {
        Self::directional(Vec3::ONE, 1.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AmbientLight {
    pub color: Vec3,
    pub intensity: f32,
}

impl Default for AmbientLight {
    fn default() -> Self {
        Self {
            color: Vec3::new(0.6, 0.7, 1.0),
            intensity: 0.15,
        }
    }
}
