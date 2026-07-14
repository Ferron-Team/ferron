//! Narrowphase: exact overlap tests per shape pair, producing a [`Contact`],
//! plus the MTV split used to resolve solid–solid penetration.
//!
//! Every function here upholds the module's normal convention: the returned
//! contact normal is unit length and points from the *first* argument's shape
//! toward the second's.

use glam::Vec3;

use super::{Aabb, Contact, WorldShape};

/// Dispatch to the right shape–shape test. The Sphere/Box case reuses the
/// Box/Sphere test with the normal flipped to keep the a→b convention.
pub fn test(a: &WorldShape, b: &WorldShape) -> Option<Contact> {
    match (a, b) {
        (WorldShape::Box(a), WorldShape::Box(b)) => aabb_aabb(a, b),
        (
            WorldShape::Sphere { center: ca, radius: ra },
            WorldShape::Sphere { center: cb, radius: rb },
        ) => sphere_sphere(*ca, *ra, *cb, *rb),
        (WorldShape::Box(a), WorldShape::Sphere { center, radius }) => {
            aabb_sphere(a, *center, *radius)
        }
        (WorldShape::Sphere { center, radius }, WorldShape::Box(b)) => {
            aabb_sphere(b, *center, *radius)
                .map(|contact| Contact { normal: -contact.normal, ..contact })
        }
    }
}

fn aabb_aabb(a: &Aabb, b: &Aabb) -> Option<Contact> {
    // TODO(owner): compute the overlap extent on each axis
    // (min(a.max, b.max) - max(a.min, b.min)); any non-positive extent means
    // no collision. Otherwise the *smallest* overlap axis is the separation
    // axis: normal is ±that axis unit vector (sign chosen so it points from
    // a's center toward b's), depth is that extent, and the contact point is
    // the center of the overlap box. Picking the smallest axis is what makes
    // the MTV "minimum" — pushing out along any other axis moves further.
    todo!("narrowphase::aabb_aabb")
}

fn sphere_sphere(ca: Vec3, ra: f32, cb: Vec3, rb: f32) -> Option<Contact> {
    // TODO(owner): let d = cb - ca. Colliding iff |d| < ra + rb. Normal is
    // d.normalize_or(Vec3::X) — the fallback handles concentric spheres, where
    // any direction is as good as any other but NaN is not. Depth is
    // (ra + rb) - |d|; contact point sits on a's surface along the normal
    // (ca + normal * ra), or the midpoint of the overlap if you prefer —
    // just be consistent.
    todo!("narrowphase::sphere_sphere")
}

fn aabb_sphere(a: &Aabb, center: Vec3, radius: f32) -> Option<Contact> {
    // TODO(owner): clamp the sphere center to the box (component-wise
    // center.clamp(a.min, a.max)) to get the closest point on/in the box.
    // Colliding iff distance(closest, center) < radius. Normal points from
    // the closest point toward the sphere center; depth is
    // radius - distance; contact point is the closest point. Watch the
    // degenerate case where the center is *inside* the box (closest ==
    // center, distance == 0): fall back to pushing out along the axis where
    // the center is nearest a face, like the aabb_aabb smallest-axis rule.
    todo!("narrowphase::aabb_sphere")
}

/// Split a solid–solid penetration into per-body displacement offsets:
/// returns `(offset_a, offset_b)` for the contact's `a` and `b` entities.
pub fn resolve_offsets(contact: &Contact) -> (Vec3, Vec3) {
    // TODO(owner): the minimum translation vector is normal * depth. With no
    // mass or velocity yet, split it evenly: a moves -normal * depth * 0.5,
    // b moves +normal * depth * 0.5. (When mass arrives, this becomes an
    // inverse-mass-weighted split, and `0.5` is the equal-mass special case.)
    todo!("narrowphase::resolve_offsets")
}

#[cfg(test)]
mod tests {
    use super::*;

    // Red until the narrowphase functions are implemented:
    // `cargo test -p renderer-prototype collision`.

    fn unit_box_at(center: Vec3) -> Aabb {
        Aabb { min: center - Vec3::splat(0.5), max: center + Vec3::splat(0.5) }
    }

    #[test]
    fn separated_shapes_do_not_collide() {
        assert!(aabb_aabb(&unit_box_at(Vec3::ZERO), &unit_box_at(Vec3::new(3.0, 0.0, 0.0))).is_none());
        assert!(sphere_sphere(Vec3::ZERO, 0.5, Vec3::new(3.0, 0.0, 0.0), 0.5).is_none());
        assert!(aabb_sphere(&unit_box_at(Vec3::ZERO), Vec3::new(3.0, 0.0, 0.0), 0.5).is_none());
    }

    #[test]
    fn aabb_aabb_picks_smallest_axis() {
        // Offset 0.8 on x, 0.2 on y: overlap is 0.2 on x, 0.8 on y, 1.0 on z,
        // so the MTV must be along +x with depth 0.2.
        let a = unit_box_at(Vec3::ZERO);
        let b = unit_box_at(Vec3::new(0.8, 0.2, 0.0));
        let contact = aabb_aabb(&a, &b).expect("boxes overlap");
        assert!((contact.normal - Vec3::X).length() < 1e-5);
        assert!((contact.depth - 0.2).abs() < 1e-5);
    }

    #[test]
    fn sphere_sphere_reports_depth_and_direction() {
        let contact =
            sphere_sphere(Vec3::ZERO, 0.5, Vec3::new(0.8, 0.0, 0.0), 0.5).expect("spheres overlap");
        assert!((contact.normal - Vec3::X).length() < 1e-5);
        assert!((contact.depth - 0.2).abs() < 1e-5);
    }

    #[test]
    fn aabb_sphere_from_the_side() {
        // Sphere just left of the box's -x face, overlapping by 0.2.
        let contact = aabb_sphere(&unit_box_at(Vec3::ZERO), Vec3::new(-0.8, 0.0, 0.0), 0.5)
            .expect("overlapping");
        // Normal points box → sphere, i.e. -x.
        assert!((contact.normal - Vec3::NEG_X).length() < 1e-5);
        assert!((contact.depth - 0.2).abs() < 1e-5);
    }

    #[test]
    fn resolve_offsets_split_the_mtv() {
        let contact = Contact { point: Vec3::ZERO, normal: Vec3::X, depth: 0.2 };
        let (offset_a, offset_b) = resolve_offsets(&contact);
        assert!((offset_a - Vec3::new(-0.1, 0.0, 0.0)).length() < 1e-5);
        assert!((offset_b - Vec3::new(0.1, 0.0, 0.0)).length() < 1e-5);
        // Together they exactly cancel the penetration.
        assert!(((offset_b - offset_a).length() - contact.depth).abs() < 1e-5);
    }
}
