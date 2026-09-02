//! Drawing a [`Value`] as editable widgets, with no knowledge of the type it
//! came from.
//!
//! The default every registration gets. A component whose numbers mean
//! something — a light's lumens, an angle in degrees — replaces it through
//! [`Registry::set_inspector`](crate::Registry::set_inspector); everything else,
//! including every component a game assembly registers, is inspectable the
//! moment it is registered rather than when someone writes a panel for it.
//!
//! What this deliberately cannot do is change a value's *shape*: no switching
//! enum variant, no adding or removing list elements. Both need the set of
//! variants and the element's default, which the v1 vtable does not carry —
//! and a caller that could only offer half the variants would be worse than one
//! that offers none.

use crate::value::Value;

/// Draw `value`'s widgets, reporting whether the user moved any of them.
pub type InspectFn = fn(&mut egui::Ui, &mut Value) -> bool;

/// Speed of every unlabelled numeric drag. Slow enough that a drag across the
/// panel is a readable adjustment rather than a jump to a random magnitude.
const DRAG_SPEED: f64 = 0.05;

/// The default [`InspectFn`]: one row per field, nested under a collapsing
/// header wherever a field is itself composite.
pub fn value(ui: &mut egui::Ui, value: &mut Value) -> bool {
    match value {
        Value::Struct(fields) => fields_rows(ui, fields),
        Value::Enum { variant, fields } => {
            // The variant is shown, not chosen — see the module docs.
            ui.horizontal(|ui| {
                ui.weak("variant");
                ui.label(variant.as_str());
            });
            fields_rows(ui, fields)
        }
        Value::List(items) => {
            let mut changed = false;
            for (index, item) in items.iter_mut().enumerate() {
                changed |= row(ui, &format!("[{index}]"), item);
            }
            changed
        }
        leaf => ui.horizontal(|ui| self::leaf(ui, leaf)).inner,
    }
}

fn fields_rows(ui: &mut egui::Ui, fields: &mut [(String, Value)]) -> bool {
    let mut changed = false;
    for (name, field) in fields {
        changed |= row(ui, name, field);
    }
    changed
}

/// A leaf gets a label beside its widget; anything composite gets a header it
/// can be folded away under, because a component of components otherwise fills
/// the panel with rows whose nesting is invisible.
fn row(ui: &mut egui::Ui, name: &str, field: &mut Value) -> bool {
    match field {
        Value::Struct(_) | Value::Enum { .. } | Value::List(_) => egui::CollapsingHeader::new(name)
            .default_open(true)
            .show(ui, |ui| value(ui, field))
            .body_returned
            .unwrap_or(false),
        leaf => {
            ui.horizontal(|ui| {
                ui.label(name);
                self::leaf(ui, leaf)
            })
            .inner
        }
    }
}

fn leaf(ui: &mut egui::Ui, value: &mut Value) -> bool {
    match value {
        Value::Bool(v) => ui.checkbox(v, "").changed(),
        Value::I32(v) => ui.add(egui::DragValue::new(v)).changed(),
        Value::U32(v) => ui.add(egui::DragValue::new(v)).changed(),
        Value::F32(v) => ui.add(egui::DragValue::new(v).speed(DRAG_SPEED)).changed(),
        Value::String(v) => ui.text_edit_singleline(v).changed(),
        Value::Vec3(v) => {
            let mut changed = false;
            for component in [&mut v.x, &mut v.y, &mut v.z] {
                changed |= ui
                    .add(egui::DragValue::new(component).speed(DRAG_SPEED))
                    .changed();
            }
            changed
        }
        Value::Quat(v) => quat(ui, v),
        // Retargeting a reference needs a picker over the live world, which is
        // one level above a function that has only the value. Shown so the link
        // is at least visible.
        Value::Entity(id) => {
            ui.weak(id.to_string());
            false
        }
        // Routed to `value` by `row` and never reached here.
        Value::Struct(_) | Value::Enum { .. } | Value::List(_) => false,
    }
}

/// Edited as Euler degrees in the same convention as the C# scripting API
/// (`Quaternion.Euler`): Z-X-Y application, i.e. glam's intrinsic
/// `EulerRot::YXZ`. The three fields read X, Y, Z (pitch, yaw, roll) while the
/// conversion takes and yields (yaw, pitch, roll), hence the reordering.
///
/// Written back only when a drag actually moved, because euler is a lossy
/// coordinate for a quaternion: recomposing every frame would let a rotation
/// creep while nobody was touching it.
fn quat(ui: &mut egui::Ui, value: &mut glam::Quat) -> bool {
    let (yaw, pitch, roll) = value.to_euler(glam::EulerRot::YXZ);
    let mut euler = [pitch.to_degrees(), yaw.to_degrees(), roll.to_degrees()];
    let mut changed = false;
    for angle in &mut euler {
        changed |= ui
            .add(egui::DragValue::new(angle).speed(0.5).suffix("°"))
            .changed();
    }
    if changed {
        *value = glam::Quat::from_euler(
            glam::EulerRot::YXZ,
            euler[1].to_radians(),
            euler[0].to_radians(),
            euler[2].to_radians(),
        );
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    use crate::EntityId;
    use glam::{EulerRot, Quat, Vec3};

    /// One of every variant, nested, so that adding a `Value` variant without
    /// teaching the inspector about it is caught here rather than by a component
    /// that silently draws nothing.
    fn every_shape() -> Value {
        Value::strukt([
            ("flag", Value::Bool(true)),
            ("count", Value::I32(-3)),
            ("index", Value::U32(7)),
            ("speed", Value::F32(1.5)),
            ("label", Value::String("cube".to_owned())),
            ("position", Value::Vec3(Vec3::new(1.0, 2.0, 3.0))),
            // A compound rotation, not a single-axis one: the euler round trip
            // is exact for the latter, so a fixture built from one would let the
            // creep this test exists for pass unnoticed.
            (
                "rotation",
                Value::Quat(Quat::from_euler(EulerRot::YXZ, 0.3, 0.4, 0.5)),
            ),
            ("target", Value::Entity(EntityId::NIL)),
            ("points", Value::List(vec![Value::F32(0.5)])),
            (
                "light",
                Value::enumeration("Spot", [("angle", Value::F32(30.0))]),
            ),
        ])
    }

    /// Drawing is not editing. Every widget here reports "unchanged" when nobody
    /// touches it, and the value must come back byte-identical — which is the
    /// claim `quat`'s write-back guard exists to keep, since euler is a lossy
    /// coordinate and recomposing unconditionally would let an untouched
    /// rotation creep every frame the panel was open.
    #[test]
    fn drawing_without_interacting_reports_nothing_and_changes_nothing() {
        let before = every_shape();
        // `__run_test_ui` takes an `Fn`, so the value under edit lives behind a
        // cell rather than being captured mutably.
        let after = RefCell::new(before.clone());
        egui::__run_test_ui(|ui| {
            assert!(!value(ui, &mut after.borrow_mut()));
        });
        assert_eq!(crate::diff(&before, &after.into_inner()), Vec::new());
    }

    #[test]
    fn a_bare_leaf_draws_without_a_surrounding_struct() {
        let leaf = RefCell::new(Value::F32(2.0));
        egui::__run_test_ui(|ui| {
            assert!(!value(ui, &mut leaf.borrow_mut()));
        });
        assert_eq!(leaf.into_inner(), Value::F32(2.0));
    }
}
