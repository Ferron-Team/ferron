//! Right panel: edit the selected entity. Each section takes a fresh,
//! short-lived borrow of the world so the `RefCell` component storage never
//! double-borrows.

use glam::{EulerRot, Quat};

use orrin_ecs::{Entity, World};
use orrin_registry::Registry;

use super::{color_row, figures, vec3_row};
use crate::editor::icons;
use crate::editor::state::EditorState;
#[cfg(feature = "scripting")]
use crate::scene::ScriptComponent;
use crate::scene::{
    Assets, Light, LocalTransform, LogBuffer, LogLevel, MaterialHandle, MeshHandle, Name, Time,
    WorldTransform,
};

pub fn body(ui: &mut egui::Ui, world: &mut World, state: &mut EditorState, registry: &Registry) {
    let Some(entity) = state.selected else {
        ui.weak("Select an entity in the hierarchy.");
        return;
    };
    if !world.is_alive(entity) {
        state.selected = None;
        return;
    }

    name_section(ui, world, entity);
    transform_section(ui, world, entity);
    mesh_material_section(ui, world, entity);
    light_section(ui, world, entity);
    #[cfg(feature = "scripting")]
    script_section(ui, world, entity);
    registered_sections(ui, world, registry, entity);
    actions_section(ui, world, registry, state, entity);
}

/// The components above name a Rust type each and draw widgets tuned to it.
/// Anything else the registry knows about is drawn from its vtable instead —
/// which is what makes a component the engine has never heard of, registered by
/// a game assembly after a hot reload, inspectable without a panel being written
/// for it.
const HAND_DRAWN: [orrin_registry::ComponentId; 3] = [
    crate::scene::registry::TRANSFORM,
    crate::scene::registry::NAME,
    crate::scene::registry::LIGHT,
];

/// Draw every registered component that has no section of its own.
///
/// The edit is made on a detached copy and posted back as the [`diff`] between
/// the two, rather than written straight into storage. That is the discipline
/// architecture §4.4 asks for: one mutation stream, so undo, prefab overrides
/// and eventually sync all see the same changes an inspector drag produced,
/// rather than each having to observe the world for them.
///
/// [`diff`]: orrin_registry::diff
fn registered_sections(ui: &mut egui::Ui, world: &mut World, registry: &Registry, entity: Entity) {
    for vtable in registry.components() {
        if HAND_DRAWN.contains(&vtable.id) {
            continue;
        }
        let Some(before) = vtable.read(world, entity) else {
            continue;
        };

        let mut after = before.clone();
        let touched = egui::CollapsingHeader::new(vtable.name.as_ref())
            .default_open(true)
            .show(ui, |ui| (vtable.inspect)(ui, &mut after))
            .body_returned
            .unwrap_or(false);
        if !touched {
            continue;
        }

        let changes = orrin_registry::diff(&before, &after);
        if let Err(error) = vtable.apply(world, entity, &changes) {
            let frame = world
                .get_resource::<Time>()
                .map_or(0, |time| time.frame_count());
            if let Some(mut log) = world.get_resource_mut::<LogBuffer>() {
                log.push(
                    LogLevel::Error,
                    format!("{} on entity {}: {error}", vtable.id, entity.index()),
                    frame,
                );
            }
        }
    }
}

/// What can be done to the selected entity as a whole, rather than to one of
/// its components.
fn actions_section(
    ui: &mut egui::Ui,
    world: &mut World,
    registry: &Registry,
    state: &mut EditorState,
    entity: Entity,
) {
    ui.separator();
    ui.horizontal(|ui| {
        if ui.button("Dump to console").clicked() {
            dump(world, registry, entity);
        }
        // The hierarchy deletes the row under the pointer; this deletes the
        // thing being looked at. Both go through the same deferred request —
        // despawning while a panel is iterating the tree is what that defers.
        let name = world.get::<Name>(entity).map_or_else(
            || format!("entity {}", entity.index()),
            |name| name.0.clone(),
        );
        let trash = icons::inline(ui, icons::trash(), icons::ROW);
        if ui
            .add(egui::Button::image_and_text(trash, "Delete"))
            .on_hover_text(format!("Despawn {name}"))
            .clicked()
        {
            state.request_despawn(entity);
        }
    });
}

/// Print the entity exactly as the registry sees it.
///
/// Every section above names a concrete component type; this one names none —
/// it asks each registered type whether it's present and prints whatever comes
/// back. That difference is the point of the registry, and until the inspector
/// itself is registry-driven this button is how a component's registration gets
/// eyeballed.
pub(super) fn dump(world: &mut World, registry: &Registry, entity: Entity) {
    let mut text = String::new();
    orrin_registry::write_entity(&mut text, registry, world, entity);

    // Both resource borrows are taken after the dump is built, so no component
    // borrow is still open when the log is written.
    let frame = world
        .get_resource::<Time>()
        .map_or(0, |time| time.frame_count());
    if let Some(mut log) = world.get_resource_mut::<LogBuffer>() {
        log.push(LogLevel::Info, text, frame);
    }
}

fn name_section(ui: &mut egui::Ui, world: &World, entity: Entity) {
    if let Some(mut name) = world.get_mut::<Name>(entity) {
        ui.horizontal(|ui| {
            ui.label("Name");
            ui.text_edit_singleline(&mut name.0);
        });
    }
    ui.label(figures(format!("id {}", entity.index())).weak());
}

fn transform_section(ui: &mut egui::Ui, world: &World, entity: Entity) {
    let Some(mut t) = world.get_mut::<LocalTransform>(entity) else {
        return;
    };
    egui::CollapsingHeader::new("Transform")
        .default_open(true)
        .show(ui, |ui| {
            vec3_row(ui, "Position", &mut t.translation, 0.05);

            // Edited as Euler degrees in the same convention as the C# scripting API
            // (`Quaternion.Euler` / `eulerAngles`): Z-X-Y application, i.e. glam's
            // intrinsic `EulerRot::YXZ`. Matching it means the numbers shown here equal
            // a script's `eulerAngles` for the same rotation. Mind the reordering: YXZ
            // takes/yields angles as (yaw=Y, pitch=X, roll=Z), but the three fields are
            // laid out X, Y, Z (pitch, yaw, roll).
            let (yaw, pitch, roll) = t.rotation.to_euler(EulerRot::YXZ);
            let mut euler = [pitch.to_degrees(), yaw.to_degrees(), roll.to_degrees()];
            let mut changed = false;
            ui.horizontal(|ui| {
                ui.label("Rotation");
                for angle in &mut euler {
                    changed |= ui
                        .add(egui::DragValue::new(angle).speed(0.5).suffix("°"))
                        .changed();
                }
            });
            if changed {
                t.rotation = Quat::from_euler(
                    EulerRot::YXZ,
                    euler[1].to_radians(), // yaw   (Y)
                    euler[0].to_radians(), // pitch (X)
                    euler[2].to_radians(), // roll  (Z)
                );
            }

            vec3_row(ui, "Scale", &mut t.scale, 0.05);

            // The fields above are parent-relative. For anything with a parent
            // that makes them baffling on their own — a cube sitting visibly at
            // x=12 reads "Position 0". The world position is the reconciling
            // number, and it is read-only because it is derived.
            if !crate::scene::is_transform_root(world, entity)
                && let Some(world_transform) = world.get::<WorldTransform>(entity)
            {
                let position = world_transform.translation();
                ui.horizontal(|ui| {
                    ui.weak("World");
                    ui.label(
                        figures(format!(
                            "{:.2}, {:.2}, {:.2}",
                            position.x, position.y, position.z
                        ))
                        .weak(),
                    );
                });
            }
        });
}

fn mesh_material_section(ui: &mut egui::Ui, world: &mut World, entity: Entity) {
    if !world.has::<MeshHandle>(entity) {
        return;
    }
    egui::CollapsingHeader::new("Mesh & Material")
        .default_open(true)
        .show(ui, |ui| {
            if let Some(handle) = mesh_picker(ui, world, entity) {
                world.insert(entity, handle);
            }
            if let Some(handle) = material_picker(ui, world, entity) {
                world.insert(entity, handle);
            }
        });
}

/// Returns a new handle if the user picked a different one. Drops all world
/// borrows before returning, so the caller can safely `insert`.
fn mesh_picker(ui: &mut egui::Ui, world: &World, entity: Entity) -> Option<MeshHandle> {
    let mut options: Vec<(String, MeshHandle)> = world
        .resource::<Assets>()
        .meshes()
        .map(|(n, h)| (n.to_owned(), h))
        .collect();
    options.sort_by(|a, b| a.0.cmp(&b.0));

    let current = world.get::<MeshHandle>(entity).map(|h| *h);
    let mut chosen = current;
    let label = name_of(&options, current);

    egui::ComboBox::from_label("Mesh")
        .selected_text(label)
        .show_ui(ui, |ui| {
            for (name, handle) in &options {
                ui.selectable_value(&mut chosen, Some(*handle), name.as_str());
            }
        });

    (chosen != current).then(|| chosen).flatten()
}

fn material_picker(ui: &mut egui::Ui, world: &World, entity: Entity) -> Option<MaterialHandle> {
    let mut options: Vec<(String, MaterialHandle)> = world
        .resource::<Assets>()
        .materials()
        .map(|(n, h)| (n.to_owned(), h))
        .collect();
    options.sort_by(|a, b| a.0.cmp(&b.0));

    let current = world.get::<MaterialHandle>(entity).map(|h| *h);
    let mut chosen = current;
    let label = name_of(&options, current);

    egui::ComboBox::from_label("Material")
        .selected_text(label)
        .show_ui(ui, |ui| {
            for (name, handle) in &options {
                ui.selectable_value(&mut chosen, Some(*handle), name.as_str());
            }
        });

    (chosen != current).then(|| chosen).flatten()
}

fn name_of<H: Copy + PartialEq>(options: &[(String, H)], handle: Option<H>) -> String {
    handle
        .and_then(|h| options.iter().find(|(_, opt)| *opt == h))
        .map(|(name, _)| name.clone())
        .unwrap_or_else(|| "—".to_owned())
}

#[cfg(feature = "scripting")]
fn script_section(ui: &mut egui::Ui, world: &World, entity: Entity) {
    let Some(mut script) = world.get_mut::<ScriptComponent>(entity) else {
        return;
    };
    egui::CollapsingHeader::new("Script")
        .default_open(true)
        .show(ui, |ui| {
            ui.weak(script.type_name.clone());
            // Only the desired state is written here; the script tick sees the
            // change next frame and dispatches OnEnable/OnDisable itself.
            ui.checkbox(&mut script.enabled, "Enabled");
            if script.faulted {
                // A hook threw and the engine disabled the script (see the log
                // for the exception). Clearing re-arms it from next tick — the
                // manual counterpart to "re-enable on hot reload".
                icons::labelled(ui, icons::warning(), crate::editor::theme::ERROR, "faulted");
                if ui.button("Clear fault").clicked() {
                    script.faulted = false;
                }
            } else {
                ui.weak(match (script.active, script.started) {
                    (true, _) => "active",
                    (false, true) => "inactive",
                    (false, false) => "not started",
                });
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Collider, ColliderShape, Spin, Tag, register_components};
    use glam::Vec3;
    use orrin_registry::Value;

    fn registry() -> Registry {
        let mut registry = Registry::new();
        register_components(&mut registry);
        registry
    }

    /// A hand-written section skips its component here by id. Rename the id and
    /// the skip stops matching, so the component is drawn twice — once tuned and
    /// once generic — which looks like a duplicated panel and reads as a
    /// rendering bug rather than as the rename it is.
    #[test]
    fn every_hand_drawn_component_is_registered_under_the_id_it_is_skipped_by() {
        let registry = registry();
        for id in &HAND_DRAWN {
            assert!(
                registry.get(id).is_some(),
                "`{id}` is skipped by the inspector but not registered"
            );
        }
    }

    /// The other side of the same coin: every component the registry knows and
    /// no section draws must reach the generic inspector. Named here so that
    /// adding a component without a section is visibly a decision.
    #[test]
    fn everything_else_is_drawn_from_its_vtable() {
        let registry = registry();
        let generic: Vec<&str> = registry
            .components()
            .filter(|c| !HAND_DRAWN.contains(&c.id))
            .map(|c| c.id.as_str())
            .collect();
        assert_eq!(
            generic,
            ["orrin.tag", "orrin.collider", "orrin.spin"],
            "registration order changed, or a component gained/lost a section"
        );
    }

    /// The path an inspector drag takes: read the component, edit the detached
    /// value, diff, apply. Driven here without egui, because what is worth
    /// asserting is that the change survives the round trip onto a real
    /// component — including one whose shape is an enum.
    #[test]
    fn an_edit_travels_as_a_diff_and_lands_on_the_component() {
        let registry = registry();
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(
            entity,
            Collider {
                shape: ColliderShape::Sphere { radius: 1.0 },
                is_trigger: false,
            },
        );

        let vtable = registry.get(&crate::scene::registry::COLLIDER).unwrap();
        let before = vtable.read(&world, entity).unwrap();
        let mut after = before.clone();
        *after.field_mut("is_trigger").unwrap() = Value::Bool(true);
        *after
            .field_mut("shape")
            .unwrap()
            .field_mut("radius")
            .unwrap() = Value::F32(2.5);

        let changes = orrin_registry::diff(&before, &after);
        assert_eq!(changes.len(), 2);
        vtable.apply(&mut world, entity, &changes).unwrap();

        let collider = world.get::<Collider>(entity).unwrap();
        assert!(collider.is_trigger);
        assert!(matches!(collider.shape, ColliderShape::Sphere { radius } if radius == 2.5));
    }

    /// `Spin::from_value` refuses an axis it cannot normalize. That refusal has
    /// to survive the diff route as well, or an inspector drag becomes the one
    /// way into the world that skips a component's own validation.
    #[test]
    fn a_component_still_refuses_a_value_its_constructor_rejects() {
        let registry = registry();
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Spin::new(Vec3::Y, 1.0));

        let vtable = registry.get(&crate::scene::registry::SPIN).unwrap();
        let before = vtable.read(&world, entity).unwrap();
        let mut after = before.clone();
        *after.field_mut("axis").unwrap() = Value::Vec3(Vec3::ZERO);

        let error = vtable
            .apply(&mut world, entity, &orrin_registry::diff(&before, &after))
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "field `axis`: expected a non-zero axis, found Vec3(0.0, 0.0, 0.0)"
        );
        assert_eq!(vtable.read(&world, entity), Some(before));
    }

    /// A hand-written section is the exception, so the generic one has to handle
    /// a component it has never been told about — the case every game assembly
    /// component is in.
    #[test]
    fn a_component_with_no_section_reads_back_what_the_inspector_drew() {
        let registry = registry();
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Tag::new("player"));

        let vtable = registry.get(&crate::scene::registry::TAG).unwrap();
        let before = vtable.read(&world, entity).unwrap();
        let drawn = std::cell::RefCell::new(before.clone());
        egui::__run_test_ui(|ui| {
            assert!(!(vtable.inspect)(ui, &mut drawn.borrow_mut()));
        });
        assert_eq!(drawn.into_inner(), before);
    }
}

/// Range is where the light is cut off, not how far it carries — inverse-square
/// decides the latter and never reaches zero. Worth spelling out on the tooltip
/// now that brightness is a physical quantity: the instinct with a unitless
/// intensity was to reach for this slider to make a light dimmer, and it is the
/// one control here that cannot do that.
fn range_row(ui: &mut egui::Ui, range: &mut f32) {
    ui.add(
        egui::Slider::new(range, 0.0..=100.0)
            .suffix(" m")
            .text("Range"),
    )
    .on_hover_text(
        "Where the falloff is clamped to zero, so this light stops costing anything \
         past it. A budget, not a brightness.",
    );
}

fn light_section(ui: &mut egui::Ui, world: &World, entity: Entity) {
    let Some(mut light) = world.get_mut::<Light>(entity) else {
        return;
    };
    egui::CollapsingHeader::new("Light")
        .default_open(true)
        .show(ui, |ui| match &mut *light {
            Light::Directional { color, lux } => {
                color_row(ui, "Color", color);
                // Logarithmic, as every one of these is: the useful range spans
                // five orders of magnitude, and on a linear slider everything
                // dimmer than an overcast day is the leftmost pixel.
                ui.add(
                    egui::Slider::new(lux, 1.0..=150_000.0)
                        .logarithmic(true)
                        .suffix(" lx")
                        .text("Illuminance"),
                )
                .on_hover_text("Noon sun 100 000, overcast 20 000, sunset under 1 000");
            }
            Light::Point {
                color,
                lumens,
                range,
                casts_shadows,
            } => {
                color_row(ui, "Color", color);
                ui.add(
                    egui::Slider::new(lumens, 1.0..=100_000.0)
                        .logarithmic(true)
                        .suffix(" lm")
                        .text("Power"),
                )
                .on_hover_text("Domestic bulb 800, shop light 4 000, flood tens of thousands");
                range_row(ui, range);
                ui.checkbox(casts_shadows, "Casts shadows")
                    .on_hover_text("Six atlas tiles, and only if the budget reaches this light");
            }
            Light::Spot {
                color,
                lumens,
                range,
                inner_angle,
                outer_angle,
                reflector,
                casts_shadows,
            } => {
                color_row(ui, "Color", color);
                ui.add(
                    egui::Slider::new(lumens, 1.0..=100_000.0)
                        .logarithmic(true)
                        .suffix(" lm")
                        .text("Power"),
                )
                .on_hover_text("The bulb's own output, before the cone concentrates it");
                ui.checkbox(reflector, "Reflector").on_hover_text(
                    "On, the cone concentrates the power and narrowing it brightens the beam. \
                     Off, the cone only masks a bare bulb and the angle is free of brightness.",
                );
                range_row(ui, range);
                // Dragged together: the inner cone cannot leave the outer one,
                // and the falloff divides by their difference — so clamping
                // here is what keeps the editor from authoring a light the
                // shader has to defend itself against.
                ui.add(egui::Slider::new(outer_angle, 1.0..=89.0).text("Outer angle"))
                    .on_hover_text("Half angle from the axis, where the cone reaches nothing");
                *inner_angle = inner_angle.min(*outer_angle);
                ui.add(egui::Slider::new(inner_angle, 0.0..=*outer_angle).text("Inner angle"))
                    .on_hover_text("Half angle out to which it is still at full brightness");
                ui.checkbox(casts_shadows, "Casts shadows")
                    .on_hover_text("One atlas tile, and only if the budget reaches this light");
            }
        });
}
