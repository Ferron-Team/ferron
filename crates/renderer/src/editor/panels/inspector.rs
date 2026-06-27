//! Right panel: edit the selected entity. Each section takes a fresh,
//! short-lived borrow of the world so the `RefCell` component storage never
//! double-borrows.

use glam::{EulerRot, Quat};

use ferron_ecs::{Entity, World};

use super::{color_row, vec3_row};
use crate::editor::state::EditorState;
use crate::scene::{Assets, Light, LocalTransform, MaterialHandle, MeshHandle, Name};

pub fn show(ctx: &egui::Context, world: &mut World, state: &mut EditorState) {
    egui::SidePanel::right("inspector")
        .resizable(true)
        .default_width(280.0)
        .show(ctx, |ui| {
            ui.add_space(4.0);
            ui.heading("Inspector");
            ui.separator();

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
        });
}

fn name_section(ui: &mut egui::Ui, world: &World, entity: Entity) {
    if let Some(mut name) = world.get_mut::<Name>(entity) {
        ui.horizontal(|ui| {
            ui.label("Name");
            ui.text_edit_singleline(&mut name.0);
        });
    }
    ui.weak(format!("id {}", entity.index()));
}

fn transform_section(ui: &mut egui::Ui, world: &World, entity: Entity) {
    let Some(mut t) = world.get_mut::<LocalTransform>(entity) else {
        return;
    };
    egui::CollapsingHeader::new("Transform")
        .default_open(true)
        .show(ui, |ui| {
            vec3_row(ui, "Position", &mut t.translation, 0.05);

            // Rotation is edited as XYZ Euler degrees, rebuilt into the quat.
            let (rx, ry, rz) = t.rotation.to_euler(EulerRot::XYZ);
            let mut euler = [rx.to_degrees(), ry.to_degrees(), rz.to_degrees()];
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
                    EulerRot::XYZ,
                    euler[0].to_radians(),
                    euler[1].to_radians(),
                    euler[2].to_radians(),
                );
            }

            vec3_row(ui, "Scale", &mut t.scale, 0.05);
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

fn light_section(ui: &mut egui::Ui, world: &World, entity: Entity) {
    let Some(mut light) = world.get_mut::<Light>(entity) else {
        return;
    };
    egui::CollapsingHeader::new("Light")
        .default_open(true)
        .show(ui, |ui| match &mut *light {
            Light::Directional { color, intensity } => {
                color_row(ui, "Color", color);
                ui.add(egui::Slider::new(intensity, 0.0..=20.0).text("Intensity"));
            }
            Light::Point {
                color,
                intensity,
                range,
            } => {
                color_row(ui, "Color", color);
                ui.add(egui::Slider::new(intensity, 0.0..=50.0).text("Intensity"));
                ui.add(egui::Slider::new(range, 0.0..=100.0).text("Range"));
            }
        });
}
