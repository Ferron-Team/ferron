//! C# scripting integration (engine side), behind the `scripting` feature.
//!
//! The generic ABI lives in `ferron-script`; the transform functions need the
//! engine's own `LocalTransform`, so they're defined here and assembled into the
//! table with `..ferron_script::default_api()`.

use std::ffi::CString;
use std::path::Path;

use glam::{Quat, Vec3};

use ferron_ecs::{Entity, World};
use ferron_script::{CEntity, CTransform, FerronApi, ScriptHost};

use crate::scene::{LocalTransform, ScriptComponent};

extern "C" fn get_transform(entity: CEntity, out: *mut CTransform) -> bool {
    if out.is_null() {
        return false;
    }
    ferron_script::with_world(false, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        let Some(transform) = world.get::<LocalTransform>(entity) else {
            return false;
        };
        // SAFETY: `out` is a valid, writable `CTransform` supplied by C#.
        unsafe {
            *out = CTransform {
                position: transform.translation.to_array(),
                rotation: transform.rotation.to_array(),
                scale: transform.scale.to_array(),
            };
        }
        true
    })
}

extern "C" fn set_transform(entity: CEntity, value: *const CTransform) -> bool {
    if value.is_null() {
        return false;
    }
    // SAFETY: `value` is a valid `CTransform` supplied by C#.
    let value = unsafe { *value };
    ferron_script::with_world(false, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        let Some(mut transform) = world.get_mut::<LocalTransform>(entity) else {
            return false;
        };
        transform.translation = Vec3::from_array(value.position);
        transform.rotation = Quat::from_array(value.rotation);
        transform.scale = Vec3::from_array(value.scale);
        true
    })
}

fn build_api() -> FerronApi {
    FerronApi {
        get_transform,
        set_transform,
        ..ferron_script::default_api()
    }
}

pub struct Scripting {
    host: ScriptHost,
}

impl Scripting {
    /// Boot the runtime, loading the managed assembly from `assembly_dir`.
    /// Returns `None` (with a logged reason) if the runtime can't start.
    pub fn boot(assembly_dir: &Path) -> Option<Self> {
        match ScriptHost::boot(&build_api(), assembly_dir) {
            Ok(host) => Some(Self { host }),
            Err(err) => {
                eprintln!("scripting disabled: {err}");
                None
            }
        }
    }

    /// Attach a C# `Behaviour` (by assembly-qualified type name) to `entity`.
    pub fn attach(&self, world: &mut World, entity: Entity, type_name: &str) {
        let Ok(name) = CString::new(type_name) else {
            return;
        };
        let handle = self.host.create(
            CEntity {
                index: entity.index,
                generation: entity.generation,
            },
            &name,
        );
        if handle != 0 {
            world.insert(
                entity,
                ScriptComponent {
                    handle,
                    started: false,
                },
            );
        }
    }

    /// Tick every script. Collect handles first, drop the world borrow, then
    /// dispatch — so the ABI's `&mut World` reconstruction never aliases.
    pub fn tick(&self, world: &mut World, delta_time: f32) {
        let mut pending: Vec<(Entity, u64, bool)> = Vec::new();
        world
            .query::<&ScriptComponent>()
            .for_each(|entity, script| pending.push((entity, script.handle, script.started)));
        if pending.is_empty() {
            return;
        }

        ferron_script::with_active_world(world, || {
            for &(_, handle, started) in &pending {
                if !started {
                    self.host.start(handle);
                }
                self.host.update(handle, delta_time);
            }
        });

        for (entity, _, started) in pending {
            if !started {
                if let Some(mut script) = world.get_mut::<ScriptComponent>(entity) {
                    script.started = true;
                }
            }
        }
    }
}
