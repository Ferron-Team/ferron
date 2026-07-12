//! C# scripting integration (engine side), behind the `scripting` feature.
//!
//! The generic ABI lives in `ferron-script`; the transform functions need the
//! engine's own `LocalTransform`, so they're defined here and assembled into the
//! table with `..ferron_script::default_api()`.

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use glam::{Quat, Vec3};

use ferron_ecs::{Entity, World};
use ferron_script::{CEntity, CTransform, FerronApi, ScriptHost};

use crate::scene::{
    Assets, InputState, LocalTransform, MaterialHandle, MeshHandle, Name, ScriptComponent, Tag,
    Time, Transform,
};

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

// --- input ------------------------------------------------------------------
// The `InputState` resource is engine-side, so these live here (like the
// transform functions) and read it through the active-world seam. Outside a
// dispatch window, or before the resource exists, they report "nothing held".

fn with_input(query: impl FnOnce(&InputState) -> bool) -> bool {
    ferron_script::with_world(false, |world| {
        world
            .get_resource::<InputState>()
            .is_some_and(|input| query(&input))
    })
}

extern "C" fn key_down(code: u32) -> bool {
    with_input(|input| input.key_down(code))
}

extern "C" fn key_pressed(code: u32) -> bool {
    with_input(|input| input.key_pressed(code))
}

extern "C" fn key_released(code: u32) -> bool {
    with_input(|input| input.key_released(code))
}

extern "C" fn mouse_button_down(button: u32) -> bool {
    with_input(|input| input.mouse_button_down(button))
}

extern "C" fn cursor_pos(x: *mut f32, y: *mut f32) {
    let (cx, cy) = ferron_script::with_world((0.0, 0.0), |world| {
        world
            .get_resource::<InputState>()
            .map_or((0.0, 0.0), |input| input.cursor())
    });
    if !x.is_null() {
        // SAFETY: C# passes valid, writable f32 pointers.
        unsafe { *x = cx };
    }
    if !y.is_null() {
        // SAFETY: as above.
        unsafe { *y = cy };
    }
}

// --- time ---------------------------------------------------------------------
// The `Time` resource is engine-side, so these live here (like the input
// functions) and read it through the active-world seam. Outside a dispatch
// window, or before the resource exists, they report zero.

fn with_time<R: Default>(query: impl FnOnce(&Time) -> R) -> R {
    ferron_script::with_world(R::default(), |world| {
        world
            .get_resource::<Time>()
            .map_or_else(R::default, |time| query(&time))
    })
}

extern "C" fn time_delta() -> f32 {
    with_time(|time| time.delta_time())
}

extern "C" fn time_total() -> f32 {
    with_time(|time| time.elapsed_time())
}

extern "C" fn time_frame_count() -> u64 {
    with_time(|time| time.frame_count())
}

// --- entity querying -----------------------------------------------------------
// Read-only world inspection for scripts. These are leaf calls: any RefCell
// borrow a query takes lives only inside the `with_world` closure and is
// released before control returns to C#, so no storage borrow is ever held
// across a dispatch. The find functions copy results into caller-owned memory
// for the same reason — results outlive the query borrow, never vice versa.
//
// `kind` numbering is lock-step with C# `Ferron.ComponentKind` (same rule as
// key codes: append, never renumber): 0 = Transform (LocalTransform), 1 = Tag.

extern "C" fn find_by_tag(tag: *const c_char, out: *mut CEntity) -> bool {
    if tag.is_null() || out.is_null() {
        return false;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let tag = unsafe { CStr::from_ptr(tag) }.to_string_lossy();
    ferron_script::with_world(false, |world| {
        let found = world.query::<&Tag>().find(|_, t| t.as_str() == tag.as_ref());

        match found {
            // SAFETY: `out` was null-checked above; C# pass a pointer to a
            // single stack allocated Entity slot (see Native.FindByTag)
            Some(e) => { unsafe { *out = CEntity { index: e.index, generation: e.generation } } true }
            None => return false
        }
    })
}

extern "C" fn find_all_by_tag(tag: *const c_char, out: *mut CEntity, capacity: i32) -> i32 {
    if tag.is_null() || (out.is_null() && capacity > 0) {
        return 0;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let tag = unsafe { CStr::from_ptr(tag) }.to_string_lossy();
    ferron_script::with_world(0, |world| {
        let mut matches: Vec<CEntity> = Vec::new();
        world.query::<&Tag>().for_each(|e, t| {
            if t.as_str() == tag.as_ref() {
                matches.push(CEntity { index: e.index, generation: e.generation });
            }
        });

        let n = matches.len().min(capacity.max(0) as usize);
        if n > 0 {
            // SAFETY: C# guarantees `out` points at `capacity` writable CEntity slots
            // (pinned managed Entity[] in Native.FindAllByTag); src is our own Vec,
            // so the ranges cannot overlap.
            unsafe { std::ptr::copy_nonoverlapping(matches.as_ptr(), out, n) };
        }
        matches.len() as i32
    })
}

extern "C" fn has_component(entity: CEntity, kind: u32) -> bool {
    ferron_script::with_world(false, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        // TODO: match `kind` per the numbering above (0 =>
        // world.has::<LocalTransform>(entity), 1 => world.has::<Tag>(entity)),
        // returning false for unknown kinds — a newer C# assembly probing an
        // older engine must degrade gracefully, not misbehave.
        todo!("map kind to a world.has::<T>() check")
    })
}

// String out-param protocol: returns the tag's UTF-8 byte length (no nul
// terminator — the return value carries the size), or -1 if the entity has no
// Tag. Writes min(len, capacity) bytes; C# retries with an exact-size buffer
// when len > capacity.
extern "C" fn get_tag(entity: CEntity, out: *mut c_char, capacity: i32) -> i32 {
    if out.is_null() && capacity > 0 {
        return -1;
    }
    ferron_script::with_world(-1, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        // TODO: world.get::<Tag>(entity); if present, copy up to `capacity`
        // bytes of its UTF-8 into `out` (SAFETY to uphold: C# guarantees
        // `capacity` writable bytes) and return the full byte length; -1 when
        // the component is absent or the handle is stale.
        todo!("copy tag bytes into out, return full byte length")
    })
}

extern "C" fn set_tag(entity: CEntity, tag: *const c_char) -> bool {
    if tag.is_null() {
        return false;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let tag = unsafe { CStr::from_ptr(tag) }.to_string_lossy().into_owned();
    ferron_script::with_world(false, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        if !world.is_alive(entity) {
            return false;
        }
        COMMANDS.with(|commands| commands.borrow_mut().push(Command::SetTag { entity, tag }));
        true
    })
}

// --- deferred structural changes ---------------------------------------------
// Structural edits requested from inside a script dispatch are queued and
// applied by `apply_commands` once the dispatch window closes. Direct mutation
// happens to be safe today (the tick holds no borrows while dispatching), but
// deferral keeps two hazards off the table for good: a despawn dropping a
// `ScriptComponent` (and freeing its GCHandle) while this tick's handle list
// still references it, and any future engine code that holds borrows during
// dispatch. Entity ids are still reserved eagerly — the allocator touches no
// component storage — so scripts get a real handle back synchronously.

enum Command {
    SpawnRenderable {
        entity: Entity,
        mesh: MeshHandle,
        material: MaterialHandle,
        transform: CTransform,
    },
    Despawn(Entity),
    // Adding a component changes which entities queries match, so it defers
    // like the other structural edits; a same-tick FindByTag won't see it.
    SetTag { entity: Entity, tag: String },
}

thread_local! {
    static COMMANDS: RefCell<Vec<Command>> = const { RefCell::new(Vec::new()) };
}

extern "C" fn spawn_renderable(
    mesh: *const c_char,
    material: *const c_char,
    transform: *const CTransform,
) -> CEntity {
    if mesh.is_null() || material.is_null() || transform.is_null() {
        return CEntity::NULL;
    }
    // SAFETY: C# passes valid, null-terminated UTF-8 buffers and a valid transform.
    let mesh_name = unsafe { CStr::from_ptr(mesh) }.to_string_lossy();
    let material_name = unsafe { CStr::from_ptr(material) }.to_string_lossy();
    let transform = unsafe { *transform };

    ferron_script::with_world(CEntity::NULL, |world| {
        // Resolve asset handles now so a bad name fails loudly at the call
        // site instead of silently when the queue drains.
        let (mesh, material) = {
            let Some(assets) = world.get_resource::<Assets>() else {
                eprintln!("[script] spawn_renderable: no Assets resource");
                return CEntity::NULL;
            };
            match (assets.mesh(&mesh_name), assets.material(&material_name)) {
                (Some(mesh), Some(material)) => (mesh, material),
                (mesh, material) => {
                    if mesh.is_none() {
                        eprintln!("[script] spawn_renderable: unknown mesh {mesh_name:?}");
                    }
                    if material.is_none() {
                        eprintln!("[script] spawn_renderable: unknown material {material_name:?}");
                    }
                    return CEntity::NULL;
                }
            }
        };

        let entity = world.spawn();
        COMMANDS.with(|commands| {
            commands.borrow_mut().push(Command::SpawnRenderable {
                entity,
                mesh,
                material,
                transform,
            })
        });
        CEntity {
            index: entity.index,
            generation: entity.generation,
        }
    })
}

extern "C" fn despawn(entity: CEntity) -> bool {
    ferron_script::with_world(false, |world| {
        let entity = Entity {
            index: entity.index,
            generation: entity.generation,
        };
        if !world.is_alive(entity) {
            return false;
        }
        COMMANDS.with(|commands| commands.borrow_mut().push(Command::Despawn(entity)));
        true
    })
}

/// Apply the structural changes scripts queued during a dispatch. Runs with no
/// other world borrows held, so a despawn can drop a `ScriptComponent` (and
/// free its GCHandle) safely.
fn apply_commands(world: &mut World) {
    let commands: Vec<Command> = COMMANDS.with(|c| c.borrow_mut().drain(..).collect());
    for command in commands {
        match command {
            Command::SpawnRenderable {
                entity,
                mesh,
                material,
                transform,
            } => {
                world.insert(entity, Name::new(format!("Scripted {}", entity.index)));
                world.insert(
                    entity,
                    LocalTransform::from(Transform {
                        translation: Vec3::from_array(transform.position),
                        rotation: Quat::from_array(transform.rotation),
                        scale: Vec3::from_array(transform.scale),
                    }),
                );
                world.insert(entity, mesh);
                world.insert(entity, material);
            }
            Command::Despawn(entity) => {
                world.despawn(entity);
            }
            Command::SetTag { entity, tag } => {
                // `insert` is a stale-handle no-op, so a despawn queued earlier
                // this tick wins — same rule as the other commands.
                world.insert(entity, Tag::new(tag));
            }
        }
    }
}

fn build_api() -> FerronApi {
    FerronApi {
        get_transform,
        set_transform,
        key_down,
        key_pressed,
        key_released,
        mouse_button_down,
        cursor_pos,
        spawn_renderable,
        despawn,
        time_delta,
        time_total,
        time_frame_count,
        find_by_tag,
        find_all_by_tag,
        has_component,
        get_tag,
        set_tag,
        ..ferron_script::default_api()
    }
}

pub struct Scripting {
    host: ScriptHost,
}

impl Scripting {
    /// Locate the built `Ferron` managed assembly: `FERRON_SCRIPT_DIR` wins if
    /// set; otherwise probe `scripting/Ferron/bin/{Debug,Release}/net*`
    /// (relative to the working directory) and pick the most recently built.
    pub fn find_assembly_dir() -> Option<PathBuf> {
        if let Ok(dir) = std::env::var("FERRON_SCRIPT_DIR") {
            return Some(PathBuf::from(dir));
        }
        let mut best: Option<(SystemTime, PathBuf)> = None;
        for config in ["Debug", "Release"] {
            let bin = Path::new("scripting/Ferron/bin").join(config);
            let Ok(entries) = std::fs::read_dir(&bin) else {
                continue;
            };
            for entry in entries.flatten() {
                let dir = entry.path();
                let dll = dir.join("Ferron.dll");
                if !dir.join("Ferron.runtimeconfig.json").is_file() || !dll.is_file() {
                    continue;
                }
                let modified = dll
                    .metadata()
                    .and_then(|meta| meta.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                if best.as_ref().map_or(true, |(t, _)| modified > *t) {
                    best = Some((modified, dir));
                }
            }
        }
        best.map(|(_, dir)| dir)
    }

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
                    // Desired-on from birth; `active` stays false until the
                    // first tick actually dispatches OnEnable.
                    enabled: true,
                    active: false,
                },
            );
        }
    }

    /// Request an activation change; the transition (OnEnable/OnDisable) is
    /// dispatched by the next `tick`, not here.
    pub fn set_enabled(&self, world: &mut World, entity: Entity, enabled: bool) {
        if let Some(mut script) = world.get_mut::<ScriptComponent>(entity) {
            script.enabled = enabled;
        }
    }

    /// Tick every script. Collect handles first, drop the world borrow, then
    /// dispatch — so the ABI's `&mut World` reconstruction never aliases.
    pub fn tick(&self, world: &mut World, delta_time: f32) {
        struct Pending {
            entity: Entity,
            handle: u64,
            started: bool,
            enabled: bool,
            active: bool,
        }

        let mut pending: Vec<Pending> = Vec::new();
        world.query::<&ScriptComponent>().for_each(|entity, script| {
            pending.push(Pending {
                entity,
                handle: script.handle,
                started: script.started,
                enabled: script.enabled,
                active: script.active,
            })
        });
        if pending.is_empty() {
            return;
        }

        ferron_script::with_active_world(world, || {
            for script in &mut pending {
                if script.enabled && !script.active {
                    self.host.enable(script.handle);
                    script.active = true;
                    if !script.started {
                        self.host.start(script.handle);
                        script.started = true;
                    }
                } else if !script.enabled && script.active {
                    self.host.disable(script.handle);
                    script.active = false;
                    continue; // Skip tick update
                }

                if script.active {
                    self.host.update(script.handle, delta_time);
                }
            }
        });

        // Structural changes the scripts queued land now, after every script
        // has run — so this frame's extraction already sees new renderables.
        apply_commands(world);

        for script in &pending {
            if let Some(mut component) = world.get_mut::<ScriptComponent>(script.entity) {
                component.active = script.active;
                component.started = script.started;
            }
        }
    }
}
