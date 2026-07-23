//! C# scripting host for the Ferron engine.
//!
//! [`ScriptHost`] boots CoreCLR, loads the `Ferron` assembly, hands it the engine
//! function table ([`FerronApi`]), and exposes the lifecycle entry points
//! (`Create`/`Enable`/`Start`/`Update`/`Disable`, plus the global
//! [`destroy_handle`]) the engine drives. Generic, World-only ABI functions
//! live here; component-specific ones are supplied by the engine via
//! `FerronApi { get_transform, set_transform, ..default_api() }`.

use std::cell::Cell;
use std::ffi::{c_char, CStr};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use ferron_ecs::World;
use netcorehost::hostfxr::{HostfxrContext, InitializedForRuntimeConfig};
use netcorehost::{nethost, pdcstr};

/// Engine function table handed to managed code at startup. Field order and
/// signatures must stay in lock-step with the C# `Ferron.FerronApi` struct.
#[repr(C)]
pub struct FerronApi {
    pub log: extern "C" fn(*const c_char),
    pub spawn: extern "C" fn() -> CEntity,
    pub get_transform: extern "C" fn(CEntity, *mut CTransform) -> bool,
    pub set_transform: extern "C" fn(CEntity, *const CTransform) -> bool,
    // Input polling; key/button codes are the engine's stable numbering (see
    // the renderer's `scene::input::map_key` and C# `Ferron.KeyCode`).
    pub key_down: extern "C" fn(u32) -> bool,
    pub key_pressed: extern "C" fn(u32) -> bool,
    pub key_released: extern "C" fn(u32) -> bool,
    pub mouse_button_down: extern "C" fn(u32) -> bool,
    pub cursor_pos: extern "C" fn(*mut f32, *mut f32),
    // Structural ops. The engine queues these and applies them after the
    // dispatch window closes, but `spawn_renderable` reserves and returns a
    // real entity id immediately.
    pub spawn_renderable:
        extern "C" fn(*const c_char, *const c_char, *const CTransform) -> CEntity,
    pub despawn: extern "C" fn(CEntity) -> bool,
    // Frame timing, read from the engine's `Time` resource. Engine-side (like
    // input), so the renderer supplies the real implementations.
    pub time_delta: extern "C" fn() -> f32,
    pub time_total: extern "C" fn() -> f32,
    pub time_frame_count: extern "C" fn() -> u64,
    // Entity querying. Engine-side (the `Tag` component and the component-kind
    // numbering live in the renderer). `find_all_by_tag` and `get_tag` write
    // into caller-allocated buffers and return the *total* count/byte length —
    // snprintf semantics, so C# can resize and retry when the buffer was too
    // small. `has_component` takes the kind numbering shared with C#
    // `Ferron.ComponentKind`. `set_tag` is a deferred structural change, like
    // `despawn`.
    pub find_by_tag: extern "C" fn(*const c_char, *mut CEntity) -> bool,
    pub find_all_by_tag: extern "C" fn(*const c_char, *mut CEntity, i32) -> i32,
    pub has_component: extern "C" fn(CEntity, u32) -> bool,
    pub get_tag: extern "C" fn(CEntity, *mut c_char, i32) -> i32,
    pub set_tag: extern "C" fn(CEntity, *const c_char) -> bool,
    // Collision + composition. All four are deferred structural changes (like
    // `set_tag`): they validate eagerly and apply after the dispatch window.
    // Box half extents are passed as three floats to keep the signature
    // blittable-trivial; `bool` is one byte on both sides (C# `byte`).
    pub add_box_collider: extern "C" fn(CEntity, f32, f32, f32, bool) -> bool,
    pub add_sphere_collider: extern "C" fn(CEntity, f32, bool) -> bool,
    pub set_material: extern "C" fn(CEntity, *const c_char) -> bool,
    pub add_script: extern "C" fn(CEntity, *const c_char) -> bool,
    // Developer debug utilities. `log_warn`/`log_error` mirror `log` at higher
    // severity; the engine routes all three into its editor console. Logging is
    // always live. `debug_draw_line` records one world-space line into the
    // engine's per-frame overlay buffer: endpoints, RGBA, then a lifetime in
    // seconds (<= 0 = a single frame). Editor-only — no-op in export builds.
    // The line is passed as loose floats to stay blittable-trivial, the same
    // choice as the collider extents.
    pub log_warn: extern "C" fn(*const c_char),
    pub log_error: extern "C" fn(*const c_char),
    pub debug_draw_line:
        extern "C" fn(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32),
}

/// A table with the generic functions wired and the rest stubbed; the engine
/// overrides `get_transform`/`set_transform` and the input functions with real
/// implementations (they touch engine-side resources).
pub fn default_api() -> FerronApi {
    FerronApi {
        log: ferron_log,
        spawn: ferron_spawn,
        get_transform: stub_get_transform,
        set_transform: stub_set_transform,
        key_down: stub_key_query,
        key_pressed: stub_key_query,
        key_released: stub_key_query,
        mouse_button_down: stub_key_query,
        cursor_pos: stub_cursor_pos,
        spawn_renderable: stub_spawn_renderable,
        despawn: stub_despawn,
        time_delta: stub_time_seconds,
        time_total: stub_time_seconds,
        time_frame_count: stub_time_frame_count,
        find_by_tag: stub_find_by_tag,
        find_all_by_tag: stub_find_all_by_tag,
        has_component: stub_has_component,
        get_tag: stub_get_tag,
        set_tag: stub_set_tag,
        add_box_collider: stub_add_box_collider,
        add_sphere_collider: stub_add_sphere_collider,
        set_material: stub_set_material,
        add_script: stub_add_script,
        log_warn: ferron_log_warn,
        log_error: ferron_log_error,
        debug_draw_line: stub_debug_draw_line,
    }
}

extern "C" fn stub_debug_draw_line(
    _fx: f32,
    _fy: f32,
    _fz: f32,
    _tx: f32,
    _ty: f32,
    _tz: f32,
    _r: f32,
    _g: f32,
    _b: f32,
    _a: f32,
    _duration: f32,
) {
}

extern "C" fn stub_add_box_collider(
    _entity: CEntity,
    _hx: f32,
    _hy: f32,
    _hz: f32,
    _is_trigger: bool,
) -> bool {
    false
}

extern "C" fn stub_add_sphere_collider(_entity: CEntity, _radius: f32, _is_trigger: bool) -> bool {
    false
}

extern "C" fn stub_set_material(_entity: CEntity, _material: *const c_char) -> bool {
    false
}

extern "C" fn stub_add_script(_entity: CEntity, _type_name: *const c_char) -> bool {
    false
}

extern "C" fn stub_find_by_tag(_tag: *const c_char, _out: *mut CEntity) -> bool {
    false
}

extern "C" fn stub_find_all_by_tag(_tag: *const c_char, _out: *mut CEntity, _capacity: i32) -> i32 {
    0
}

extern "C" fn stub_has_component(_entity: CEntity, _kind: u32) -> bool {
    false
}

extern "C" fn stub_get_tag(_entity: CEntity, _out: *mut c_char, _capacity: i32) -> i32 {
    -1
}

extern "C" fn stub_set_tag(_entity: CEntity, _tag: *const c_char) -> bool {
    false
}

extern "C" fn stub_time_seconds() -> f32 {
    0.0
}

extern "C" fn stub_time_frame_count() -> u64 {
    0
}

extern "C" fn stub_spawn_renderable(
    _mesh: *const c_char,
    _material: *const c_char,
    _transform: *const CTransform,
) -> CEntity {
    CEntity::NULL
}

extern "C" fn stub_despawn(_entity: CEntity) -> bool {
    false
}

extern "C" fn stub_get_transform(_entity: CEntity, _out: *mut CTransform) -> bool {
    false
}

extern "C" fn stub_set_transform(_entity: CEntity, _value: *const CTransform) -> bool {
    false
}

extern "C" fn stub_key_query(_code: u32) -> bool {
    false
}

extern "C" fn stub_cursor_pos(x: *mut f32, y: *mut f32) {
    if !x.is_null() {
        // SAFETY: C# passes valid, writable f32 pointers.
        unsafe { *x = 0.0 };
    }
    if !y.is_null() {
        // SAFETY: as above.
        unsafe { *y = 0.0 };
    }
}

/// Logging callback C# invokes through [`FerronApi::log`].
pub extern "C" fn ferron_log(message: *const c_char) {
    if message.is_null() {
        return;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    println!("[c#] {text}");
}

/// Default warning sink (the engine overrides it to route into the editor
/// console). Prints to stderr so a script's diagnostics are still visible when
/// no console is present.
pub extern "C" fn ferron_log_warn(message: *const c_char) {
    if message.is_null() {
        return;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    eprintln!("[c# WARN] {text}");
}

/// Default error sink; see [`ferron_log_warn`].
pub extern "C" fn ferron_log_error(message: *const c_char) {
    if message.is_null() {
        return;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    eprintln!("[c# ERROR] {text}");
}

/// Spawn a new, empty entity and return its handle.
pub extern "C" fn ferron_spawn() -> CEntity {
    with_world(CEntity::NULL, |world| {
        let entity = world.spawn();
        CEntity {
            index: entity.index,
            generation: entity.generation,
        }
    })
}

thread_local! {
    static ACTIVE_WORLD: Cell<*mut World> = const { Cell::new(std::ptr::null_mut()) };
}

/// Install `world` as the active world while `dispatch` runs, then clear it.
///
/// The engine must hold no other borrow of this `World` for the duration of
/// `dispatch` — the ABI functions reconstruct an exclusive `&mut World` from the
/// stored pointer. (Hence the tick collects handles first, then dispatches.)
pub fn with_active_world<R>(world: &mut World, dispatch: impl FnOnce() -> R) -> R {
    struct Clear;
    impl Drop for Clear {
        fn drop(&mut self) {
            ACTIVE_WORLD.with(|w| w.set(std::ptr::null_mut()));
        }
    }
    ACTIVE_WORLD.with(|w| w.set(world as *mut World));
    let _clear = Clear;
    dispatch()
}

/// Run `op` against the active world, or return `default` if none is installed
/// (an ABI call outside a dispatch window).
pub fn with_world<R>(default: R, op: impl FnOnce(&mut World) -> R) -> R {
    let ptr = ACTIVE_WORLD.with(|w| w.get());
    if ptr.is_null() {
        return default;
    }
    // SAFETY: within a dispatch window `ptr` is a valid, uniquely-owned
    // `&mut World`; scripts are single-threaded and this borrow lives only for
    // the leaf `op` call, so it never aliases another.
    op(unsafe { &mut *ptr })
}

/// Signature of the C# `Ferron.Behaviours.Destroy(nint)` entry point; `extern
/// "system"` matches the convention netcorehost hands back.
type DestroyFn = extern "system" fn(u64);

static DESTROY_HANDLE: AtomicUsize = AtomicUsize::new(0);

/// Install the managed teardown callback (called once by [`ScriptHost::boot`]).
pub fn set_destroy_handle(destroy: DestroyFn) {
    DESTROY_HANDLE.store(destroy as usize, Ordering::Release);
}

/// Tear down the behaviour behind `handle`: the managed side fires
/// OnDisable (if still active) and OnDestroy, then frees the `GCHandle` — in
/// that order, enforced in one place (C# `Behaviours.Destroy`). A no-op before
/// the host is booted, so it's safe to call unconditionally from
/// `ScriptComponent::drop`.
///
/// This usually runs from `Drop`, *outside* a dispatch window: `ACTIVE_WORLD`
/// is null, so any engine API the callbacks touch returns defaults instead of
/// aliasing whatever `&mut World` triggered the despawn. Safe, but it means
/// OnDestroy cannot usefully read the world — document that as a script-facing
/// limitation (Unity has an equivalent one).
pub fn destroy_handle(handle: u64) {
    let ptr = DESTROY_HANDLE.load(Ordering::Acquire);
    if ptr != 0 {
        // SAFETY: only ever set by `set_destroy_handle` from a valid C# `DestroyFn`.
        let destroy: DestroyFn = unsafe { std::mem::transmute::<usize, DestroyFn>(ptr) };
        destroy(handle);
    }
}

/// C ABI mirror of `ferron_ecs::Entity` (blittable: two `u32`s).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CEntity {
    pub index: u32,
    pub generation: u32,
}

impl CEntity {
    /// Returned when an ABI call can't produce a real entity. `u32::MAX` is
    /// never handed out by the ECS allocator (it would require that many live
    /// slots), so this can't collide with a real handle the way `{0, 0}` did —
    /// `{0, 0}` is the first entity ever spawned. The `SparseSet` already treats
    /// a `u32::MAX` index as its empty `SENTINEL`, so every lookup rejects this
    /// handle as "not found" rather than aliasing a live entity. C# mirrors it
    /// as `Ferron.Entity.Null`; keep the two in sync.
    pub const NULL: Self = Self {
        index: u32::MAX,
        generation: u32::MAX,
    };
}

/// C ABI transform: position, rotation (quaternion `xyzw`), scale. Matches the
/// C# `Ferron.Transform`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CTransform {
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

/// C ABI collision event payload, as seen by one participant: the *other*
/// entity, the world-space contact point, and the contact normal pointing from
/// the receiving entity toward `other`. Matches the C# `Ferron.Collision`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CCollision {
    pub other: CEntity,
    pub point: [f32; 3],
    pub normal: [f32; 3],
}

/// The booted .NET runtime plus the managed lifecycle entry points. Holding the
/// `HostfxrContext` keeps the runtime resident for the host's lifetime.
pub struct ScriptHost {
    _context: HostfxrContext<InitializedForRuntimeConfig>,
    create_fn: extern "system" fn(CEntity, *const c_char) -> u64,
    // The lifecycle hooks return a fault byte (0 = clean, nonzero = the C# hook
    // threw and was contained): the managed side never lets an exception cross
    // into these native frames, and reports it here so the engine can disable
    // the offending script. Signatures stay in lock-step with C#
    // `Ferron.Behaviours` (which returns `byte`).
    start_fn: extern "system" fn(u64) -> u8,
    update_fn: extern "system" fn(u64, f32) -> u8,
    enable_fn: extern "system" fn(u64) -> u8,
    disable_fn: extern "system" fn(u64) -> u8,
    collision_enter_fn: extern "system" fn(u64, *const CCollision) -> u8,
    collision_exit_fn: extern "system" fn(u64, *const CCollision) -> u8,
}

impl ScriptHost {
    /// Boot CoreCLR and load `Ferron.dll` from `assembly_dir`, handing C# the
    /// `api` table. `assembly_dir` must contain `Ferron.dll` and
    /// `Ferron.runtimeconfig.json`.
    pub fn boot(api: &FerronApi, assembly_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        // hostfxr resolves paths relative to the working directory, and
        // netcorehost's runtime path type isn't public here, so load with the cwd
        // pointed at the assembly dir, then restore it.
        let previous = std::env::current_dir()?;
        std::env::set_current_dir(assembly_dir)?;
        let result = Self::load(api);
        let _ = std::env::set_current_dir(previous);
        result
    }

    fn load(api: &FerronApi) -> Result<Self, Box<dyn std::error::Error>> {
        let hostfxr = nethost::load_hostfxr()?;
        let context = hostfxr.initialize_for_runtime_config(pdcstr!("Ferron.runtimeconfig.json"))?;

        // Deref each `ManagedFunction` to its raw `extern "system"` fn pointer.
        // The loader is scoped so its borrow of `context` ends before the move.
        #[rustfmt::skip]
        let (
            init, create_fn, start_fn, update_fn, enable_fn, disable_fn,
            collision_enter_fn, collision_exit_fn, destroy,
        ) = {
            let loader = context.get_delegate_loader_for_assembly(pdcstr!("Ferron.dll"))?;
            (
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(*const FerronApi) -> i32>(
                    pdcstr!("Ferron.Bootstrap, Ferron"),
                    pdcstr!("Init"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(CEntity, *const c_char) -> u64>(
                    pdcstr!("Ferron.Behaviours, Ferron"),
                    pdcstr!("Create"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64) -> u8>(
                    pdcstr!("Ferron.Behaviours, Ferron"),
                    pdcstr!("Start"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64, f32) -> u8>(
                    pdcstr!("Ferron.Behaviours, Ferron"),
                    pdcstr!("Update"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64) -> u8>(
                    pdcstr!("Ferron.Behaviours, Ferron"),
                    pdcstr!("Enable"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64) -> u8>(
                    pdcstr!("Ferron.Behaviours, Ferron"),
                    pdcstr!("Disable"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64, *const CCollision) -> u8>(
                    pdcstr!("Ferron.Behaviours, Ferron"),
                    pdcstr!("CollisionEnter"),
                )?,
                *loader.get_function_with_unmanaged_callers_only::<extern "system" fn(u64, *const CCollision) -> u8>(
                    pdcstr!("Ferron.Behaviours, Ferron"),
                    pdcstr!("CollisionExit"),
                )?,
                // Teardown is a process-wide global, not a `ScriptHost` method:
                // it must stay reachable from `ScriptComponent::drop`, which
                // has no `&Scripting` in scope.
                *loader.get_function_with_unmanaged_callers_only::<DestroyFn>(
                    pdcstr!("Ferron.Behaviours, Ferron"),
                    pdcstr!("Destroy"),
                )?,
            )
        };

        let status = init(api);
        if status != 0 {
            return Err(format!("Ferron.Bootstrap.Init returned {status}").into());
        }
        set_destroy_handle(destroy);

        Ok(Self {
            _context: context,
            create_fn,
            start_fn,
            update_fn,
            enable_fn,
            disable_fn,
            collision_enter_fn,
            collision_exit_fn,
        })
    }

    /// Instantiate a `Behaviour` by assembly-qualified type name, attached to
    /// `entity`. Returns its `GCHandle` (as `u64`), or `0` on failure.
    pub fn create(&self, entity: CEntity, type_name: &CStr) -> u64 {
        (self.create_fn)(entity, type_name.as_ptr())
    }

    // Each hook returns `true` if the C# side caught an exception (the script is
    // now faulted and should not be dispatched to again until the fault clears).

    pub fn start(&self, handle: u64) -> bool {
        (self.start_fn)(handle) != 0
    }

    pub fn update(&self, handle: u64, delta_time: f32) -> bool {
        (self.update_fn)(handle, delta_time) != 0
    }

    pub fn enable(&self, handle: u64) -> bool {
        (self.enable_fn)(handle) != 0
    }

    pub fn disable(&self, handle: u64) -> bool {
        (self.disable_fn)(handle) != 0
    }

    pub fn collision_enter(&self, handle: u64, collision: &CCollision) -> bool {
        (self.collision_enter_fn)(handle, collision) != 0
    }

    pub fn collision_exit(&self, handle: u64, collision: &CCollision) -> bool {
        (self.collision_exit_fn)(handle, collision) != 0
    }
}
