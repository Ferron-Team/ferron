//! A fly-through editor camera, like a game engine's scene view: hold the right
//! mouse button to look (WASD to move, Q/E down/up, Shift to go faster), and
//! scroll to dolly forward/back.
//!
//! It drives the [`Camera`] resource. Movement is integrated per frame from held
//! keys, so [`update`](CameraController::update) needs the frame delta. Input is
//! gated on egui: events the editor wants (a click on a panel, typing in a field)
//! are passed through with `egui_wants = true` and ignored here.

use glam::Vec3;
use winit::event::{DeviceEvent, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::scene::Camera;

/// Keep the camera from flipping over at the poles (just under 90°).
const PITCH_LIMIT: f32 = 1.55;

#[derive(Default)]
struct Keys {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    fast: bool,
}

impl Keys {
    fn moving(&self) -> bool {
        self.forward || self.back || self.left || self.right || self.up || self.down
    }
}

pub struct CameraController {
    yaw: f32,
    pitch: f32,
    move_speed: f32,
    /// Multiplier applied while Shift is held.
    boost: f32,
    /// Radians of rotation per pixel of mouse motion.
    sensitivity: f32,
    /// World units dollied per scroll-wheel line.
    scroll_speed: f32,

    looking: bool,
    keys: Keys,
    // Input accumulated since the last `update`, then cleared.
    look_delta: (f32, f32),
    scroll: f32,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            move_speed: 6.0,
            boost: 4.0,
            sensitivity: 0.0025,
            scroll_speed: 1.5,
            looking: false,
            keys: Keys::default(),
            look_delta: (0.0, 0.0),
            scroll: 0.0,
        }
    }
}

impl CameraController {
    pub fn new() -> Self {
        Self::default()
    }

    /// Seed yaw/pitch from a camera's current facing so control starts smoothly.
    pub fn sync_from(&mut self, camera: &Camera) {
        let dir = (camera.target - camera.position).normalize_or_zero();
        if dir.length_squared() > 1e-6 {
            self.yaw = dir.x.atan2(-dir.z);
            self.pitch = dir.y.clamp(-1.0, 1.0).asin();
        }
    }

    /// `true` while the right mouse button is held for look mode; the caller uses
    /// this to hide/grab the cursor.
    pub fn looking(&self) -> bool {
        self.looking
    }

    /// Feed a window event. `egui_wants` is the editor's "I want this event" flag:
    /// presses it claims are ignored, but releases are always honored so keys and
    /// look mode never get stuck.
    pub fn process_window_event(&mut self, event: &WindowEvent, egui_wants: bool) {
        match event {
            WindowEvent::MouseInput {
                button: MouseButton::Right,
                state,
                ..
            } => {
                if state.is_pressed() {
                    self.looking = !egui_wants;
                } else {
                    self.looking = false;
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = state.is_pressed();
                if pressed && egui_wants {
                    return; // typing in a text field, not driving the camera
                }
                match code {
                    KeyCode::KeyW => self.keys.forward = pressed,
                    KeyCode::KeyS => self.keys.back = pressed,
                    KeyCode::KeyA => self.keys.left = pressed,
                    KeyCode::KeyD => self.keys.right = pressed,
                    KeyCode::KeyE => self.keys.up = pressed,
                    KeyCode::KeyQ => self.keys.down = pressed,
                    KeyCode::ShiftLeft | KeyCode::ShiftRight => self.keys.fast = pressed,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if egui_wants {
                    return;
                }
                self.scroll += match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
                };
            }
            _ => {}
        }
    }

    /// Feed a device event. Only raw mouse motion is used, and only while looking.
    pub fn process_device_event(&mut self, event: &DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if self.looking {
                self.look_delta.0 += *dx as f32;
                self.look_delta.1 += *dy as f32;
            }
        }
    }

    /// Apply this frame's input to `camera`. Does nothing when idle, so the
    /// editor's manual camera edits are left untouched.
    pub fn update(&mut self, camera: &mut Camera, dt: f32) {
        let active = self.looking || self.keys.moving() || self.scroll != 0.0;
        if !active {
            self.look_delta = (0.0, 0.0);
            self.scroll = 0.0;
            return;
        }

        // Re-derive from the camera each frame so edits made elsewhere (e.g. the
        // environment panel) are respected, then fold in this frame's look delta.
        self.sync_from(camera);
        if self.looking {
            self.yaw += self.look_delta.0 * self.sensitivity;
            self.pitch =
                (self.pitch - self.look_delta.1 * self.sensitivity).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        }

        let forward = forward_dir(self.yaw, self.pitch);
        let right = forward.cross(Vec3::Y).normalize_or_zero();

        let speed = self.move_speed * if self.keys.fast { self.boost } else { 1.0 } * dt;
        let mut pos = camera.position;
        if self.keys.forward {
            pos += forward * speed;
        }
        if self.keys.back {
            pos -= forward * speed;
        }
        if self.keys.right {
            pos += right * speed;
        }
        if self.keys.left {
            pos -= right * speed;
        }
        if self.keys.up {
            pos += Vec3::Y * speed;
        }
        if self.keys.down {
            pos -= Vec3::Y * speed;
        }
        pos += forward * self.scroll * self.scroll_speed;

        camera.position = pos;
        camera.target = pos + forward;

        self.look_delta = (0.0, 0.0);
        self.scroll = 0.0;
    }
}

/// Unit forward vector for a yaw/pitch pair (yaw 0 faces `-Z`).
fn forward_dir(yaw: f32, pitch: f32) -> Vec3 {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    Vec3::new(sy * cp, sp, -cy * cp)
}
