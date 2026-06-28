//! In-window editor UI (egui overlay).
//!
//! Rendering is decoupled from the renderer: [`Editor::run`] builds the UI
//! against the [`World`], and [`Editor::draw`] composites it onto the final
//! swapchain image via the renderer's overlay hook
//! (`VulkanRenderer::render_with_overlay`). New tools are added as `panels`
//! modules.

mod panels;
mod state;
mod theme;

use std::sync::Arc;

use egui_winit_vulkano::{Gui, GuiConfig};
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::swapchain::Surface;
use vulkano::sync::GpuFuture;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;

use ferron_ecs::World;

use self::state::EditorState;

pub struct Editor {
    gui: Gui,
    state: EditorState,
}

impl Editor {
    /// `format` is the swapchain color format the UI composites onto
    /// (`VulkanRenderer::color_format`).
    pub fn new(
        event_loop: &ActiveEventLoop,
        surface: Arc<Surface>,
        queue: Arc<Queue>,
        format: Format,
    ) -> Self {
        let gui = Gui::new(
            event_loop,
            surface,
            queue,
            format,
            GuiConfig {
                // Load (don't clear) so the UI draws over the rendered scene; the
                // swapchain target is sRGB.
                is_overlay: true,
                allow_srgb_render_target: true,
                ..Default::default()
            },
        );
        theme::apply(&gui.context());
        Self {
            gui,
            state: EditorState::default(),
        }
    }

    /// Returns `true` if egui wants the event (e.g. the pointer is over a panel),
    /// so the caller can withhold it from game/camera input.
    pub fn on_window_event(&mut self, event: &WindowEvent) -> bool {
        self.gui.update(event)
    }

    /// Build this frame's UI, then apply any spawn/despawn it requested. Call once
    /// per frame before [`draw`](Self::draw).
    pub fn run(&mut self, world: &mut World) {
        // Destructure for disjoint borrows: `gui` drives egui while the closure
        // edits `state`/`world`.
        let Editor { gui, state } = self;
        gui.immediate_ui(|gui| {
            let ctx = gui.context();
            panels::draw(&ctx, world, state);
        });
        state.apply(world);
    }

    /// Renderer overlay hook: composite the built UI onto `image`, waiting on
    /// `before` (the rendered scene), and return the future to present.
    pub fn draw(
        &mut self,
        before: Box<dyn GpuFuture>,
        image: Arc<ImageView>,
    ) -> Box<dyn GpuFuture> {
        self.gui.draw_on_image(before, image)
    }
}
