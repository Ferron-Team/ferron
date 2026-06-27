//! ECS component types attached to scene entities.
//!
//! Each component is a plain data struct stored in a `ferron_ecs::World`.
//! Systems query these to drive simulation and rendering.

mod light;
mod local_transform;
mod material_handle;
mod mesh_handle;
mod name;
mod spin;

pub use light::{AmbientLight, Light};
pub use local_transform::LocalTransform;
pub use material_handle::MaterialHandle;
pub use mesh_handle::MeshHandle;
pub use name::Name;
pub use spin::Spin;
