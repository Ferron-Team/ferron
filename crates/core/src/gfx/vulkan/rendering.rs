//! What each pass renders into, without render pass objects or framebuffers.
//!
//! Every graphics pass in `gfx/vulkan/` opens a rendering instance with
//! [`VK_KHR_dynamic_rendering`], promoted to core in Vulkan 1.3 and required by
//! `context.rs`. That replaces three things this module now owns between them:
//! the `RenderPass` each pass used to declare, the `Framebuffer` binding its
//! attachments, and the positional `Vec<Option<ClearValue>>` that had to line up
//! with the framebuffer's attachment order.
//!
//! Two consequences are worth naming, because they are why the change is worth
//! making beyond the deletion:
//!
//! - **Load and store ops become a record-time decision.** They were baked into
//!   the render pass, which is why the shadow atlas needed a second render pass
//!   differing from the cascades' only in `load_op`, and why `PassFramebuffers`
//!   kept four forward render passes to spell out four attachment shapes.
//! - **Attachment layouts are stated per attachment, at the point of use.** The
//!   read-only prepass depth that `oit.rs`, `refraction.rs` and the one-sample
//!   forward pass all borrow is simply an attachment in
//!   [`ImageLayout::DepthStencilReadOnlyOptimal`]. That is exactly what
//!   `single_pass_renderpass!` could not express and what those three modules
//!   hand-built whole `RenderPassCreateInfo`s to say.
//!
//! [`VK_KHR_dynamic_rendering`]: https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_dynamic_rendering.html

use std::sync::Arc;

use glam::Vec3;
use vulkano::command_buffer::{
    RenderingAttachmentInfo, RenderingAttachmentResolveInfo, RenderingInfo,
};
use vulkano::format::{ClearValue, Format};
use vulkano::image::view::ImageView;
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};

use super::frame::{FrameIds, PassBody};
use super::resources::GraphImages;

/// The formats a pipeline will be used with.
///
/// This is what a `GraphicsPipelineCreateInfo::subpass` carries now, and it is
/// the whole of what a pipeline needs to know about its target: the render pass
/// object it used to point at said nothing else a pipeline read. Each pass
/// module builds its own from the format constants it already declares, so a
/// format still has exactly one definition — it is just no longer restated in a
/// render pass beside the image that carries it.
pub(super) fn pipeline_info(
    color: &[Format],
    depth: Option<Format>,
) -> PipelineRenderingCreateInfo {
    PipelineRenderingCreateInfo {
        color_attachment_formats: color.iter().copied().map(Some).collect(),
        depth_attachment_format: depth,
        ..Default::default()
    }
}

/// A colour attachment that is cleared, drawn into, and kept.
fn cleared(view: Arc<ImageView>, clear: ClearValue) -> RenderingAttachmentInfo {
    RenderingAttachmentInfo {
        load_op: AttachmentLoadOp::Clear,
        store_op: AttachmentStoreOp::Store,
        clear_value: Some(clear),
        ..RenderingAttachmentInfo::image_view(view)
    }
}

/// A colour attachment whose previous contents do not matter and which is
/// entirely rewritten — the swapchain image, and the shadow atlas whose live
/// tiles `record_atlas` clears itself.
fn overwritten(view: Arc<ImageView>) -> RenderingAttachmentInfo {
    RenderingAttachmentInfo {
        load_op: AttachmentLoadOp::DontCare,
        store_op: AttachmentStoreOp::Store,
        ..RenderingAttachmentInfo::image_view(view)
    }
}

/// A multisampled colour attachment that is cleared, drawn into, resolved into
/// `resolve`, and then discarded.
///
/// `DontCare` on the multisampled image is the point of it: nothing reads the
/// samples after the resolve, so on a tiler they never reach memory at all, and
/// the image carries `TRANSIENT_ATTACHMENT` for that reason.
fn resolved(
    view: Arc<ImageView>,
    resolve: Arc<ImageView>,
    clear: ClearValue,
) -> RenderingAttachmentInfo {
    RenderingAttachmentInfo {
        load_op: AttachmentLoadOp::Clear,
        store_op: AttachmentStoreOp::DontCare,
        clear_value: Some(clear),
        resolve_info: Some(RenderingAttachmentResolveInfo::image_view(resolve)),
        ..RenderingAttachmentInfo::image_view(view)
    }
}

/// A depth attachment this pass rasterises for itself: cleared to far, tested
/// and written, and discarded at the end because nothing samples it.
fn depth_scratch(view: Arc<ImageView>) -> RenderingAttachmentInfo {
    RenderingAttachmentInfo {
        load_op: AttachmentLoadOp::Clear,
        store_op: AttachmentStoreOp::DontCare,
        clear_value: Some(1.0.into()),
        ..RenderingAttachmentInfo::image_view(view)
    }
}

/// A depth attachment that is cleared, written, and kept for later passes to
/// sample — the geometry prepass's, and each shadow cascade's.
fn depth_written(view: Arc<ImageView>) -> RenderingAttachmentInfo {
    RenderingAttachmentInfo {
        load_op: AttachmentLoadOp::Clear,
        store_op: AttachmentStoreOp::Store,
        clear_value: Some(1.0.into()),
        ..RenderingAttachmentInfo::image_view(view)
    }
}

/// The geometry prepass's depth, borrowed by a later pass that tests against it
/// and never writes it.
///
/// `DepthStencilReadOnlyOptimal` is the layout the barrier plan says this image
/// is in — every pass from the prepass onwards samples it — and stating it here
/// is what keeps this attachment from transitioning an image nobody wrote. Under
/// render pass objects that took a hand-built `RenderPassCreateInfo` in each of
/// the three modules that borrow it; here it is one field.
///
/// Stored rather than `DontCare` for the same reason it was before: nothing was
/// written, so there is nothing to discard, and `DontCare` would license a
/// driver to leave the depth undefined for the passes that read it after this
/// one — which is most of them.
fn depth_borrowed(view: Arc<ImageView>) -> RenderingAttachmentInfo {
    RenderingAttachmentInfo {
        image_layout: vulkano::image::ImageLayout::DepthStencilReadOnlyOptimal,
        load_op: AttachmentLoadOp::Load,
        store_op: AttachmentStoreOp::Store,
        ..RenderingAttachmentInfo::image_view(view)
    }
}

/// What one graphics pass renders into, or `None` for a pass that dispatches
/// compute or is recorded outside the executor entirely.
///
/// This is the whole of what `PassFramebuffers::build` and `clear_values` used
/// to do between them, and it is a fraction of the size because the two halves
/// are now one: an attachment carries its own clear, so there is no positional
/// list to keep in step with a framebuffer's attachment order, and no need to
/// tell four forward shapes apart by counting attachments and reading a sample
/// count back off the first one.
pub(super) fn rendering_info(
    ids: &FrameIds,
    images: &GraphImages,
    swapchain: &Arc<ImageView>,
    body: PassBody,
    background: Vec3,
) -> Option<RenderingInfo> {
    let color_only = |attachment: RenderingAttachmentInfo| RenderingInfo {
        color_attachments: vec![Some(attachment)],
        ..Default::default()
    };

    Some(match body {
        // Flat +Z in the normal buffer, no motion in the velocity buffer, far in
        // depth. Zero velocity is what the sky and any unrasterised pixel are
        // left with, and the resolve reads that as "reproject with the camera
        // alone" rather than as a stationary surface.
        //
        // A black `f0` at zero roughness in the material target, so a pixel
        // nothing rasterised reflects nothing rather than mirroring the sky at
        // whatever the previous contents implied.
        PassBody::GeometryPrepass => {
            let prepass = ids.prepass.expect("prepass without its resources");
            RenderingInfo {
                color_attachments: vec![
                    Some(cleared(
                        images.view(prepass.normal),
                        [0.5, 0.5, 1.0, 0.0].into(),
                    )),
                    Some(cleared(
                        images.view(prepass.velocity),
                        [0.0, 0.0, 0.0, 0.0].into(),
                    )),
                    Some(cleared(
                        images.view(prepass.material),
                        [0.0, 0.0, 0.0, 0.0].into(),
                    )),
                ],
                depth_attachment: Some(depth_written(images.view(prepass.depth))),
                ..Default::default()
            }
        }
        // 1.0 = fully unoccluded, so an untouched pixel darkens nothing.
        PassBody::SsaoResolve => {
            let ssao = ids.ssao.expect("SSAO pass without SSAO resources");
            color_only(cleared(
                images.view(ssao.raw_ao),
                [1.0, 0.0, 0.0, 0.0].into(),
            ))
        }
        PassBody::SsaoBlur => {
            let ssao = ids.ssao.expect("SSAO pass without SSAO resources");
            color_only(cleared(images.view(ssao.ao), [1.0, 0.0, 0.0, 0.0].into()))
        }
        // 1.0 = fully lit, for the same reason: the pass writes every pixel, so
        // this only decides what a pixel would be if it somehow did not.
        PassBody::ContactShadows => color_only(cleared(
            images.view(
                ids.contact_shadows
                    .expect("contact shadow pass without its mask"),
            ),
            [1.0, 0.0, 0.0, 0.0].into(),
        )),
        // The four shapes that used to be four render passes and four
        // framebuffers, now two independent questions asked in the order they
        // are decided: does this frame resolve, and does it diffuse.
        //
        // The diffusible target is cleared to zero, and that clear is the
        // feature's mask: the skybox and the debug lines render alongside this
        // pass and write nothing to that attachment, so zero is what a pixel
        // they covered holds — and a zero radius is exactly "nothing scattered
        // here".
        PassBody::Forward => {
            let color = ClearValue::from([background.x, background.y, background.z, 1.0]);
            let diffusible = ClearValue::from([0.0, 0.0, 0.0, 0.0]);
            let sss = ids.subsurface;
            match ids.msaa {
                Some(msaa) => {
                    let mut color_attachments = vec![Some(resolved(
                        images.view(msaa.hdr),
                        images.view(ids.hdr_color),
                        color,
                    ))];
                    if let Some(sss) = sss {
                        color_attachments.push(Some(resolved(
                            images.view(
                                sss.msaa_diffusible
                                    .expect("a multisampled frame resolves its diffusible"),
                            ),
                            images.view(sss.diffusible),
                            diffusible,
                        )));
                    }
                    RenderingInfo {
                        color_attachments,
                        depth_attachment: Some(depth_scratch(images.view(msaa.depth))),
                        ..Default::default()
                    }
                }
                // One sample: no resolve, so `hdr_color` is the colour
                // attachment itself and is stored; and the depth is the
                // prepass's, borrowed for the `EQUAL` test.
                None => {
                    let mut color_attachments =
                        vec![Some(cleared(images.view(ids.hdr_color), color))];
                    if let Some(sss) = sss {
                        color_attachments
                            .push(Some(cleared(images.view(sss.diffusible), diffusible)));
                    }
                    RenderingInfo {
                        color_attachments,
                        depth_attachment: Some(depth_borrowed(
                            images.view(
                                ids.prepass
                                    .expect("a one-sample forward pass borrows the prepass depth")
                                    .depth,
                            ),
                        )),
                        ..Default::default()
                    }
                }
            }
        }
        // Nothing accumulated, and everything revealed. The second is the one
        // that matters: revealage is a running *product* of `1 - alpha`, so a
        // clear of zero would hide the opaque frame everywhere rather than
        // nowhere.
        PassBody::OitAccumulate => {
            let oit = ids
                .transparency
                .expect("transparency pass without its targets");
            let prepass = ids
                .prepass
                .expect("the graph scheduled transparency with no prepass");
            RenderingInfo {
                color_attachments: vec![
                    Some(cleared(images.view(oit.accum), [0.0, 0.0, 0.0, 0.0].into())),
                    Some(cleared(
                        images.view(oit.reveal),
                        [1.0, 0.0, 0.0, 0.0].into(),
                    )),
                ],
                depth_attachment: Some(depth_borrowed(images.view(prepass.depth))),
                ..Default::default()
            }
        }
        // Nothing covered, so nothing replaced: the composite reads this as the
        // frame's own colour showing through in full. Zero rather than one is
        // what makes it an `over` of nothing instead of a black wall.
        PassBody::RefractionDraw => {
            let refraction = ids.refraction.expect("refraction pass without its targets");
            let prepass = ids
                .prepass
                .expect("the graph scheduled refraction with no prepass");
            RenderingInfo {
                color_attachments: vec![Some(cleared(
                    images.view(refraction.accum),
                    [0.0, 0.0, 0.0, 0.0].into(),
                ))],
                depth_attachment: Some(depth_borrowed(images.view(prepass.depth))),
                ..Default::default()
            }
        }
        // One layer of the cascade array, not the array view: an attachment is a
        // single layer, and handing it the whole array would make every cascade
        // clear and overwrite the same texels without erroring.
        //
        // Entirely re-rendered, so clearing it whole is exactly what it wants: a
        // texel nobody drew into reads as far, and therefore as lit.
        PassBody::ShadowCascade(cascade) => RenderingInfo {
            depth_attachment: Some(depth_written(images.layer_view(
                ids.shadows.expect("shadow pass without a shadow image"),
                cascade,
            ))),
            ..Default::default()
        },
        // The atlas is not re-rendered whole. It holds up to 64 tiles of which a
        // frame lights a fraction, so it loads `DontCare` and `record_atlas`
        // clears the tiles actually assigned.
        //
        // Under render pass objects that difference — one `load_op` — cost a
        // second render pass object. Here it is this arm.
        PassBody::PunctualShadows => RenderingInfo {
            depth_attachment: Some(RenderingAttachmentInfo {
                load_op: AttachmentLoadOp::DontCare,
                store_op: AttachmentStoreOp::Store,
                ..RenderingAttachmentInfo::image_view(
                    images.view(
                        ids.shadow_atlas
                            .expect("punctual shadow pass without an atlas"),
                    ),
                )
            }),
            ..Default::default()
        },
        // The acquired swapchain image, every pixel of which the tonemap writes.
        PassBody::Tonemap => color_only(overwritten(swapchain.clone())),
        // Dispatches, and the overlay, which records its own command buffer.
        PassBody::Overlay
        | PassBody::SsrHiz
        | PassBody::SsrSource
        | PassBody::SsrTrace
        | PassBody::SsrResolve
        | PassBody::FogScatter
        | PassBody::FogIntegrate
        | PassBody::SubsurfaceBlurHorizontal
        | PassBody::SubsurfaceBlurVertical
        | PassBody::SubsurfaceComposite
        | PassBody::OitComposite
        | PassBody::RefractionScene
        | PassBody::RefractionComposite
        | PassBody::TaaResolve
        | PassBody::DofPrefilter
        | PassBody::DofTileMax
        | PassBody::DofGather
        | PassBody::DofComposite
        | PassBody::MotionBlurTileMax
        | PassBody::MotionBlurNeighbourMax
        | PassBody::MotionBlurGather
        | PassBody::LuminanceHistogram
        | PassBody::LuminanceAverage
        | PassBody::BloomPrefilter
        | PassBody::BloomDownsample(_)
        | PassBody::BloomUpsample(_) => return None,
    })
}
