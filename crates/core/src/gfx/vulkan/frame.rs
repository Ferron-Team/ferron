//! The engine's frame, declared as a graph.
//!
//! This module is the one place that says what a frame *is*. It takes no
//! `Device` and allocates nothing: it registers resources and passes and hands
//! back the compiled result, so the same function that runs on the GPU each
//! frame is the one CI compiles and asserts the barrier plan of.
//!
//! Adding a pass is a registration here plus a [`PassBody`] arm in `execute`.
//! Nothing computes an execution order, a layout, or a barrier by hand, which is
//! what makes the shadow cascades of Part 1 a loop over `declare` rather than
//! surgery on a fixed pipeline.

use vulkano::format::Format;
use vulkano::image::ImageLayout;

use crate::gfx::graph::{
    Access, Extent, FrameGraph, GraphBuilder, GraphError, ImageDesc, PassId, PassKind, ResourceId,
    compile,
};
use crate::gfx::shadows::MAX_CASCADES;

use super::MSAA_SAMPLES;
use super::bloom::MAX_BLOOM_MIPS;
use super::contact_shadows::MASK_FORMAT;
use super::dof::COC_TILE_SHIFT;
use super::hdr::HDR_FORMAT;
use super::motion_blur::TILE_SHIFT;
use super::oit::{ACCUM_FORMAT, REVEAL_FORMAT};
use super::prepass::{MATERIAL_FORMAT, NORMAL_FORMAT, VELOCITY_FORMAT};
use super::refraction::{ACCUM_FORMAT as REFRACTION_ACCUM_FORMAT, SCENE_LEVELS};
use super::ssao::AO_FORMAT;
use super::ssr::{HIZ_FORMAT, HIZ_LEVELS, RAY_FORMAT, SOURCE_LEVELS};
use super::swapchain::DEPTH_FORMAT;

/// What a frame's structure depends on. A change to any of these recompiles the
/// graph; nothing else does, which is the "recompiled on structure change, not
/// per frame" rule made concrete — the field list *is* the rule.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FrameConfig {
    pub color_format: Format,
    pub ssao: bool,
    /// Whether the frame marches the depth buffer for the shadow band the
    /// cascades cannot resolve. Structural like the rest, and one more consumer
    /// that keeps the geometry prepass alive on its own — it reads the depth and
    /// the normals that pass writes, and its result is what the forward pass
    /// multiplies the sun's term by.
    pub contact_shadows: bool,
    /// Whether the frame reflects off itself. Structural like the rest, and
    /// another consumer that can keep the geometry prepass alive on its own —
    /// it reads depth, normals and the material target the prepass writes.
    pub ssr: bool,
    /// Whether the frame resolves against a reprojected history. Structural
    /// twice over: it registers the resolve node, and it is what makes the
    /// geometry prepass exist in a frame that has SSAO switched off — the
    /// prepass is where the motion vectors come from.
    pub taa: bool,
    /// Whether the frame meters its own luminance. Off is a different graph, not
    /// a flag read at record time: the two compute passes are never registered
    /// and the tonemap pass falls back to the manual exposure it is pushed.
    pub auto_exposure: bool,
    /// Whether the frame reconstructs the shutter's exposure from the motion
    /// vectors. Structural for the same reason TAA is, and with the same
    /// consequence: it is one of the things that makes the geometry prepass
    /// exist in a frame with SSAO switched off.
    pub motion_blur: bool,
    /// Whether the frame defocuses what the lens is not focused on. Structural,
    /// and the other consumer that can keep the prepass alive on its own —
    /// depth is all it needs from it.
    pub dof: bool,
    /// Whether the frame draws blended geometry. Structural like the rest: it
    /// registers the accumulation and composite nodes, and it is one more thing
    /// that keeps the geometry prepass alive — the accumulation depth-tests
    /// against the depth that pass writes, because the forward pass's own is a
    /// memoryless attachment that does not survive its render pass.
    pub transparency: bool,
    /// Whether the frame draws refractive geometry. Structural like the rest,
    /// and it keeps the geometry prepass alive for the reason transparency
    /// does — the draw depth-tests against the depth that pass writes.
    ///
    /// Independent of `transparency` rather than folded into it: the two queues
    /// answer to opposite rules about ordering, so a frame may reasonably want
    /// one without the other, and each is its own A/B for "is this a
    /// transparency bug or a refraction bug?".
    pub refraction: bool,
    /// Levels in the bloom chain, zero for none. Derived from the frame's extent
    /// rather than set, so the number of passes registered cannot disagree with
    /// the number of levels there is room for — the same reason
    /// `shadow_cascades` is sourced from the cascade set.
    pub bloom_mips: u8,
    /// Whether the editor's egui overlay draws over the frame. Off for headless
    /// and export renders.
    pub overlay: bool,
    pub shadow_cascades: u8,
    pub shadow_resolution: u32,
    /// Edge of the punctual shadow atlas in texels, zero for a frame where
    /// nothing punctual casts. Sourced from the fitted atlas rather than from
    /// the setting, for the reason `shadow_cascades` is: a frame whose lights
    /// all opted out declares no atlas rather than clearing one nobody reads.
    ///
    /// One number and one pass however many lights cast — the tile count varies
    /// frame to frame *inside* the pass, which is exactly why it is not here.
    pub shadow_atlas: u32,
}

/// Which piece of engine code a graph node runs.
///
/// The graph knows a pass by its declarations; this is the other half of the
/// mapping, kept as data so it stays device-free and so the executor's dispatch
/// is exhaustive by the compiler's own reckoning rather than by convention.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PassBody {
    GeometryPrepass,
    SsaoResolve,
    SsaoBlur,
    /// The sun's visibility over the short range a cascade texel cannot resolve.
    ContactShadows,
    Forward,
    /// Every level of the reflection trace's min-depth pyramid, in one dispatch.
    SsrHiz,
    /// Every level of the lit frame's pyramid, likewise — what a ray's cone is
    /// sampled out of.
    SsrSource,
    /// One importance-sampled reflection ray per half-resolution pixel.
    SsrTrace,
    /// Upsamples those rays and swaps the environment's reflection for them.
    SsrResolve,
    /// Every blended surface, into the two targets whose blend equations
    /// commute — which is what makes the draw order irrelevant.
    OitAccumulate,
    /// Divides that accumulation by its own coverage and mixes it over the lit
    /// frame by the transmittance beside it.
    OitComposite,
    /// The lit frame as a mip pyramid, so a rough refraction reads a cone.
    RefractionScene,
    /// Every refractive surface, back to front, into a premultiplied target.
    RefractionDraw,
    /// Puts that target over the lit frame.
    RefractionComposite,
    /// Reprojects the history onto this frame and accumulates into it.
    TaaResolve,
    /// Half the frame, carrying its own circle of confusion.
    DofPrefilter,
    /// The widest near and far circle per tile, which sizes the gather's kernel.
    DofTileMax,
    /// Spreads each half-resolution texel over its circle, into a near and a far
    /// field.
    DofGather,
    /// Puts both fields back over the sharp frame at full resolution.
    DofComposite,
    /// The longest blur vector in each tile.
    MotionBlurTileMax,
    /// Dilates that by one ring, so every pixel a blur can reach knows about it.
    MotionBlurNeighbourMax,
    /// Reconstructs the exposure by walking the dominant blur vector.
    MotionBlurGather,
    LuminanceHistogram,
    LuminanceAverage,
    /// Half the frame, exposed and firefly-weighted: the chain's first level.
    BloomPrefilter,
    /// Writes down-chain level `n`, reading level `n - 1`.
    BloomDownsample(u32),
    /// Writes up-chain level `n`, reading the coarser level and down-chain `n`.
    BloomUpsample(u32),
    Tonemap,
    Overlay,
    ShadowCascade(u32),
    /// Every point and spot light's faces, in one pass over one atlas. A tile is
    /// a viewport rather than a node, which is what keeps this a single barrier
    /// instead of one per face.
    PunctualShadows,
}

/// Handles the executor needs to bind per-frame resources and to find the
/// graph-owned views a pass draws into.
#[derive(Clone, Copy, Debug)]
pub struct FrameIds {
    pub object_transforms: ResourceId,
    pub swapchain_color: ResourceId,
    pub hdr_color: ResourceId,
    /// What the tonemap, metering and bloom passes read: whichever image the
    /// optical chain left the frame's colour in — `hdr_color` in the barest
    /// frame, and otherwise the last of the TAA resolve, the lens and the
    /// shutter that ran. Named once here so nothing downstream has to ask which
    /// frame it is in, which is what makes inserting a stage a change to one
    /// binding rather than to every consumer.
    pub scene_color: ResourceId,
    pub msaa_hdr: ResourceId,
    pub msaa_depth: ResourceId,
    /// Written by the metering passes and read by the tonemap pass. Always
    /// declared, because the tonemap pass binds it whether or not anything wrote
    /// it this frame — an import may be read without a writer, which is exactly
    /// the case metering-off produces.
    pub exposure: ResourceId,
    /// `None` when metering is off, along with the two passes that touch it.
    pub histogram: Option<ResourceId>,
    pub bloom: Option<BloomIds>,
    /// Present whenever anything downstream needs depth, normals or motion —
    /// SSAO, TAA, motion blur, depth of field, or any combination of them.
    pub prepass: Option<PrepassIds>,
    pub ssao: Option<SsaoIds>,
    /// The sun-visibility mask, when the frame marches for one. One image and no
    /// struct: the pass reads the prepass and writes this, and there is nothing
    /// else to name.
    pub contact_shadows: Option<ResourceId>,
    pub ssr: Option<SsrIds>,
    pub transparency: Option<TransparencyIds>,
    pub refraction: Option<RefractionIds>,
    pub taa: Option<TaaIds>,
    pub dof: Option<DofIds>,
    pub motion_blur: Option<MotionBlurIds>,
    pub shadows: Option<ResourceId>,
    /// The punctual atlas, when anything punctual casts. One image and no
    /// struct, like the contact-shadow mask: one pass writes it and the forward
    /// pass reads it.
    pub shadow_atlas: Option<ResourceId>,
}

/// The two chains bloom needs, each level a separate image.
///
/// Fixed-length arrays rather than `Vec` so `FrameIds` stays `Copy`; only the
/// first `mips` entries of `down` and the first `mips - 1` of `up` are ever
/// populated, and the rest stay `None` so a length mistake is a panic naming the
/// level rather than a read of some other pass's image.
#[derive(Clone, Copy, Debug)]
pub struct BloomIds {
    pub mips: u8,
    pub down: [Option<ResourceId>; MAX_BLOOM_MIPS],
    pub up: [Option<ResourceId>; MAX_BLOOM_MIPS],
}

impl BloomIds {
    pub fn down(&self, level: usize) -> ResourceId {
        self.down[level].expect("bloom down-chain level was never declared")
    }

    pub fn up(&self, level: usize) -> ResourceId {
        self.up[level].expect("bloom up-chain level was never declared")
    }

    /// What the tonemap pass composites: the finest level of the up chain, or —
    /// for a one-level chain, which has no upsample step — the down chain's only
    /// level.
    pub fn result(&self) -> ResourceId {
        if self.mips >= 2 {
            self.up(0)
        } else {
            self.down(0)
        }
    }
}

/// What the one geometry pass in front of shading leaves behind. Written
/// together because they come off the same rasterisation, and read apart: SSAO
/// wants depth and normals, TAA and motion blur want depth and motion, depth of
/// field wants depth alone, reflections want all but the motion.
#[derive(Clone, Copy, Debug)]
pub struct PrepassIds {
    pub normal: ResourceId,
    pub velocity: ResourceId,
    /// `rgb` = the surface's `f0`, `a` = perceptual roughness. What a reflection
    /// is tinted by and how wide its lobe is — the two things a screen-space
    /// trace cannot recover from depth and normals alone.
    pub material: ResourceId,
    pub depth: ResourceId,
}

#[derive(Clone, Copy, Debug)]
pub struct SsaoIds {
    pub raw_ao: ResourceId,
    pub ao: ResourceId,
}

/// The reflection trace's three images, and what it was handed.
#[derive(Clone, Copy, Debug)]
pub struct SsrIds {
    /// The lit frame the rays sample, recorded rather than re-derived for the
    /// reason depth of field records its own: this pass sits in a chain, and
    /// the executor must bind exactly what `declare` said it would read.
    pub source: ResourceId,
    /// The min-depth pyramid, one image with every level — see
    /// [`ImageDesc::mip_levels`](crate::gfx::graph::ImageDesc::mip_levels) for
    /// why a single pass has to write all of them.
    pub hiz: ResourceId,
    /// Half the frame with a pyramid over it: the lit colour, prefiltered, so a
    /// ray from a rough surface reads the average of what its cone covers
    /// rather than one texel inside it.
    pub source_pyramid: ResourceId,
    /// Half the frame: the radiance each ray found, with its confidence in
    /// alpha.
    pub rays: ResourceId,
    pub output: ResourceId,
}

/// Weighted-blended transparency's two targets, and what it was handed.
///
/// `source` is recorded rather than re-derived for the reason depth of field
/// records its own: this pass sits in a chain — it composites over the
/// reflections' output in a frame that has them and the forward pass's target in
/// a frame that does not — and the executor must bind exactly what `declare`
/// said it would read.
#[derive(Clone, Copy, Debug)]
pub struct TransparencyIds {
    pub source: ResourceId,
    /// `rgb` = weighted premultiplied radiance summed, `a` = weighted coverage
    /// summed.
    pub accum: ResourceId,
    /// The product of `1 - alpha` over everything that covered the pixel. One
    /// channel, and cleared to 1.0 rather than 0 because it is a product.
    pub reveal: ResourceId,
    pub output: ResourceId,
}

/// Screen-space refraction's three images, and what it was handed.
///
/// `source` is recorded rather than re-derived for the reason the transparency
/// ids record theirs: this pass sits in a chain, and the executor must bind
/// exactly what `declare` said it would read.
#[derive(Clone, Copy, Debug)]
pub struct RefractionIds {
    pub source: ResourceId,
    /// The lit frame reduced to a mip chain, at full resolution: a clear pane
    /// of glass shows the world behind it as sharply as the frame recorded it,
    /// and only a rough one reads a coarser level.
    pub scene: ResourceId,
    /// `rgb` = premultiplied radiance, `a` = coverage. Cleared to zero, so a
    /// pixel nothing refractive covered leaves the frame as it found it.
    pub accum: ResourceId,
    pub output: ResourceId,
}

/// The two images TAA carries across the frame boundary.
///
/// Both are **imported**, for the reason the exposure buffer is: a transient is
/// `Undefined` at every frame's start by contract, and a history that survives
/// one frame is precisely what this pass needs. The backing allocations are
/// ping-ponged, so the image bound as `output` this frame is the one bound as
/// `history` next — which is why `output` leaves the frame in the layout
/// `history` declares it enters in.
#[derive(Clone, Copy, Debug)]
pub struct TaaIds {
    /// What this frame's colour is at the point the resolve accumulates it —
    /// the forward pass's target, or the reflection composite's where that ran.
    /// Recorded rather than re-derived for the reason depth of field records its
    /// own source: the executor must bind exactly what `declare` said it reads.
    pub source: ResourceId,
    pub history: ResourceId,
    pub output: ResourceId,
}

/// Depth of field's three images, and what it was handed.
///
/// `source` is recorded rather than re-derived because this pass sits in a
/// chain: it reads the TAA resolve's output in a frame that has one and the
/// forward pass's in a frame that does not, and the executor must bind exactly
/// what `declare` said it would read. The same reason the bloom upsample's
/// coarse level is looked up the same way in both places.
#[derive(Clone, Copy, Debug)]
pub struct DofIds {
    pub source: ResourceId,
    /// Half the frame: colour, with the signed circle of confusion in alpha.
    pub prefiltered: ResourceId,
    /// The widest near and far circle per tile. What the gather sizes its kernel
    /// from, so a fixed tap budget lands inside the blur rather than around it.
    pub tile: ResourceId,
    /// The two fields, kept apart because they composite differently — a near
    /// one spills over what it occludes and a far one does not.
    pub near: ResourceId,
    pub far: ResourceId,
    pub output: ResourceId,
}

/// Motion blur's velocity pyramid, and what it was handed.
#[derive(Clone, Copy, Debug)]
pub struct MotionBlurIds {
    pub source: ResourceId,
    /// The longest blur vector per tile, and that dilated by one ring.
    pub tile: ResourceId,
    pub neighbour: ResourceId,
    pub output: ResourceId,
}

pub struct Frame {
    pub graph: FrameGraph,
    pub ids: FrameIds,
    /// Indexed by [`PassId`], so a pass's declarations and its body cannot drift.
    pub bodies: Vec<PassBody>,
}

pub fn declare(config: FrameConfig) -> Result<Frame, GraphError> {
    let mut builder = GraphBuilder::new();
    let mut bodies = Vec::new();
    let record = |id: PassId, body: PassBody, bodies: &mut Vec<PassBody>| {
        debug_assert_eq!(id.index(), bodies.len());
        bodies.push(body);
    };

    // Host-written each frame and read by both geometry passes; the per-object
    // inverse-transpose is too expensive to compute twice, so the two passes
    // share one upload and the graph records that they do.
    let object_transforms = builder.import_buffer("object_transforms");

    let swapchain_color = builder.import_image(
        "swapchain_color",
        ImageDesc::new(config.color_format),
        // An acquired image's contents are not ours to keep, and the
        // presentation engine wants it back in `PresentSrc`.
        ImageLayout::Undefined,
        ImageLayout::PresentSrc,
    );

    let shadows = (config.shadow_cascades > 0).then(|| {
        builder.create_image(
            "shadow_cascades",
            ImageDesc::new(DEPTH_FORMAT)
                .extent(Extent::Fixed([config.shadow_resolution; 2]))
                .array_layers(config.shadow_cascades as u32),
        )
    });

    // Resource and pass names live in separate namespaces, so a level's image
    // and the pass that writes it can share a name — and it reads well in the
    // plan, where `bloom_down_2` writes `bloom_down_2`.
    const BLOOM_DOWN_NAMES: [&str; MAX_BLOOM_MIPS] = [
        "bloom_down_0",
        "bloom_down_1",
        "bloom_down_2",
        "bloom_down_3",
        "bloom_down_4",
        "bloom_down_5",
    ];
    const BLOOM_UP_NAMES: [&str; MAX_BLOOM_MIPS] = [
        "bloom_up_0",
        "bloom_up_1",
        "bloom_up_2",
        "bloom_up_3",
        "bloom_up_4",
        "bloom_up_5",
    ];
    // Level 0's downsample is the prefilter, which has its own name; the slot is
    // present so the two tables index alike.
    const BLOOM_DOWN_PASS_NAMES: [&str; MAX_BLOOM_MIPS] = BLOOM_DOWN_NAMES;
    const BLOOM_UP_PASS_NAMES: [&str; MAX_BLOOM_MIPS] = BLOOM_UP_NAMES;

    const CASCADE_PASS_NAMES: [&str; MAX_CASCADES] = [
        "shadow_cascade_0",
        "shadow_cascade_1",
        "shadow_cascade_2",
        "shadow_cascade_3",
    ];

    if let Some(shadows) = shadows {
        for cascade in 0..config.shadow_cascades as u32 {
            let id = builder
                .pass(CASCADE_PASS_NAMES[cascade as usize], PassKind::Inline)
                .access(object_transforms, Access::StorageRead)
                .access(shadows, Access::DepthAttachment)
                .build();
            record(id, PassBody::ShadowCascade(cascade), &mut bodies);
        }
    }

    // Beside the cascades rather than after the prepass, because it is the same
    // kind of thing: geometry rasterised from a light's point of view, needed
    // before anything shades. One node for every face of every light — a tile is
    // a viewport into this attachment, so six faces of sixteen lights would
    // still be one barrier.
    let shadow_atlas = (config.shadow_atlas > 0).then(|| {
        let atlas = builder.create_image(
            "shadow_atlas",
            ImageDesc::new(DEPTH_FORMAT).extent(Extent::Fixed([config.shadow_atlas; 2])),
        );

        let id = builder
            .pass("punctual_shadows", PassKind::Inline)
            .access(object_transforms, Access::StorageRead)
            .access(atlas, Access::DepthAttachment)
            .build();
        record(id, PassBody::PunctualShadows, &mut bodies);

        atlas
    });

    // One prepass serves every consumer rather than one each: they need the same
    // rasterisation, and running it twice to hand each part of the result to a
    // different reader is the cost the shared node exists to avoid. It writes
    // all three targets whichever consumer asked for it — a second pipeline that
    // dropped the normal attachment for a TAA-without-SSAO frame would buy a
    // target's bandwidth at the price of a second render pass to keep in step.
    // Seven readers now want different parts of it: SSAO and contact shadows
    // take depth and normals, TAA and motion blur take depth and motion, depth
    // of field takes depth alone, reflections take depth, normals and the
    // material target — and transparency and refraction both attach the depth
    // read-only, the two readers that want it as an attachment rather than as a
    // texture.
    let prepass = (config.ssao
        || config.contact_shadows
        || config.taa
        || config.motion_blur
        || config.dof
        || config.ssr
        || config.transparency
        || config.refraction)
        .then(|| PrepassIds {
            normal: builder.create_image("prepass_normal", ImageDesc::new(NORMAL_FORMAT)),
            velocity: builder.create_image("prepass_velocity", ImageDesc::new(VELOCITY_FORMAT)),
            material: builder.create_image("prepass_material", ImageDesc::new(MATERIAL_FORMAT)),
            depth: builder.create_image("prepass_depth", ImageDesc::new(DEPTH_FORMAT)),
        });

    let ssao = config.ssao.then(|| SsaoIds {
        raw_ao: builder.create_image("ssao_raw_ao", ImageDesc::new(AO_FORMAT)),
        ao: builder.create_image("ssao_ao", ImageDesc::new(AO_FORMAT)),
    });

    let msaa_hdr =
        builder.create_image("msaa_hdr", ImageDesc::new(HDR_FORMAT).samples(MSAA_SAMPLES));
    let msaa_depth = builder.create_image(
        "msaa_depth",
        ImageDesc::new(DEPTH_FORMAT).samples(MSAA_SAMPLES),
    );
    let hdr_color = builder.create_image("hdr_color", ImageDesc::new(HDR_FORMAT));

    if let Some(prepass) = prepass {
        let id = builder
            .pass("geometry_prepass", PassKind::Inline)
            .access(object_transforms, Access::StorageRead)
            .access(prepass.normal, Access::ColorAttachment)
            .access(prepass.velocity, Access::ColorAttachment)
            .access(prepass.material, Access::ColorAttachment)
            .access(prepass.depth, Access::DepthAttachment)
            .build();
        record(id, PassBody::GeometryPrepass, &mut bodies);
    }

    if let Some(ssao) = ssao {
        let prepass = prepass.expect("SSAO reads the geometry prepass");
        let id = builder
            .pass("ssao_resolve", PassKind::Inline)
            .access(prepass.depth, Access::Sampled)
            .access(prepass.normal, Access::Sampled)
            .access(ssao.raw_ao, Access::ColorAttachment)
            .build();
        record(id, PassBody::SsaoResolve, &mut bodies);

        let id = builder
            .pass("ssao_blur", PassKind::Inline)
            .access(ssao.raw_ao, Access::Sampled)
            .access(ssao.ao, Access::ColorAttachment)
            .build();
        record(id, PassBody::SsaoBlur, &mut bodies);
    }

    // Between the prepass and shading, and it can be nowhere else: it marches
    // the depth that pass wrote, and the mask it leaves is what the forward pass
    // multiplies the sun's term by. Unblurred, unlike the AO beside it — this is
    // a hard visibility term whose noise is a dithered ray origin, and a spatial
    // filter would smear the contact it exists to sharpen. The temporal resolve
    // is what averages it.
    let contact_shadows = config.contact_shadows.then(|| {
        let prepass = prepass.expect("contact shadows read the geometry prepass");
        let mask = builder.create_image("contact_shadow_mask", ImageDesc::new(MASK_FORMAT));

        let id = builder
            .pass("contact_shadows", PassKind::Inline)
            .access(prepass.depth, Access::Sampled)
            .access(prepass.normal, Access::Sampled)
            .access(mask, Access::ColorAttachment)
            .build();
        record(id, PassBody::ContactShadows, &mut bodies);

        mask
    });

    let mut forward = builder
        .pass("forward", PassKind::Inline)
        .access(object_transforms, Access::StorageRead);
    if let Some(ssao) = ssao {
        forward = forward.access(ssao.ao, Access::Sampled);
    }
    if let Some(mask) = contact_shadows {
        forward = forward.access(mask, Access::Sampled);
    }
    if let Some(shadows) = shadows {
        forward = forward.access(shadows, Access::Sampled);
    }
    if let Some(atlas) = shadow_atlas {
        forward = forward.access(atlas, Access::Sampled);
    }
    let id = forward
        .access(msaa_hdr, Access::ColorAttachment)
        .access(msaa_depth, Access::DepthAttachment)
        .access(hdr_color, Access::ResolveAttachment)
        .build();
    record(id, PassBody::Forward, &mut bodies);

    // Between shading and the temporal resolve, and it has to be both: a ray can
    // only sample radiance that has been lit, and one ray per pixel is noise
    // until TAA has accumulated it over the jitter sequence. Putting it after
    // the resolve instead would mean denoising it separately, with a second
    // history of its own.
    let ssr = config.ssr.then(|| {
        let prepass = prepass.expect("screen-space reflections read the geometry prepass");
        let source = hdr_color;

        // One image carrying the whole pyramid rather than one per level, which
        // is what lets the trace pick a level per step with `textureLod`. The
        // build writes every level in a single pass because it must: a pass per
        // level, each reading the level above from the same resource, is a cycle
        // by the rule that makes readers follow writers.
        let hiz =
            builder.create_image("ssr_hiz", ImageDesc::new(HIZ_FORMAT).mip_levels(HIZ_LEVELS));
        // The same shape over the lit frame, and the reason a rough reflection
        // is affordable: a ray sampled from a wide lobe is a cone, and the level
        // whose texel matches the cone's footprint carries the average of what
        // the cone covers instead of one sample from inside it. Based at half
        // the frame, which is the resolution the trace works at anyway.
        let source_pyramid = builder.create_image(
            "ssr_source",
            ImageDesc::new(HDR_FORMAT)
                .extent(Extent::FrameDiv(1))
                .mip_levels(SOURCE_LEVELS),
        );
        // Half the frame, for the reason depth of field's gather is: the cost is
        // the ray, and a stochastic ray is denoised temporally either way, so
        // tracing four times as many buys far less than it costs.
        let rays = builder.create_image(
            "ssr_rays",
            ImageDesc::new(RAY_FORMAT).extent(Extent::FrameDiv(1)),
        );
        let output = builder.create_image("ssr_color", ImageDesc::new(HDR_FORMAT));

        let id = builder
            .pass("ssr_hiz", PassKind::Compute)
            .access(prepass.depth, Access::Sampled)
            .access(hiz, Access::StorageWrite)
            .build();
        record(id, PassBody::SsrHiz, &mut bodies);

        let id = builder
            .pass("ssr_source", PassKind::Compute)
            .access(source, Access::Sampled)
            .access(source_pyramid, Access::StorageWrite)
            .build();
        record(id, PassBody::SsrSource, &mut bodies);

        let id = builder
            .pass("ssr_trace", PassKind::Compute)
            .access(hiz, Access::Sampled)
            .access(prepass.depth, Access::Sampled)
            .access(prepass.normal, Access::Sampled)
            .access(prepass.material, Access::Sampled)
            .access(source_pyramid, Access::Sampled)
            .access(rays, Access::StorageWrite)
            .build();
        record(id, PassBody::SsrTrace, &mut bodies);

        // Reads `source` again rather than accumulating into it, for the reason
        // the depth-of-field composite does: resources are unversioned, so a
        // pass that both read and wrote the frame's colour would make "readers
        // after all writers" point in two directions and `compile` would report
        // a cycle.
        let id = builder
            .pass("ssr_resolve", PassKind::Compute)
            .access(source, Access::Sampled)
            .access(rays, Access::Sampled)
            .access(prepass.depth, Access::Sampled)
            .access(prepass.normal, Access::Sampled)
            .access(prepass.material, Access::Sampled)
            .access(output, Access::StorageWrite)
            .build();
        record(id, PassBody::SsrResolve, &mut bodies);

        SsrIds {
            source,
            hiz,
            source_pyramid,
            rays,
            output,
        }
    });

    // What the temporal resolve accumulates, and what the optical chain starts
    // from in a frame with no resolve: the reflections' output where they ran,
    // and the forward pass's own target where they did not.
    let mut shaded = ssr.map_or(hdr_color, |ssr| ssr.output);

    // After the reflections and before the resolve, and both halves matter.
    //
    // After, because the prepass records only opaque surfaces: a reflection
    // traces the depth and normals of the world *behind* the glass, and
    // compositing first would put the glass into a source the trace then
    // reflects as though it were a wall.
    //
    // Before, because the accumulation rasterises with the frame's jitter like
    // everything else, and a subpixel offset nothing averages is a shimmer. The
    // cost is that a moving transparent surface reprojects along the *opaque*
    // motion vectors under it and ghosts; the neighbourhood clamp takes most of
    // it, and the alternative is transparency with no antialiasing at all.
    let transparency = config.transparency.then(|| {
        let prepass = prepass.expect("transparency depth-tests the geometry prepass");
        let source = shaded;

        let accum = builder.create_image("oit_accum", ImageDesc::new(ACCUM_FORMAT));
        let reveal = builder.create_image("oit_reveal", ImageDesc::new(REVEAL_FORMAT));
        let output = builder.create_image("oit_color", ImageDesc::new(HDR_FORMAT));

        // The depth is *attached*, not sampled: a fixed-function depth test is
        // what makes a transparent surface disappear behind a wall, and a test
        // in the shader would need the depth in a second layout and a discard
        // per fragment. `DepthAttachmentRead` is the declaration that keeps it
        // read-only — the prepass is still its only writer, so every other
        // reader of it is unaffected.
        //
        // The same screen-space and shadow inputs the forward pass declares,
        // because it shades with the same `shading.glsl` and therefore samples
        // the same set.
        let mut accumulate = builder
            .pass("oit_accumulate", PassKind::Inline)
            .access(object_transforms, Access::StorageRead);
        if let Some(ssao) = ssao {
            accumulate = accumulate.access(ssao.ao, Access::Sampled);
        }
        if let Some(mask) = contact_shadows {
            accumulate = accumulate.access(mask, Access::Sampled);
        }
        if let Some(shadows) = shadows {
            accumulate = accumulate.access(shadows, Access::Sampled);
        }
        if let Some(atlas) = shadow_atlas {
            accumulate = accumulate.access(atlas, Access::Sampled);
        }
        let id = accumulate
            .access(accum, Access::ColorAttachment)
            .access(reveal, Access::ColorAttachment)
            .access(prepass.depth, Access::DepthAttachmentRead)
            .build();
        record(id, PassBody::OitAccumulate, &mut bodies);

        // Reads `source` again rather than accumulating into it, for the reason
        // the reflection and depth-of-field composites do: resources are
        // unversioned, so a pass that both read and wrote the frame's colour
        // would make "readers after all writers" point in two directions and
        // `compile` would report a cycle.
        let id = builder
            .pass("oit_composite", PassKind::Compute)
            .access(source, Access::Sampled)
            .access(accum, Access::Sampled)
            .access(reveal, Access::Sampled)
            .access(output, Access::StorageWrite)
            .build();
        record(id, PassBody::OitComposite, &mut bodies);

        TransparencyIds {
            source,
            accum,
            reveal,
            output,
        }
    });
    if let Some(transparency) = transparency {
        shaded = transparency.output;
    }

    // After the blended queue, so glass refracts what that queue composited, and
    // before the temporal resolve, so it rasterises with the frame's jitter like
    // every other geometry pass.
    //
    // The ordering between the two queues is a choice with no correct answer:
    // neither writes depth, so nothing sorts one against the other. Refraction
    // last means a blended surface *behind* glass is correctly refracted through
    // it, and one in front of glass is not — the commoner case wins, and the
    // uncommon one is what a screen-space technique cannot represent at all.
    let refraction = config.refraction.then(|| {
        let prepass = prepass.expect("refraction depth-tests the geometry prepass");
        let source = shaded;

        // Half the frame, the same base the reflection trace's pyramid of this
        // shape uses. Building it is the most expensive thing this feature does,
        // and every consumer of it is a surface rough enough to be averaging
        // over a cone anyway — the one consumer that is not, a clear pane, reads
        // `source` directly through the draw's second binding below.
        let scene = builder.create_image(
            "refraction_scene",
            ImageDesc::new(HDR_FORMAT)
                .extent(Extent::FrameDiv(1))
                .mip_levels(SCENE_LEVELS),
        );
        let accum =
            builder.create_image("refraction_accum", ImageDesc::new(REFRACTION_ACCUM_FORMAT));
        let output = builder.create_image("refraction_color", ImageDesc::new(HDR_FORMAT));

        let id = builder
            .pass("refraction_scene", PassKind::Compute)
            .access(source, Access::Sampled)
            .access(scene, Access::StorageWrite)
            .build();
        record(id, PassBody::RefractionScene, &mut bodies);

        // The depth is *attached*, not sampled, for the reason the transparency
        // accumulation attaches it: a fixed-function depth test is what makes a
        // refractive surface disappear behind a wall. `DepthAttachmentRead` is
        // the declaration that keeps it read-only.
        //
        // The same screen-space and shadow inputs the forward pass declares,
        // because it shades with the same `shading.glsl` — plus the pyramid
        // above, which is the one input no other geometry pass has.
        let mut draw = builder
            .pass("refraction_draw", PassKind::Inline)
            .access(object_transforms, Access::StorageRead)
            .access(scene, Access::Sampled)
            .access(source, Access::Sampled);
        if let Some(ssao) = ssao {
            draw = draw.access(ssao.ao, Access::Sampled);
        }
        if let Some(mask) = contact_shadows {
            draw = draw.access(mask, Access::Sampled);
        }
        if let Some(shadows) = shadows {
            draw = draw.access(shadows, Access::Sampled);
        }
        if let Some(atlas) = shadow_atlas {
            draw = draw.access(atlas, Access::Sampled);
        }
        let id = draw
            .access(accum, Access::ColorAttachment)
            .access(prepass.depth, Access::DepthAttachmentRead)
            .build();
        record(id, PassBody::RefractionDraw, &mut bodies);

        // Reads `source` again rather than accumulating into it, for the reason
        // every other composite in this frame does: resources are unversioned,
        // so a pass that both read and wrote the frame's colour would make
        // "readers after all writers" point in two directions and `compile`
        // would report a cycle.
        let id = builder
            .pass("refraction_composite", PassKind::Compute)
            .access(source, Access::Sampled)
            .access(accum, Access::Sampled)
            .access(output, Access::StorageWrite)
            .build();
        record(id, PassBody::RefractionComposite, &mut bodies);

        RefractionIds {
            source,
            scene,
            accum,
            output,
        }
    });
    if let Some(refraction) = refraction {
        shaded = refraction.output;
    }

    let taa = config.taa.then(|| {
        let prepass = prepass.expect("TAA reads the geometry prepass");
        // Entry `ShaderReadOnlyOptimal` states the steady state, which the
        // ping-pong guarantees: `output` below leaves every frame in exactly
        // that layout, and it is the allocation `history` names next frame. The
        // one frame where it is not true — the first after an allocation — is
        // the frame the pass is told to ignore its history anyway.
        let history = builder.import_image(
            "taa_history",
            ImageDesc::new(HDR_FORMAT),
            ImageLayout::ShaderReadOnlyOptimal,
            ImageLayout::ShaderReadOnlyOptimal,
        );
        // Entered `Undefined` because every texel is written, and left where the
        // next frame wants to find it.
        let output = builder.import_image(
            "taa_color",
            ImageDesc::new(HDR_FORMAT),
            ImageLayout::Undefined,
            ImageLayout::ShaderReadOnlyOptimal,
        );

        let id = builder
            .pass("taa_resolve", PassKind::Compute)
            .access(shaded, Access::Sampled)
            .access(prepass.velocity, Access::Sampled)
            .access(prepass.depth, Access::Sampled)
            .access(history, Access::Sampled)
            .access(output, Access::StorageWrite)
            .build();
        record(id, PassBody::TaaResolve, &mut bodies);

        TaaIds {
            source: shaded,
            history,
            output,
        }
    });

    // Everything past shading reads this rather than `hdr_color`, so inserting
    // the resolve is a change to one binding rather than to every consumer.
    // Each optical stage below rebinds it in turn, which is what makes the
    // chain's order a property of this function alone.
    let mut scene_color = taa.map_or(shaded, |taa| taa.output);

    // Lens before shutter before sensor, which is the order light actually
    // meets them and the order Unity and Unreal both settled on. Defocus first,
    // so what the shutter smears is already a lens image; both before bloom and
    // metering, so a defocused highlight blooms as the wide soft thing it has
    // become rather than as the point it was.
    let dof = config.dof.then(|| {
        let prepass = prepass.expect("depth of field reads the geometry prepass");
        let source = scene_color;

        // Half the frame for everything but the composite. The gather is the
        // only pass that pays for the kernel, and paying for it at a quarter of
        // the pixels is what makes a 48-tap disc affordable at all; the
        // composite's bilinear upsample puts back more detail than the extra
        // resolution would have carried, because the field it is upsampling is
        // by definition out of focus.
        let prefiltered = builder.create_image(
            "dof_prefiltered",
            ImageDesc::new(HDR_FORMAT).extent(Extent::FrameDiv(1)),
        );
        let near = builder.create_image(
            "dof_near",
            ImageDesc::new(HDR_FORMAT).extent(Extent::FrameDiv(1)),
        );
        let far = builder.create_image(
            "dof_far",
            ImageDesc::new(HDR_FORMAT).extent(Extent::FrameDiv(1)),
        );
        // Two channels of maxima in an `HDR_FORMAT` image, for the reason the
        // motion blur tiles are: `R16G16_SFLOAT` is only a guaranteed storage
        // format behind an optional device feature, and at one texel per 4096
        // the unused half is not worth a feature flag.
        let tile = builder.create_image(
            "dof_tile",
            ImageDesc::new(HDR_FORMAT).extent(Extent::FrameDiv(COC_TILE_SHIFT)),
        );
        let output = builder.create_image("dof_color", ImageDesc::new(HDR_FORMAT));

        let id = builder
            .pass("dof_prefilter", PassKind::Compute)
            .access(source, Access::Sampled)
            .access(prepass.depth, Access::Sampled)
            .access(prefiltered, Access::StorageWrite)
            .build();
        record(id, PassBody::DofPrefilter, &mut bodies);

        // Between the prefilter and the gather for the same reason motion blur's
        // tile pass sits between the velocities and its gather: a fixed tap
        // budget has to be spread over the blur that is actually there, not over
        // the widest one the lens could produce.
        let id = builder
            .pass("dof_tile_max", PassKind::Compute)
            .access(prefiltered, Access::Sampled)
            .access(tile, Access::StorageWrite)
            .build();
        record(id, PassBody::DofTileMax, &mut bodies);

        // One dispatch writing both fields rather than two over the same tiles:
        // they are gathered with different kernels but read the same maxima.
        let id = builder
            .pass("dof_gather", PassKind::Compute)
            .access(prefiltered, Access::Sampled)
            .access(tile, Access::Sampled)
            .access(near, Access::StorageWrite)
            .access(far, Access::StorageWrite)
            .build();
        record(id, PassBody::DofGather, &mut bodies);

        // Reads `source` again rather than accumulating into it, and has to:
        // resources are unversioned, so a pass that both read and wrote the
        // frame's colour would make "readers after all writers" point in two
        // directions at once and `compile` would report a cycle.
        let id = builder
            .pass("dof_composite", PassKind::Compute)
            .access(source, Access::Sampled)
            .access(prepass.depth, Access::Sampled)
            .access(near, Access::Sampled)
            .access(far, Access::Sampled)
            .access(output, Access::StorageWrite)
            .build();
        record(id, PassBody::DofComposite, &mut bodies);

        DofIds {
            source,
            prefiltered,
            tile,
            near,
            far,
            output,
        }
    });
    if let Some(dof) = dof {
        scene_color = dof.output;
    }

    let motion_blur = config.motion_blur.then(|| {
        let prepass = prepass.expect("motion blur reads the geometry prepass");
        let source = scene_color;

        // `FrameDiv(TILE_SHIFT)` and not a fixed size, because a tile has to be
        // exactly what successive halvings produce: the gather derives its tile
        // from its own pixel coordinate, and a level sized any other way is a
        // texel off from the one it means to read at odd extents.
        //
        // Both carry a two-component vector in an `HDR_FORMAT` image. A storage
        // image is only guaranteed to support `R16G16_SFLOAT` behind an optional
        // device feature, while `R16G16B16A16_SFLOAT` is always available — and
        // at one texel per 256 the two unused channels are not worth a feature
        // flag on the device.
        let tile = builder.create_image(
            "motion_blur_tile",
            ImageDesc::new(HDR_FORMAT).extent(Extent::FrameDiv(TILE_SHIFT)),
        );
        let neighbour = builder.create_image(
            "motion_blur_neighbour",
            ImageDesc::new(HDR_FORMAT).extent(Extent::FrameDiv(TILE_SHIFT)),
        );
        let output = builder.create_image("motion_blur_color", ImageDesc::new(HDR_FORMAT));

        // Depth as well as velocity, because where nothing was rasterised the
        // prepass wrote no motion and the sky still sweeps when the camera
        // turns. Both this pass and the gather recover it the same way TAA does,
        // by reprojecting the far plane.
        let id = builder
            .pass("motion_blur_tile_max", PassKind::Compute)
            .access(prepass.velocity, Access::Sampled)
            .access(prepass.depth, Access::Sampled)
            .access(tile, Access::StorageWrite)
            .build();
        record(id, PassBody::MotionBlurTileMax, &mut bodies);

        let id = builder
            .pass("motion_blur_neighbour_max", PassKind::Compute)
            .access(tile, Access::Sampled)
            .access(neighbour, Access::StorageWrite)
            .build();
        record(id, PassBody::MotionBlurNeighbourMax, &mut bodies);

        let id = builder
            .pass("motion_blur_gather", PassKind::Compute)
            .access(source, Access::Sampled)
            .access(prepass.velocity, Access::Sampled)
            .access(prepass.depth, Access::Sampled)
            .access(neighbour, Access::Sampled)
            .access(output, Access::StorageWrite)
            .build();
        record(id, PassBody::MotionBlurGather, &mut bodies);

        MotionBlurIds {
            source,
            tile,
            neighbour,
            output,
        }
    });
    if let Some(motion_blur) = motion_blur {
        scene_color = motion_blur.output;
    }

    // Imported, not created: temporal adaptation is a value that has to outlive
    // the frame, and a transient is `Undefined` at every frame's start by
    // contract. The exposure module owns the allocation; the graph only orders
    // access to it.
    let exposure = builder.import_buffer("exposure");

    let histogram = config.auto_exposure.then(|| {
        let histogram = builder.import_buffer("luminance_histogram");

        let id = builder
            .pass("luminance_histogram", PassKind::Compute)
            .access(scene_color, Access::Sampled)
            .access(histogram, Access::StorageWrite)
            .build();
        record(id, PassBody::LuminanceHistogram, &mut bodies);

        // One `StorageWrite` covers both halves of what this pass does to each
        // buffer — it reads the bins and zeroes them, reads last frame's adapted
        // luminance and replaces it. Declaring the read separately would be two
        // accesses to one resource in one pass, which the compiler rejects
        // because a pass is a single point in the schedule.
        let id = builder
            .pass("luminance_average", PassKind::Compute)
            .access(histogram, Access::StorageWrite)
            .access(exposure, Access::StorageWrite)
            .build();
        record(id, PassBody::LuminanceAverage, &mut bodies);

        histogram
    });

    let bloom = (config.bloom_mips > 0).then(|| {
        let mips = config.bloom_mips as usize;
        let mut down = [None; MAX_BLOOM_MIPS];
        let mut up = [None; MAX_BLOOM_MIPS];

        // Level `n` is the frame halved `n + 1` times, so level 0 is half the
        // frame. Sized off the frame rather than off the level above it, so a
        // resize moves the whole chain together.
        for level in 0..mips {
            let shift = level as u32 + 1;
            down[level] = Some(builder.create_image(
                BLOOM_DOWN_NAMES[level],
                ImageDesc::new(HDR_FORMAT).extent(Extent::FrameDiv(shift)),
            ));
            if level + 1 < mips {
                up[level] = Some(builder.create_image(
                    BLOOM_UP_NAMES[level],
                    ImageDesc::new(HDR_FORMAT).extent(Extent::FrameDiv(shift)),
                ));
            }
        }

        let id = builder
            .pass("bloom_prefilter", PassKind::Compute)
            .access(scene_color, Access::Sampled)
            .access(exposure, Access::StorageRead)
            .access(down[0].unwrap(), Access::StorageWrite)
            .build();
        record(id, PassBody::BloomPrefilter, &mut bodies);

        for level in 1..mips {
            let id = builder
                .pass(BLOOM_DOWN_PASS_NAMES[level], PassKind::Compute)
                .access(down[level - 1].unwrap(), Access::Sampled)
                .access(down[level].unwrap(), Access::StorageWrite)
                .build();
            record(id, PassBody::BloomDownsample(level as u32), &mut bodies);
        }

        // Back down the chain, coarsest first. The coarsest step spreads the
        // down chain's last level; every step after it spreads the up-chain
        // level it just produced.
        for level in (0..mips.saturating_sub(1)).rev() {
            let coarse = if level + 2 < mips {
                up[level + 1].unwrap()
            } else {
                down[level + 1].unwrap()
            };
            let id = builder
                .pass(BLOOM_UP_PASS_NAMES[level], PassKind::Compute)
                .access(coarse, Access::Sampled)
                .access(down[level].unwrap(), Access::Sampled)
                .access(up[level].unwrap(), Access::StorageWrite)
                .build();
            record(id, PassBody::BloomUpsample(level as u32), &mut bodies);
        }

        BloomIds {
            mips: config.bloom_mips,
            down,
            up,
        }
    });

    // Reading `exposure` is also what orders this behind the metering: nothing
    // else connects the two, since tonemap and the histogram pass both only read
    // `hdr_color`.
    let mut tonemap = builder
        .pass("tonemap", PassKind::Inline)
        .access(scene_color, Access::Sampled)
        .access(exposure, Access::StorageRead);
    if let Some(bloom) = bloom {
        tonemap = tonemap.access(bloom.result(), Access::Sampled);
    }
    let id = tonemap
        .access(swapchain_color, Access::ColorAttachment)
        .build();
    record(id, PassBody::Tonemap, &mut bodies);

    if config.overlay {
        let id = builder
            .pass("overlay", PassKind::Raw)
            .access(swapchain_color, Access::ColorAttachment)
            .build();
        record(id, PassBody::Overlay, &mut bodies);
    }

    Ok(Frame {
        graph: compile(builder)?,
        ids: FrameIds {
            object_transforms,
            swapchain_color,
            hdr_color,
            scene_color,
            msaa_hdr,
            msaa_depth,
            exposure,
            histogram,
            bloom,
            prepass,
            ssao,
            contact_shadows,
            ssr,
            transparency,
            refraction,
            taa,
            dof,
            motion_blur,
            shadows,
            shadow_atlas,
        },
        bodies,
    })
}
