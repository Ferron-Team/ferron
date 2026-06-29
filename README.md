# Ferron

A game engine you actually own, built in Rust, scripted in C#, with a modern
editor and a renderer aiming for all kinds of 3d games. Low poly games to hyperrealism.

Ferron is an early-stage, open game engine. The whole stack; core, renderer,
and editor, is yours to read, change, and ship. No black boxes, no licensing
traps.

## Goals

### You own the code *and* the editor
Full source for everything, the editor included. Read it, fork it, bend it to
your project. No proprietary runtime, no revenue cut, no lock-in — your game and
your tools belong to you.

### An editor that's a joy to use
A clean, modern, good-looking editor that stays out of your way. Tooling should
feel as good as the games you make with it.

### Built for collaboration
Working together is a first-class concern, not an afterthought — a project
structure and workflow designed for teams, with real-time collaborative editing
as the goal.

### Real-time ray tracing
A renderer built for modern hardware, with real-time ray-traced lighting as the
north star.

## What works today
- **Rust core** with a lightweight ECS — entities, components, queries, resources.
- **Vulkan renderer** (`vulkano`): forward+ shading, MSAA, SSAO, HDR tonemapping,
  point & directional lights, textured materials.
- **In-window editor** (`egui`): scene hierarchy, inspector, environment controls,
  and a live performance HUD (FPS, CPU/GPU frame time, VRAM).
- **C# scripting** (.NET / CoreCLR): Unity-style `Behaviour` scripts with
  `OnStart`/`OnUpdate`, driven from the engine.

## Tech
- **Core & renderer:** Rust + Vulkan (`vulkano`)
- **Scripting:** C# on .NET, hosted in-process via CoreCLR
- **Editor:** `egui`

## Quick start
```sh
# Run the engine
cargo run -p renderer-prototype

# With C# scripting (requires the .NET SDK)
dotnet build scripting/Ferron -c Debug
cargo run -p renderer-prototype --features scripting
```

## Status
Early and moving fast — expect APIs to change. Real-time ray tracing and
collaborative editing are goals on the horizon, not shipped features yet.
