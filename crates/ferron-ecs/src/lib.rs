//! `ferron-ecs` - a small, dependency-free Entity Component System for game engines

// FERRON-ECS
// AUTHOR: @AlternativeLua

#![forbid(unsafe_code)]

use std::any::{Any, TypeId};
use std::cell::{Ref, RefCell, RefMut};
use std::collections::HashMap;
use std::marker::PhantomData;

//
// Entities
//

/// A small, copyable handle to a single entity.
///
/// An entity is identified by a slot `index` plus a `generation`. When a slot
/// is reused by a later entity the generation changes, so a stale handle to a
/// despawned entity can be detected instead of silently aliasing the new one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Entity {
    /// Index of the storage slot this entity occupies.
    pub index: u32,
    /// How many times this slot has been reused; bumped on every despawn.
    pub generation: u32,
}

impl Entity {
    /// The storage slot this entity occupies.
    #[inline]
    pub fn index(self) -> u32 {
        self.index
    }

    /// The generation stamp, used to tell a live handle from a stale one.
    #[inline]
    pub fn generation(self) -> u32 {
        self.generation
    }
}

// Hands out entity ids and recycles the indices
#[derive(Default)]
struct EntityAllocator {
    generations: Vec<u32>,
    alive: Vec<bool>,
    free: Vec<u32>,
}

impl EntityAllocator {
    fn allocate(&mut self) -> Entity {
        if let Some(index) = self.free.pop() {
            self.alive[index as usize] = true;
            Entity {
                index,
                generation: self.generations[index as usize],
            }
        } else {
            let index = self.generations.len() as u32;
            self.generations.push(0);
            self.alive.push(true);
            Entity {
                index,
                generation: 0
            }
        }
    }

    fn deallocate(&mut self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }

        let i = entity.index() as usize;
        self.generations[i] = self.generations[i].wrapping_add(1);
        self.alive[i] = false;
        self.free.push(entity.index);
        true
    }

    fn is_alive(&self, entity: Entity) -> bool {
        let i = entity.index as usize;
        i < self.generations.len() && self.alive[i] && self.generations[i] == entity.generation
    }

    /// Iterate over every currently-live entity, skipping freed slots.
    fn iter_alive(&self) -> impl Iterator<Item = Entity> + '_ {
        self.alive
            .iter()
            .enumerate()
            .filter_map(move |(i, &alive)| {
                alive.then(|| Entity {
                    index: i as u32,
                    generation: self.generations[i],
                })
            })
    }
}

//
// Component Storage
//

const SENTINEL: u32 = u32::MAX;

/// Dense storage for a single component type `T`.
///
/// `sparse` maps an entity index to a slot in the packed `dense_*` arrays, and
/// `dense_entities` maps back the other way so lookups can run a generation
/// check. Keeping the values packed means iteration never walks empty holes.
pub struct SparseSet<T> {
    sparse: Vec<u32>,
    dense_entities: Vec<Entity>,
    dense_values: Vec<T>,
}

impl<T> SparseSet<T> {
    fn new() -> Self {
        Self {
            sparse: Vec::new(),
            dense_entities: Vec::new(),
            dense_values: Vec::new(),
        }
    }

    fn dense_index(&self, entity: Entity) -> Option<usize> {
        let i = entity.index() as usize;
        let d = *self.sparse.get(i)?;
        if d == SENTINEL {
            return None;
        }

        if self.dense_entities[d as usize] == entity {
            Some(d as usize)
        } else {
            None
        }
    }

    fn insert(&mut self, entity: Entity, value: T) -> Option<T> {
        let i = entity.index as usize;
        if i >= self.sparse.len() {
            self.sparse.resize(i + 1, SENTINEL);
        }
        let d = self.sparse[i];
        if d != SENTINEL && self.dense_entities[d as usize] == entity {
            return Some(std::mem::replace(&mut self.dense_values[d as usize], value));
        }
        self.sparse[i] = self.dense_values.len() as u32;
        self.dense_entities.push(entity);
        self.dense_values.push(value);
        None
    }

    fn remove(&mut self, entity: Entity) -> Option<T> {
        let d = self.dense_index(entity)?;
        let last = self.dense_values.len() - 1;
        self.dense_values.swap(d, last);
        self.dense_entities.swap(d, last);
        let moved = self.dense_entities[d];
        self.sparse[moved.index as usize] = d as u32;
        self.sparse[entity.index as usize] = SENTINEL;
        self.dense_entities.pop();
        self.dense_values.pop()
    }

    fn get(&self, entity: Entity) -> Option<&T> {
        self.dense_index(entity).map(|d| &self.dense_values[d])
    }

    fn get_mut(&mut self, entity: Entity) -> Option<&mut T> {
        match self.dense_index(entity) {
            Some(d) => Some(&mut self.dense_values[d]),
            None => None,
        }
    }
}

trait AnyStorage: Any {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn remove_entity(&mut self, entity: Entity);
}

impl<T: 'static> AnyStorage for SparseSet<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn remove_entity(&mut self, entity: Entity) {
        self.remove(entity);
    }
}

//
// World
//

/// The container for all entities, their components, and global resources.
///
/// Almost everything goes through a `World`: [`spawn`](World::spawn) creates
/// entities, [`insert`](World::insert) attaches components, and
/// [`query`](World::query) iterates over them.
#[derive(Default)]
pub struct World {
    entities: EntityAllocator,
    storages: HashMap<TypeId, RefCell<Box<dyn AnyStorage>>>,
    resources: HashMap<TypeId, RefCell<Box<dyn Any>>>,
}

impl World {
    /// Create an empty world.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new entity with no components and return its handle.
    pub fn spawn(&mut self) -> Entity {
        self.entities.allocate()
    }

    /// Spawn a new entity and return a builder for attaching components to it in
    /// one chained expression:
    ///
    /// ```
    /// # use ferron_ecs::World;
    /// # struct Position(f32);
    /// # struct Velocity(f32);
    /// let mut world = World::new();
    /// let entity = world
    ///     .spawn_entity()
    ///     .with(Position(0.0))
    ///     .with(Velocity(1.0))
    ///     .id();
    /// ```
    pub fn spawn_entity(&mut self) -> EntityBuilder<'_> {
        let entity = self.spawn();
        EntityBuilder { world: self, entity }
    }

    /// Returns `true` while `entity` refers to a live (not yet despawned) entity.
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity)
    }

    /// Iterate over every live entity in the world.
    ///
    /// Unlike [`query`](World::query), this needs no component and visits even
    /// entities with no components attached — handy for tooling that walks the
    /// whole world (e.g. an editor hierarchy, or a "despawn everything" pass).
    pub fn entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.entities.iter_alive()
    }

    /// Remove an entity and all of its components.
    ///
    /// Returns `false` if the handle was already stale.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }
        for storage in self.storages.values() {
            storage.borrow_mut().remove_entity(entity);
        }
        self.entities.deallocate(entity)
    }

    /// Attach a component to `entity`, returning the previous value if one was
    /// already present.
    ///
    /// Does nothing and returns `None` if `entity` is stale (already despawned).
    /// Without this guard a write through a dangling handle would create a
    /// "zombie" component that no later despawn can ever reclaim.
    pub fn insert<T: 'static>(&mut self, entity: Entity, component: T) -> Option<T> {
        if !self.entities.is_alive(entity) {
            return None;
        }
        let cell = self
            .storages
            .entry(TypeId::of::<T>())
            .or_insert_with(|| RefCell::new(Box::new(SparseSet::<T>::new())));
        let mut guard = cell.borrow_mut();
        let set = guard
            .as_any_mut()
            .downcast_mut::<SparseSet<T>>()
            .expect("storage type mismatch");
        set.insert(entity, component)
    }

    /// Detach and return `entity`'s component of type `T`, if it has one.
    pub fn remove<T: 'static>(&mut self, entity: Entity) -> Option<T> {
        let cell = self.storages.get(&TypeId::of::<T>())?;
        let mut guard = cell.borrow_mut();
        let set = guard
            .as_any_mut()
            .downcast_mut::<SparseSet<T>>()
            .expect("storage type mismatch");
        set.remove(entity)
    }

    /// Returns `true` if `entity` currently has a component of type `T`.
    pub fn has<T: 'static>(&self, entity: Entity) -> bool {
        self.get::<T>(entity).is_some()
    }

    /// Borrow `entity`'s component of type `T`, if present.
    pub fn get<T: 'static>(&self, entity: Entity) -> Option<Ref<'_, T>> {
        let cell = self.storages.get(&TypeId::of::<T>())?;
        Ref::filter_map(cell.borrow(), |b| {
            b.as_any()
                .downcast_ref::<SparseSet<T>>()
                .expect("storage type mismatch")
                .get(entity)
        })
            .ok()
    }

    /// Mutably borrow `entity`'s component of type `T`, if present.
    pub fn get_mut<T: 'static>(&self, entity: Entity) -> Option<RefMut<'_, T>> {
        let cell = self.storages.get(&TypeId::of::<T>())?;
        RefMut::filter_map(cell.borrow_mut(), |b| {
            b.as_any_mut()
                .downcast_mut::<SparseSet<T>>()
                .expect("storage type mismatch")
                .get_mut(entity)
        })
            .ok()
    }

    /// Iterate over every entity that has all the components in `Q`.
    ///
    /// `Q` is a reference or a tuple of references, e.g. `&Position` or
    /// `(&mut Position, &Velocity)`. Call [`for_each`](QueryRunner::for_each) on
    /// the returned runner.
    pub fn query<Q: QueryParam>(&self) -> QueryRunner<'_, Q> {
        QueryRunner {
            world: self,
            _marker: PhantomData,
        }
    }

    /// Store a unique, world-global value of type `R`, replacing any existing one.
    pub fn insert_resource<R: 'static>(&mut self, resource: R) {
        self.resources
            .insert(TypeId::of::<R>(), RefCell::new(Box::new(resource)));
    }

    /// Remove and return the resource of type `R`, if present.
    pub fn remove_resource<R: 'static>(&mut self) -> Option<R> {
        let cell = self.resources.remove(&TypeId::of::<R>())?;
        let boxed = cell.into_inner();
        boxed.downcast::<R>().ok().map(|b| *b)
    }

    /// Borrow the resource of type `R`.
    ///
    /// # Panics
    /// Panics if no resource of type `R` has been inserted.
    pub fn resource<R: 'static>(&self) -> Ref<'_, R> {
        self.get_resource::<R>()
            .expect("resource not found; insert it with `insert_resource` first")
    }

    /// Mutably borrow the resource of type `R`.
    ///
    /// # Panics
    /// Panics if no resource of type `R` has been inserted.
    pub fn resource_mut<R: 'static>(&self) -> RefMut<'_, R> {
        self.get_resource_mut::<R>()
            .expect("resource not found; insert it with `insert_resource` first")
    }

    /// Borrow the resource of type `R`, or `None` if it has not been inserted.
    pub fn get_resource<R: 'static>(&self) -> Option<Ref<'_, R>> {
        let cell = self.resources.get(&TypeId::of::<R>())?;
        Some(Ref::map(cell.borrow(), |b| {
            b.downcast_ref::<R>().expect("resource type mismatch")
        }))
    }

    /// Mutably borrow the resource of type `R`, or `None` if it is absent.
    pub fn get_resource_mut<R: 'static>(&self) -> Option<RefMut<'_, R>> {
        let cell = self.resources.get(&TypeId::of::<R>())?;
        Some(RefMut::map(cell.borrow_mut(), |b| {
            b.downcast_mut::<R>().expect("resource type mismatch")
        }))
    }
}

//
// Entity builder
//

/// A fluent builder for attaching components to a freshly-spawned entity.
///
/// Returned by [`World::spawn_entity`]. Each [`with`](EntityBuilder::with) call
/// attaches one component and returns the builder, so spawning reads as a single
/// expression instead of a `spawn` followed by a run of `insert` calls. Finish
/// with [`id`](EntityBuilder::id) to get the [`Entity`] handle.
pub struct EntityBuilder<'w> {
    world: &'w mut World,
    entity: Entity,
}

impl<'w> EntityBuilder<'w> {
    /// Attach a component to the entity being built.
    #[inline]
    pub fn with<T: 'static>(self, component: T) -> Self {
        self.world.insert(self.entity, component);
        self
    }

    /// Finish building and return the entity's handle.
    #[inline]
    pub fn id(self) -> Entity {
        self.entity
    }
}

//
// Queries
//

/// A component access pattern that a [`World::query`] can iterate.
///
/// Implemented for `&T` (read), `&mut T` (write), and tuples of those, so a
/// query like `(&mut Position, &Velocity)` matches entities that have both.
pub trait QueryParam {
    /// Borrowed handle(s) to the backing storage for the duration of one query.
    type Fetch<'w>;
    /// What a single matched entity yields, e.g. `&T` or `(&mut A, &B)`.
    type Item<'a>;

    /// Borrow the storage this query needs, or `None` if it isn't present.
    fn init(world: &World) -> Option<Self::Fetch<'_>>;

    /// Number of candidate entities to scan (driven by the first parameter).
    fn len(fetch: &Self::Fetch<'_>) -> usize;

    /// The entity at candidate position `i`.
    fn entity_at(fetch: &Self::Fetch<'_>, i: usize) -> Entity;

    /// Fetch the item for `entity`, or `None` if it lacks one of the components.
    fn get<'a>(fetch: &'a mut Self::Fetch<'_>, entity: Entity) -> Option<Self::Item<'a>>;
}

impl<T: 'static> QueryParam for &T {
    type Fetch<'w> = Ref<'w, SparseSet<T>>;
    type Item<'a> = &'a T;

    fn init(world: &World) -> Option<Self::Fetch<'_>> {
        let cell = world.storages.get(&TypeId::of::<T>())?;
        Some(Ref::map(cell.borrow(), |b| {
            b.as_any()
                .downcast_ref::<SparseSet<T>>()
                .expect("storage type mismatch")
        }))
    }

    fn len(fetch: &Self::Fetch<'_>) -> usize {
        fetch.dense_entities.len()
    }

    fn entity_at(fetch: &Self::Fetch<'_>, i: usize) -> Entity {
        fetch.dense_entities[i]
    }

    fn get<'a>(fetch: &'a mut Self::Fetch<'_>, entity: Entity) -> Option<Self::Item<'a>> {
        fetch.get(entity)
    }
}

impl<T: 'static> QueryParam for &mut T {
    type Fetch<'w> = RefMut<'w, SparseSet<T>>;
    type Item<'a> = &'a mut T;

    fn init(world: &World) -> Option<Self::Fetch<'_>> {
        let cell = world.storages.get(&TypeId::of::<T>())?;
        Some(RefMut::map(cell.borrow_mut(), |b| {
            b.as_any_mut()
                .downcast_mut::<SparseSet<T>>()
                .expect("storage type mismatch")
        }))
    }

    fn len(fetch: &Self::Fetch<'_>) -> usize {
        fetch.dense_entities.len()
    }

    fn entity_at(fetch: &Self::Fetch<'_>, i: usize) -> Entity {
        fetch.dense_entities[i]
    }

    fn get<'a>(fetch: &'a mut Self::Fetch<'_>, entity: Entity) -> Option<Self::Item<'a>> {
        fetch.get_mut(entity)
    }
}

macro_rules! impl_query_for_tuple {
    ($first:ident => $first_idx:tt $(, $name:ident => $idx:tt)*) => {
        impl<$first: QueryParam $(, $name: QueryParam)*> QueryParam for ($first, $($name,)*) {
            type Fetch<'w> = ($first::Fetch<'w>, $($name::Fetch<'w>,)*);
            type Item<'a> = ($first::Item<'a>, $($name::Item<'a>,)*);

            fn init(world: &World) -> Option<Self::Fetch<'_>> {
                Some(($first::init(world)?, $($name::init(world)?,)*))
            }

            fn len(fetch: &Self::Fetch<'_>) -> usize {
                $first::len(&fetch.$first_idx)
            }

            fn entity_at(fetch: &Self::Fetch<'_>, i: usize) -> Entity {
                $first::entity_at(&fetch.$first_idx, i)
            }

            fn get<'a>(fetch: &'a mut Self::Fetch<'_>, entity: Entity) -> Option<Self::Item<'a>> {
                // Disjoint mutable borrows of distinct tuple fields are fine.
                Some((
                    $first::get(&mut fetch.$first_idx, entity)?,
                    $($name::get(&mut fetch.$idx, entity)?,)*
                ))
            }
        }
    };
}

impl_query_for_tuple!(A => 0, B => 1);
impl_query_for_tuple!(A => 0, B => 1, C => 2);
impl_query_for_tuple!(A => 0, B => 1, C => 2, D => 3);
impl_query_for_tuple!(A => 0, B => 1, C => 2, D => 3, E => 4);

/// Runs a prepared query. Created by [`World::query`].
pub struct QueryRunner<'w, Q> {
    world: &'w World,
    _marker: PhantomData<Q>,
}

impl<'w, Q: QueryParam> QueryRunner<'w, Q> {
    /// Call `f` once for every entity that matches the query `Q`.
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(Entity, Q::Item<'_>),
    {
        let Some(mut fetch) = Q::init(self.world) else {
            return;
        };
        let count = Q::len(&fetch);
        for i in 0..count {
            let entity = Q::entity_at(&fetch, i);
            if let Some(item) = Q::get(&mut fetch, entity) {
                f(entity, item);
            }
        }
    }

    /// Count how many entities match the query, without invoking a callback.
    pub fn count(&self) -> usize {
        let Some(mut fetch) = Q::init(self.world) else {
            return 0;
        };
        let total = Q::len(&fetch);
        let mut matched = 0;
        for i in 0..total {
            let entity = Q::entity_at(&fetch, i);
            if Q::get(&mut fetch, entity).is_some() {
                matched += 1;
            }
        }
        matched
    }

    /// The first matching entity (in storage order) for which `pred` returns true.
    pub fn find<F>(&self, mut pred: F) -> Option<Entity>
    where
        F: FnMut(Entity, Q::Item<'_>) -> bool,
    {
        let mut fetch = Q::init(self.world)?;
        let total = Q::len(&fetch);
        for i in 0..total {
            let entity = Q::entity_at(&fetch, i);
            if let Some(item) = Q::get(&mut fetch, entity) {
                if pred(entity, item) {
                    return Some(entity);
                }
            }
        }
        None
    }
}

//
// Tests
//i

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }
    #[derive(Debug)]
    struct Velocity {
        x: f32,
        y: f32,
    }
    #[derive(Debug, PartialEq)]
    struct Health(i32);
    struct Frozen; // a marker / tag component

    struct DeltaTime(f32); // a resource

    #[test]
    fn insert_get_remove() {
        let mut world = World::new();
        let e = world.spawn();
        assert!(world.insert(e, Position { x: 1.0, y: 2.0 }).is_none());
        assert_eq!(world.get::<Position>(e).unwrap().x, 1.0);
        assert!(world.has::<Position>(e));

        // Overwrite returns the old value.
        let old = world.insert(e, Position { x: 9.0, y: 9.0 }).unwrap();
        assert_eq!(old, Position { x: 1.0, y: 2.0 });

        let removed = world.remove::<Position>(e).unwrap();
        assert_eq!(removed, Position { x: 9.0, y: 9.0 });
        assert!(!world.has::<Position>(e));
    }

    #[test]
    fn movement_system() {
        let mut world = World::new();
        world.insert_resource(DeltaTime(0.5));

        for i in 0..4 {
            let e = world.spawn();
            world.insert(e, Position { x: 0.0, y: 0.0 });
            world.insert(e, Velocity { x: i as f32, y: 1.0 });
        }
        // One entity with no Velocity should be skipped by the query.
        let stationary = world.spawn();
        world.insert(stationary, Position { x: 100.0, y: 100.0 });

        let dt = world.resource::<DeltaTime>().0;
        let mut visited = 0;
        world
            .query::<(&mut Position, &Velocity)>()
            .for_each(|_e, (pos, vel)| {
                pos.x += vel.x * dt;
                pos.y += vel.y * dt;
                visited += 1;
            });

        assert_eq!(visited, 4);
        assert_eq!(world.query::<(&Position, &Velocity)>().count(), 4);
        // The stationary entity wasn't matched, so it didn't move.
        assert_eq!(world.get::<Position>(stationary).unwrap().x, 100.0);
    }

    #[test]
    fn tag_components_and_three_param_query() {
        let mut world = World::new();
        let a = world.spawn();
        world.insert(a, Position { x: 0.0, y: 0.0 });
        world.insert(a, Velocity { x: 1.0, y: 1.0 });
        world.insert(a, Frozen);

        let b = world.spawn();
        world.insert(b, Position { x: 0.0, y: 0.0 });
        world.insert(b, Velocity { x: 1.0, y: 1.0 });

        // Only `a` has all three components.
        let mut matched = Vec::new();
        world
            .query::<(&mut Position, &Velocity, &Frozen)>()
            .for_each(|e, (_pos, _vel, _frozen)| matched.push(e));
        assert_eq!(matched, vec![a]);
    }

    #[test]
    fn find_first_match_in_storage_order() {
        let mut world = World::new();
        let a = world.spawn();
        world.insert(a, Health(10));
        let b = world.spawn();
        world.insert(b, Health(3));
        world.spawn(); // no components; never a candidate

        // First match follows dense (insertion) order.
        assert_eq!(world.query::<&Health>().find(|_, h| h.0 > 0), Some(a));
        // The predicate can select a later entity.
        assert_eq!(world.query::<&Health>().find(|_, h| h.0 < 5), Some(b));
        // No matching entity, and no storage registered at all, both give None.
        assert_eq!(world.query::<&Health>().find(|_, h| h.0 > 99), None);
        assert_eq!(world.query::<&Position>().find(|_, _| true), None);
    }

    #[test]
    fn find_stops_at_first_match() {
        let mut world = World::new();
        for hp in [1, 2, 3] {
            let e = world.spawn();
            world.insert(e, Health(hp));
        }

        let mut visited = 0;
        let found = world.query::<&Health>().find(|_, _| {
            visited += 1;
            true
        });
        assert!(found.is_some());
        assert_eq!(visited, 1);
    }

    #[test]
    fn generational_indices_detect_stale_handles() {
        let mut world = World::new();
        let e1 = world.spawn();
        world.insert(e1, Health(10));
        assert!(world.despawn(e1));

        // The freed slot is recycled, but with a new generation.
        let e2 = world.spawn();
        assert_eq!(e1.index(), e2.index());
        assert_ne!(e1.generation(), e2.generation());

        // The stale handle must not resolve, even after the slot is reused.
        assert!(!world.is_alive(e1));
        assert!(world.get::<Health>(e1).is_none());
        world.insert(e2, Health(99));
        assert!(world.get::<Health>(e1).is_none());
        assert_eq!(world.get::<Health>(e2).unwrap().0, 99);
    }

    #[test]
    fn despawn_clears_all_components() {
        let mut world = World::new();
        let e = world.spawn();
        world.insert(e, Position { x: 0.0, y: 0.0 });
        world.insert(e, Health(5));
        assert!(world.despawn(e));
        assert!(world.get::<Position>(e).is_none());
        assert!(world.get::<Health>(e).is_none());
        assert!(!world.despawn(e)); // already gone
    }

    #[test]
    fn insert_on_stale_handle_is_ignored() {
        let mut world = World::new();
        let e1 = world.spawn();
        assert!(world.despawn(e1));

        // The slot is recycled with a fresh generation.
        let e2 = world.spawn();
        assert_eq!(e1.index(), e2.index());

        // Writing through the stale handle must be a no-op, not a zombie that
        // outlives every future despawn of this slot.
        assert!(world.insert(e1, Health(1)).is_none());
        assert!(!world.has::<Health>(e1));
        assert!(!world.has::<Health>(e2));

        // The live entity is unaffected and behaves normally.
        world.insert(e2, Health(7));
        assert_eq!(world.get::<Health>(e2).unwrap().0, 7);
        assert!(!world.has::<Health>(e1));
    }

    #[test]
    fn resources_round_trip() {
        let mut world = World::new();
        world.insert_resource(DeltaTime(1.0));
        world.resource_mut::<DeltaTime>().0 = 2.0;
        assert_eq!(world.resource::<DeltaTime>().0, 2.0);
        assert_eq!(world.remove_resource::<DeltaTime>().unwrap().0, 2.0);
        assert!(world.get_resource::<DeltaTime>().is_none());
    }

    #[test]
    fn entity_builder_attaches_all_components() {
        let mut world = World::new();
        let e = world
            .spawn_entity()
            .with(Position { x: 1.0, y: 2.0 })
            .with(Velocity { x: 3.0, y: 4.0 })
            .with(Frozen)
            .id();

        assert_eq!(world.get::<Position>(e).unwrap().x, 1.0);
        assert_eq!(world.get::<Velocity>(e).unwrap().y, 4.0);
        assert!(world.has::<Frozen>(e));
    }

    #[test]
    fn entities_lists_only_live_entities() {
        let mut world = World::new();
        let a = world.spawn();
        let b = world.spawn();
        let c = world.spawn();
        assert!(world.despawn(b));

        // The freed slot `b` is gone; a recycled slot reappears once reused.
        let mut live: Vec<_> = world.entities().collect();
        live.sort_by_key(|e| e.index());
        assert_eq!(live, vec![a, c]);

        let d = world.spawn(); // recycles b's index with a fresh generation
        assert_eq!(d.index(), b.index());
        assert_ne!(d.generation(), b.generation());
        assert!(world.entities().any(|e| e == d));
        assert!(!world.entities().any(|e| e == b));
        assert_eq!(world.entities().count(), 3);
    }
}
