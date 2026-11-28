//! Synchronous observer primitives (`Signal`, `ScopedSignal`, `ObservableTable`, `RefCountedSet`)
//! for keeping derived runtime and backend caches coherent with resource-table mutations.
//!
//! Unlike [`crate::signal`], which queues asynchronous device/swapchain notifications for
//! client polling via [`crate::Context::poll_signals`], this module wires internal state
//! tables directly to derived caches (residency lists, descriptor/argument tables, view caches)
//! so mutations patch or invalidate subscribers at the point of change without global
//! generation counters or full-table scans.
#![cfg_attr(
    not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))),
    allow(dead_code)
)]

use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// RAII handle for a connected observer slot.
///
/// Dropping a [`Subscription`] marks its slot inactive with a single atomic store, so
/// disconnecting never contends on or deadlocks with an in-progress [`Signal::emit`].
/// Call [`Subscription::detach`] when the slot's lifetime is already governed by a
/// [`Weak`] target registered through [`Signal::connect_weak`] or
/// [`ScopedSignal::connect_weak`].
#[must_use = "dropping a Subscription immediately disconnects the slot; call .detach() to keep a weak slot alive"]
pub(crate) struct Subscription {
    alive: Option<Arc<AtomicBool>>,
}

impl Subscription {
    fn new(alive: Arc<AtomicBool>) -> Self {
        Self { alive: Some(alive) }
    }

    /// Leave the slot connected after dropping this handle.
    ///
    /// Intended for `connect_weak` registrations where the slot automatically prunes
    /// itself once the weak target `Arc` is dropped.
    #[allow(dead_code)]
    pub(crate) fn detach(mut self) {
        self.alive = None;
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        if let Some(alive) = self.alive.take() {
            alive.store(false, Ordering::Release);
        }
    }
}

struct SlotEntry<E: ?Sized> {
    alive: Arc<AtomicBool>,
    /// Returns `true` to stay registered, `false` to prune (e.g. dead `Weak`).
    callback: Box<dyn FnMut(&E) -> bool + Send>,
}

fn retain_and_emit<E: ?Sized>(slots: &mut Vec<SlotEntry<E>>, event: &E) {
    slots.retain_mut(|entry| {
        if !entry.alive.load(Ordering::Acquire) {
            return false;
        }
        let keep = (entry.callback)(event);
        keep && entry.alive.load(Ordering::Acquire)
    });
}

/// Synchronous broadcast hub that fans events `E` out to connected slots.
pub(crate) struct Signal<E: ?Sized> {
    slots: Mutex<Vec<SlotEntry<E>>>,
}

impl<E: ?Sized> Default for Signal<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E: ?Sized> Signal<E> {
    pub(crate) fn new() -> Self {
        Self {
            slots: Mutex::new(Vec::new()),
        }
    }

    /// Connect a stateful closure slot. Disconnected when the returned [`Subscription`] drops.
    #[allow(dead_code)]
    pub(crate) fn connect(&self, mut slot: impl FnMut(&E) + Send + 'static) -> Subscription {
        let alive = Arc::new(AtomicBool::new(true));
        let entry = SlotEntry {
            alive: Arc::clone(&alive),
            callback: Box::new(move |event| {
                slot(event);
                true
            }),
        };
        self.slots.lock().unwrap().push(entry);
        Subscription::new(alive)
    }

    /// Connect a slot targeting an [`Arc`]-managed subscriber via [`Weak`].
    ///
    /// Automatically prunes itself on the first [`Signal::emit`] after `target` is dropped,
    /// and also disconnects early if the returned [`Subscription`] is dropped.
    pub(crate) fn connect_weak<T, F>(&self, target: &Arc<T>, mut slot: F) -> Subscription
    where
        T: Send + Sync + 'static,
        F: FnMut(&T, &E) + Send + 'static,
    {
        let weak: Weak<T> = Arc::downgrade(target);
        let alive = Arc::new(AtomicBool::new(true));
        let entry = SlotEntry {
            alive: Arc::clone(&alive),
            callback: Box::new(move |event| {
                let Some(strong) = weak.upgrade() else {
                    return false;
                };
                slot(&strong, event);
                true
            }),
        };
        self.slots.lock().unwrap().push(entry);
        Subscription::new(alive)
    }

    /// Deliver `event` to every live slot, pruning disconnected or expired entries in-place.
    pub(crate) fn emit(&self, event: &E) {
        let mut slots = self.slots.lock().unwrap();
        retain_and_emit(&mut slots, event);
    }

    #[cfg(test)]
    pub(crate) fn slot_count(&self) -> usize {
        let mut slots = self.slots.lock().unwrap();
        slots.retain(|entry| entry.alive.load(Ordering::Acquire));
        slots.len()
    }
}

/// Partitioned observer hub keyed by scope `S` (such as a logical device or context handle).
///
/// Emitting an event on scope `s` performs an `O(1)` lookup and invokes only slots
/// registered for `s`, avoiding cross-scope wakeups or spurious invalidations.
#[allow(dead_code)]
pub(crate) struct ScopedSignal<S, E: ?Sized> {
    buckets: Mutex<HashMap<S, Vec<SlotEntry<E>>>>,
}

impl<S: Eq + Hash, E: ?Sized> Default for ScopedSignal<S, E> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
impl<S: Eq + Hash, E: ?Sized> ScopedSignal<S, E> {
    pub(crate) fn new() -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
        }
    }

    /// Connect a slot that receives only events emitted for `scope`.
    pub(crate) fn connect(&self, scope: S, mut slot: impl FnMut(&E) + Send + 'static) -> Subscription {
        let alive = Arc::new(AtomicBool::new(true));
        let entry = SlotEntry {
            alive: Arc::clone(&alive),
            callback: Box::new(move |event| {
                slot(event);
                true
            }),
        };
        self.buckets.lock().unwrap().entry(scope).or_default().push(entry);
        Subscription::new(alive)
    }

    /// Connect a [`Weak`]-targeted slot that receives only events emitted for `scope`.
    pub(crate) fn connect_weak<T, F>(&self, scope: S, target: &Arc<T>, mut slot: F) -> Subscription
    where
        T: Send + Sync + 'static,
        F: FnMut(&T, &E) + Send + 'static,
    {
        let weak: Weak<T> = Arc::downgrade(target);
        let alive = Arc::new(AtomicBool::new(true));
        let entry = SlotEntry {
            alive: Arc::clone(&alive),
            callback: Box::new(move |event| {
                let Some(strong) = weak.upgrade() else {
                    return false;
                };
                slot(&strong, event);
                true
            }),
        };
        self.buckets.lock().unwrap().entry(scope).or_default().push(entry);
        Subscription::new(alive)
    }

    /// Deliver `event` only to slots subscribed to `scope`.
    pub(crate) fn emit(&self, scope: &S, event: &E) {
        let mut buckets = self.buckets.lock().unwrap();
        let empty = if let Some(slots) = buckets.get_mut(scope) {
            retain_and_emit(slots, event);
            slots.is_empty()
        } else {
            false
        };
        if empty {
            buckets.remove(scope);
        }
    }

    /// Drop all slot registrations under `scope`.
    pub(crate) fn remove_scope(&self, scope: &S) {
        self.buckets.lock().unwrap().remove(scope);
    }
}

/// Mutation payload emitted by [`ObservableTable`].
#[derive(Debug)]
pub(crate) enum TableEvent<'a, K, V> {
    Inserted {
        #[allow(dead_code)]
        key: &'a K,
        value: &'a V,
    },
    Replaced {
        #[allow(dead_code)]
        key: &'a K,
        old: &'a V,
        new: &'a V,
    },
    Removed {
        #[allow(dead_code)]
        key: &'a K,
        value: &'a V,
    },
}

trait TableObserver<K, V>: Send {
    fn on_event(&mut self, event: TableEvent<'_, K, V>) -> bool;
}

struct TableSlotEntry<K, V> {
    alive: Arc<AtomicBool>,
    observer: Box<dyn TableObserver<K, V>>,
}

struct ClosureTableObserver<F>(F);

impl<K, V, F> TableObserver<K, V> for ClosureTableObserver<F>
where
    F: for<'a> FnMut(TableEvent<'a, K, V>) + Send,
{
    fn on_event(&mut self, event: TableEvent<'_, K, V>) -> bool {
        (self.0)(event);
        true
    }
}

struct WeakTableObserver<T, F> {
    weak: Weak<T>,
    callback: F,
}

impl<K, V, T, F> TableObserver<K, V> for WeakTableObserver<T, F>
where
    T: Send + Sync + 'static,
    F: for<'a> FnMut(&T, TableEvent<'a, K, V>) + Send,
{
    fn on_event(&mut self, event: TableEvent<'_, K, V>) -> bool {
        let Some(strong) = self.weak.upgrade() else {
            return false;
        };
        (self.callback)(&strong, event);
        true
    }
}

fn retain_and_emit_table<K, V>(slots: &mut Vec<TableSlotEntry<K, V>>, event: &FnEmit<'_, K, V>) {
    slots.retain_mut(|entry| {
        if !entry.alive.load(Ordering::Acquire) {
            return false;
        }
        let keep = entry.observer.on_event(event.reborrow());
        keep && entry.alive.load(Ordering::Acquire)
    });
}

enum FnEmit<'a, K, V> {
    Inserted(&'a K, &'a V),
    Replaced(&'a K, &'a V, &'a V),
    Removed(&'a K, &'a V),
}

impl<'a, K, V> FnEmit<'a, K, V> {
    fn reborrow(&self) -> TableEvent<'a, K, V> {
        match *self {
            Self::Inserted(key, value) => TableEvent::Inserted { key, value },
            Self::Replaced(key, old, new) => TableEvent::Replaced { key, old, new },
            Self::Removed(key, value) => TableEvent::Removed { key, value },
        }
    }
}

/// Key-value resource table that dereferences to [`HashMap<K, V>`] for zero-overhead reads
/// while routing every mutation (`insert`, `replace_with`, `remove`, `retain`) through
/// scoped and unscoped observer slots.
///
/// Intentionally omits [`std::ops::DerefMut`] and `get_mut`: callers cannot mutate an entry
/// in place without emitting a [`TableEvent`].
pub(crate) struct ObservableTable<K, V, S = ()> {
    inner: HashMap<K, V>,
    scope_of: fn(&V) -> S,
    unscoped_slots: Mutex<Vec<TableSlotEntry<K, V>>>,
    scoped_slots: Mutex<HashMap<S, Vec<TableSlotEntry<K, V>>>>,
}

impl<K, V, S> Deref for ObservableTable<K, V, S> {
    type Target = HashMap<K, V>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<K: Eq + Hash + Clone, V> ObservableTable<K, V, ()> {
    #[allow(dead_code)]
    pub(crate) fn new() -> Self {
        Self::with_scope(|_| ())
    }
}

impl<K: Eq + Hash + Clone, V, S: Eq + Hash> ObservableTable<K, V, S> {
    pub(crate) fn with_scope(scope_of: fn(&V) -> S) -> Self {
        Self {
            inner: HashMap::new(),
            scope_of,
            unscoped_slots: Mutex::new(Vec::new()),
            scoped_slots: Mutex::new(HashMap::new()),
        }
    }

    /// Subscribe to mutations across all scopes in this table.
    #[allow(dead_code)]
    pub(crate) fn connect<F>(&self, slot: F) -> Subscription
    where
        F: for<'a> FnMut(TableEvent<'a, K, V>) + Send + 'static,
    {
        let alive = Arc::new(AtomicBool::new(true));
        self.unscoped_slots.lock().unwrap().push(TableSlotEntry {
            alive: Arc::clone(&alive),
            observer: Box::new(ClosureTableObserver(slot)),
        });
        Subscription::new(alive)
    }

    /// Subscribe to mutations for a single scope `S` (e.g. a `DeviceHandle`).
    #[allow(dead_code)]
    pub(crate) fn connect_scoped<F>(&self, scope: S, slot: F) -> Subscription
    where
        F: for<'a> FnMut(TableEvent<'a, K, V>) + Send + 'static,
    {
        let alive = Arc::new(AtomicBool::new(true));
        self.scoped_slots
            .lock()
            .unwrap()
            .entry(scope)
            .or_default()
            .push(TableSlotEntry {
                alive: Arc::clone(&alive),
                observer: Box::new(ClosureTableObserver(slot)),
            });
        Subscription::new(alive)
    }

    /// Subscribe a [`Weak`]-targeted slot to mutations for a single scope `S`.
    pub(crate) fn connect_scoped_weak<T, F>(&self, scope: S, target: &Arc<T>, slot: F) -> Subscription
    where
        T: Send + Sync + 'static,
        F: for<'a> FnMut(&T, TableEvent<'a, K, V>) + Send + 'static,
    {
        let alive = Arc::new(AtomicBool::new(true));
        self.scoped_slots
            .lock()
            .unwrap()
            .entry(scope)
            .or_default()
            .push(TableSlotEntry {
                alive: Arc::clone(&alive),
                observer: Box::new(WeakTableObserver {
                    weak: Arc::downgrade(target),
                    callback: slot,
                }),
            });
        Subscription::new(alive)
    }

    /// Remove all scoped slot registrations for `scope`.
    pub(crate) fn remove_scope(&self, scope: &S) {
        self.scoped_slots.lock().unwrap().remove(scope);
    }

    fn emit_for_scope(&self, scope: &S, event: &FnEmit<'_, K, V>) {
        {
            let mut unscoped = self.unscoped_slots.lock().unwrap();
            if !unscoped.is_empty() {
                retain_and_emit_table(&mut unscoped, event);
            }
        }
        let mut scoped = self.scoped_slots.lock().unwrap();
        let empty = if let Some(slots) = scoped.get_mut(scope) {
            retain_and_emit_table(slots, event);
            slots.is_empty()
        } else {
            false
        };
        if empty {
            scoped.remove(scope);
        }
    }

    /// Insert `value` under `key`, emitting [`TableEvent::Inserted`] or [`TableEvent::Replaced`].
    pub(crate) fn insert(&mut self, key: K, value: V) -> Option<V> {
        let old = self.inner.insert(key.clone(), value);
        let new_ref = self.inner.get(&key).expect("just inserted");
        let new_scope = (self.scope_of)(new_ref);
        match &old {
            Some(old_val) => {
                let old_scope = (self.scope_of)(old_val);
                if old_scope == new_scope {
                    self.emit_for_scope(&new_scope, &FnEmit::Replaced(&key, old_val, new_ref));
                } else {
                    self.emit_for_scope(&old_scope, &FnEmit::Removed(&key, old_val));
                    self.emit_for_scope(&new_scope, &FnEmit::Inserted(&key, new_ref));
                }
            }
            None => {
                self.emit_for_scope(&new_scope, &FnEmit::Inserted(&key, new_ref));
            }
        }
        old
    }

    /// Transform an existing entry by value and emit [`TableEvent::Replaced`].
    ///
    /// Returns `true` if `key` existed and was updated.
    pub(crate) fn replace_with(&mut self, key: &K, f: impl FnOnce(&V) -> V) -> bool {
        let Some(current) = self.inner.get(key) else {
            return false;
        };
        let next = f(current);
        let _ = self.insert(key.clone(), next);
        true
    }

    /// Remove `key` if present, emitting [`TableEvent::Removed`] before returning the value.
    pub(crate) fn remove(&mut self, key: &K) -> Option<V> {
        let removed = self.inner.remove(key)?;
        let scope = (self.scope_of)(&removed);
        self.emit_for_scope(&scope, &FnEmit::Removed(key, &removed));
        Some(removed)
    }

    /// Retain only entries satisfying `keep`, emitting [`TableEvent::Removed`] for each dropped entry.
    pub(crate) fn retain(&mut self, mut keep: impl FnMut(&K, &V) -> bool) {
        let doomed: Vec<K> = self
            .inner
            .iter()
            .filter(|(k, v)| !keep(k, v))
            .map(|(k, _)| k.clone())
            .collect();
        for key in doomed {
            let _ = self.remove(&key);
        }
    }
}

/// Incremental reference-counted unique set backed by a dense slice (`O(1)` insert/remove/replace).
///
/// Designed for derived backend caches where multiple logical handles (such as buffer views
/// slicing a shared slab) alias the same underlying native resource:
/// - Inserting an already-present item increments its reference count in `O(1)` and returns
///   `false` (the dense unique slice is unchanged).
/// - Removing an item with remaining references decrements its reference count in `O(1)` and
///   returns `false`.
/// - Only transitions across `0 <-> 1` mutate the dense slice (`push` / `swap_remove`, `O(1)`)
///   and return `true`.
#[derive(Debug, Clone)]
pub(crate) struct RefCountedSet<T> {
    /// Maps each unique item to `(refcount, dense_index)`.
    index: HashMap<T, (u32, usize)>,
    dense: Vec<T>,
}

impl<T: Eq + Hash + Copy> Default for RefCountedSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Eq + Hash + Copy> RefCountedSet<T> {
    pub(crate) fn new() -> Self {
        Self {
            index: HashMap::new(),
            dense: Vec::new(),
        }
    }

    /// Add one reference to `item`. Returns `true` iff `item` was newly added to `as_slice()`.
    pub(crate) fn insert(&mut self, item: T) -> bool {
        if let Some((count, _)) = self.index.get_mut(&item) {
            *count = count.checked_add(1).expect("RefCountedSet reference count overflow");
            return false;
        }
        let dense_idx = self.dense.len();
        self.dense.push(item);
        self.index.insert(item, (1, dense_idx));
        true
    }

    /// Release one reference to `item`. Returns `true` iff the last reference was removed.
    pub(crate) fn remove(&mut self, item: T) -> bool {
        let Some((count, dense_idx)) = self.index.get_mut(&item) else {
            return false;
        };
        if *count > 1 {
            *count -= 1;
            return false;
        }
        let removed_idx = *dense_idx;
        self.index.remove(&item);
        self.dense.swap_remove(removed_idx);
        if let Some(&moved) = self.dense.get(removed_idx) {
            if let Some((_, moved_idx)) = self.index.get_mut(&moved) {
                *moved_idx = removed_idx;
            }
        }
        true
    }

    /// Replace one reference to `old` with `new`. Returns `true` iff `as_slice()` changed.
    #[allow(dead_code)]
    pub(crate) fn replace(&mut self, old: Option<T>, new: Option<T>) -> bool {
        if old == new {
            return false;
        }
        let mut changed = false;
        if let Some(prev) = old {
            changed |= self.remove(prev);
        }
        if let Some(next) = new {
            changed |= self.insert(next);
        }
        changed
    }

    /// Contiguous unique items currently held with `refcount >= 1`.
    #[inline]
    pub(crate) fn as_slice(&self) -> &[T] {
        &self.dense
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.dense.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn signal_connect_emit_and_drop_disconnect() {
        let sig = Signal::<u32>::new();
        let sum = Arc::new(AtomicUsize::new(0));
        let sum_slot = Arc::clone(&sum);
        let sub = sig.connect(move |v| {
            sum_slot.fetch_add(*v as usize, Ordering::SeqCst);
        });

        sig.emit(&3);
        sig.emit(&4);
        assert_eq!(sum.load(Ordering::SeqCst), 7);
        assert_eq!(sig.slot_count(), 1);

        drop(sub);
        sig.emit(&10);
        assert_eq!(sum.load(Ordering::SeqCst), 7);
        assert_eq!(sig.slot_count(), 0);
    }

    #[test]
    fn subscription_drop_inside_emit_does_not_deadlock() {
        let sig = Signal::<u32>::new();
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_slot = Arc::clone(&hits);
        let holder: Arc<Mutex<Option<Subscription>>> = Arc::new(Mutex::new(None));
        let holder_slot = Arc::clone(&holder);

        let sub = sig.connect(move |_| {
            hits_slot.fetch_add(1, Ordering::SeqCst);
            let _ = holder_slot.lock().unwrap().take();
        });
        *holder.lock().unwrap() = Some(sub);

        sig.emit(&1);
        sig.emit(&2);
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        assert_eq!(sig.slot_count(), 0);
    }

    #[test]
    fn weak_slot_auto_prunes_when_subscriber_drops() {
        let sig = Signal::<u32>::new();
        let target = Arc::new(AtomicUsize::new(0));
        sig.connect_weak(&target, |t, v| {
            t.fetch_add(*v as usize, Ordering::SeqCst);
        })
        .detach();

        sig.emit(&5);
        assert_eq!(target.load(Ordering::SeqCst), 5);
        assert_eq!(sig.slot_count(), 1);

        drop(target);
        sig.emit(&5);
        assert_eq!(sig.slot_count(), 0);
    }

    #[test]
    fn observable_table_routes_events_only_to_matching_scope() {
        #[derive(Clone, Debug)]
        struct Entry {
            device: u32,
            backing: usize,
        }

        let mut table = ObservableTable::<u64, Entry, u32>::with_scope(|e| e.device);
        let dev1_events = Arc::new(AtomicUsize::new(0));
        let dev2_events = Arc::new(AtomicUsize::new(0));

        let _s1 = {
            let c = Arc::clone(&dev1_events);
            table.connect_scoped(1, move |_| {
                c.fetch_add(1, Ordering::SeqCst);
            })
        };
        let _s2 = {
            let c = Arc::clone(&dev2_events);
            table.connect_scoped(2, move |_| {
                c.fetch_add(1, Ordering::SeqCst);
            })
        };

        table.insert(
            10,
            Entry {
                device: 1,
                backing: 0x100,
            },
        );
        table.insert(
            20,
            Entry {
                device: 2,
                backing: 0x200,
            },
        );
        table.insert(
            21,
            Entry {
                device: 2,
                backing: 0x201,
            },
        );
        table.remove(&20);

        assert_eq!(
            dev1_events.load(Ordering::SeqCst),
            1,
            "device 1 must not see device 2 churn"
        );
        assert_eq!(dev2_events.load(Ordering::SeqCst), 3);

        table.retain(|_, e| e.device != 2);
        assert_eq!(dev1_events.load(Ordering::SeqCst), 1);
        assert_eq!(dev2_events.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn refcounted_set_absorbs_shared_backing_views_and_patches_swap_remove() {
        let mut set = RefCountedSet::<usize>::new();

        // Root slab + two sub-views sharing 0xA000.
        assert!(set.insert(0xA000));
        assert!(
            !set.insert(0xA000),
            "sub-view on resident backing must not mutate dense set"
        );
        assert!(!set.insert(0xA000));
        assert_eq!(set.as_slice(), &[0xA000]);

        // Add two more distinct resources so swap_remove exercises index fixup.
        assert!(set.insert(0xB000));
        assert!(set.insert(0xC000));
        assert_eq!(set.len(), 3);

        // Drop two sub-views of 0xA000: backing remains live, dense slice untouched.
        assert!(!set.remove(0xA000));
        assert!(!set.remove(0xA000));
        assert_eq!(set.len(), 3);

        // Drop final reference to 0xA000 (index 0 -> swap_remove moves 0xC000 into slot 0).
        assert!(set.remove(0xA000));
        assert_eq!(set.as_slice(), &[0xC000, 0xB000]);

        // Removing 0xC000 after swap_remove must still find its updated index cleanly.
        assert!(set.remove(0xC000));
        assert_eq!(set.as_slice(), &[0xB000]);
        assert!(
            !set.replace(Some(0xB000), Some(0xB000)),
            "same-backing replace is a no-op"
        );
        assert!(set.replace(Some(0xB000), Some(0xD000)));
        assert_eq!(set.as_slice(), &[0xD000]);
        assert!(set.remove(0xD000));
        assert!(set.as_slice().is_empty());
    }

    #[test]
    fn two_level_table_to_cache_to_context_avoids_spurious_invalidation() {
        #[derive(Clone)]
        struct Buf {
            device: u32,
            ptr: usize,
            staging_only: bool,
        }

        struct DerivedCache {
            ptrs: Mutex<RefCountedSet<usize>>,
            changed: Signal<()>,
        }

        let mut buffers = ObservableTable::<u64, Buf, u32>::with_scope(|b| b.device);
        let dev1_cache = Arc::new(DerivedCache {
            ptrs: Mutex::new(RefCountedSet::new()),
            changed: Signal::new(),
        });

        let _table_sub = buffers.connect_scoped_weak(1, &dev1_cache, |cache, ev| {
            let ptr_of = |b: &Buf| (!b.staging_only).then_some(b.ptr);
            let changed = {
                let mut set = cache.ptrs.lock().unwrap();
                match ev {
                    TableEvent::Inserted { value, .. } => ptr_of(value).is_some_and(|p| set.insert(p)),
                    TableEvent::Removed { value, .. } => ptr_of(value).is_some_and(|p| set.remove(p)),
                    TableEvent::Replaced { old, new, .. } => set.replace(ptr_of(old), ptr_of(new)),
                }
            };
            if changed {
                cache.changed.emit(&());
            }
        });

        let ctx_a_dirty = Arc::new(AtomicBool::new(false));
        let ctx_b_dirty = Arc::new(AtomicBool::new(false));
        let _sub_a = dev1_cache
            .changed
            .connect_weak(&ctx_a_dirty, |dirty, _| dirty.store(true, Ordering::Release));
        let _sub_b = dev1_cache
            .changed
            .connect_weak(&ctx_b_dirty, |dirty, _| dirty.store(true, Ordering::Release));

        // 1. Insert root slab on device 1 -> both device-1 contexts signaled.
        buffers.insert(
            1,
            Buf {
                device: 1,
                ptr: 0x1000,
                staging_only: false,
            },
        );
        assert!(ctx_a_dirty.swap(false, Ordering::AcqRel));
        assert!(ctx_b_dirty.swap(false, Ordering::AcqRel));

        // 2. Churn a sub-view on the same slab (0x1000) -> unique resource set unchanged, ZERO invalidation.
        buffers.insert(
            2,
            Buf {
                device: 1,
                ptr: 0x1000,
                staging_only: false,
            },
        );
        buffers.remove(&2);
        assert!(!ctx_a_dirty.load(Ordering::Acquire));
        assert!(!ctx_b_dirty.load(Ordering::Acquire));

        // 3. Logical metadata update keeping same backing ptr -> ZERO invalidation.
        assert!(buffers.replace_with(&1, |b| b.clone()));
        assert!(!ctx_a_dirty.load(Ordering::Acquire));

        // 4. Withdraw-staging buffer churn on device 1 -> ZERO invalidation.
        buffers.insert(
            3,
            Buf {
                device: 1,
                ptr: 0x9000,
                staging_only: true,
            },
        );
        buffers.remove(&3);
        assert!(!ctx_a_dirty.load(Ordering::Acquire));

        // 5. Device 2 buffer churn -> ZERO invalidation on device 1.
        buffers.insert(
            4,
            Buf {
                device: 2,
                ptr: 0x2000,
                staging_only: false,
            },
        );
        assert!(!ctx_a_dirty.load(Ordering::Acquire));
        assert!(!ctx_b_dirty.load(Ordering::Acquire));
    }
}
