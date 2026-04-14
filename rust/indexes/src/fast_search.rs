/// Optimized layer-0 search using co-located storage.
///
/// This is the hot path — it reads neighbor lists and vectors from the same
/// cache lines, uses batch-4 distance computation, and has full prefetch coverage.
///
/// Used by FastHNSW when a ColocatedStore is available.

use crate::distance::{compute_distance, cosine_distance_batch4, prefetch_vector, DistanceMetric};
use crate::storage::ColocatedStore;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::fast_hnsw::{Candidate, MinCand, MaxCand, VisitedList};

/// Search layer 0 using co-located storage. This is the fastest path.
pub fn search_layer0_colocated(
    store: &ColocatedStore,
    alive: &[bool],
    visited: &mut VisitedList,
    query: &[f32],
    entry_id: usize,
    ef: usize,
    metric: DistanceMetric,
) -> Vec<Candidate> {
    let entry_dist = compute_distance(query, store.get_vector(entry_id), metric);

    let mut candidates = BinaryHeap::new();
    candidates.push(MinCand(Candidate { id: entry_id, dist: entry_dist }));

    let mut results = BinaryHeap::new();
    results.push(MaxCand(Candidate { id: entry_id, dist: entry_dist }));

    while let Some(MinCand(current)) = candidates.pop() {
        let worst_dist = results.peek().unwrap().0.dist;

        let neighbors = store.get_neighbors(current.id);

        // Batch-4 distance computation (FAISS technique):
        // Collect up to 4 unvisited neighbors, compute distances simultaneously.
        let mut batch_ids: [usize; 4] = [0; 4];
        let mut batch_count = 0;

        for (ni, &nid_u32) in neighbors.iter().enumerate() {
            let nid = nid_u32 as usize;

            if visited.is_visited(nid) || !alive[nid] {
                continue;
            }
            visited.mark_visited(nid);

            // Prefetch the next unvisited neighbor's co-located data
            // (gets both neighbor list AND vector in same cache lines)
            if ni + 1 < neighbors.len() {
                let next = neighbors[ni + 1] as usize;
                if !visited.is_visited(next) {
                    let ptr = store.node_ptr(next);
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        #[cfg(target_feature = "sse")]
                        {
                            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
                            // Also prefetch the vector portion (may be in next cache line)
                            std::arch::x86_64::_mm_prefetch(ptr.add(64) as *const i8, std::arch::x86_64::_MM_HINT_T0);
                        }
                    }
                }
            }

            batch_ids[batch_count] = nid;
            batch_count += 1;

            if batch_count == 4 {
                // Compute 4 distances at once
                let v0 = store.get_vector(batch_ids[0]);
                let v1 = store.get_vector(batch_ids[1]);
                let v2 = store.get_vector(batch_ids[2]);
                let v3 = store.get_vector(batch_ids[3]);

                let (d0, d1, d2, d3) = if metric == DistanceMetric::Cosine || metric == DistanceMetric::InnerProduct {
                    cosine_distance_batch4(query, v0, v1, v2, v3)
                } else {
                    // Fall back to individual for non-dot-product metrics
                    (
                        compute_distance(query, v0, metric),
                        compute_distance(query, v1, metric),
                        compute_distance(query, v2, metric),
                        compute_distance(query, v3, metric),
                    )
                };

                let dists = [d0, d1, d2, d3];
                for i in 0..4 {
                    if dists[i] < worst_dist || results.len() < ef {
                        candidates.push(MinCand(Candidate { id: batch_ids[i], dist: dists[i] }));
                        results.push(MaxCand(Candidate { id: batch_ids[i], dist: dists[i] }));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
                batch_count = 0;
            }
        }

        // Handle remainder (< 4 neighbors)
        for i in 0..batch_count {
            let nid = batch_ids[i];
            let dist = compute_distance(query, store.get_vector(nid), metric);
            if dist < worst_dist || results.len() < ef {
                candidates.push(MinCand(Candidate { id: nid, dist }));
                results.push(MaxCand(Candidate { id: nid, dist }));
                if results.len() > ef {
                    results.pop();
                }
            }
        }

        // Termination check
        if let Some(next) = candidates.peek() {
            if next.0.dist > results.peek().unwrap().0.dist && results.len() >= ef {
                break;
            }
        }
    }

    results.into_iter().map(|MaxCand(c)| c).collect()
}
