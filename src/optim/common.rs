//! Shared helpers for the optimization search objects.

/// Optimization threshold switch.
///
/// Returns `u0 > u1` when minimizing, `u0 < u1` otherwise.
pub(crate) fn threshold_switch(u0: f32, u1: f32, minimize: bool) -> bool {
    if minimize {
        u0 > u1
    } else {
        u0 < u1
    }
}

/// Sort values by index, returning the rank (index) array.
///
/// The input `values` are not reordered. Instead, the returned vector holds the
/// indices into `values` in sorted order.
///
/// # Arguments
/// * `values` - input values
/// * `descending` - sort in descending order when `true`, ascending otherwise
#[allow(dead_code)]
pub(crate) fn sort(values: &[f32], descending: bool) -> Vec<usize> {
    let len = values.len();

    let mut rank: Vec<usize> = (0..len).collect();

    for i in 0..len {
        for j in (i + 1..len).rev() {
            if threshold_switch(values[rank[j]], values[rank[j - 1]], descending) {
                // swap elements
                rank.swap(j, j - 1);
            }
        }
    }

    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_rank() {
        let values = [3.0f32, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0];

        // ascending: values[rank[k]] is non-decreasing
        let rank = sort(&values, false);
        assert_eq!(rank.len(), values.len());
        for w in rank.windows(2) {
            assert!(values[w[0]] <= values[w[1]]);
        }

        // descending: values[rank[k]] is non-increasing
        let rank = sort(&values, true);
        for w in rank.windows(2) {
            assert!(values[w[0]] >= values[w[1]]);
        }

        // the rank is a permutation of 0..len
        let mut sorted_rank = rank.clone();
        sorted_rank.sort_unstable();
        assert_eq!(sorted_rank, (0..values.len()).collect::<Vec<_>>());

        // empty input yields empty rank
        assert!(sort(&[], false).is_empty());
    }
}
