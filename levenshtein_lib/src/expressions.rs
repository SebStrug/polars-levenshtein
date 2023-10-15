use polars::prelude::*;
use pyo3_polars::{derive::polars_expr, export::polars_core::export::num::ToPrimitive};
use std::cmp::max;
use std::cmp::min;

fn levenshtein_distance(str_a: &str, str_b: &str) -> i32 {
    let len_a = str_a.len();
    let len_b = str_b.len();

    // Consider empty string as a starting point, so grid is one larger than both strings
    let mut vec = vec![vec![0; len_b + 1]; len_a + 1];

    for i in 0..=len_a {
        vec[i][0] = i as i32;
    }

    for j in 0..=len_b {
        vec[0][j] = j as i32;
    }

    for i in 1..=len_a {
        for j in 1..=len_b {
            // using `.nth` makes this a naive implementation since it always starts from 0th index
            let cost = if str_a.chars().nth(i - 1).unwrap() == str_b.chars().nth(j - 1).unwrap() {
                0
            } else {
                1
            };
            vec[i][j] = min(
                vec[i - 1][j] + 1,
                min(vec[i][j - 1] + 1, vec[i - 1][j - 1] + cost),
            );
        }
    }
    return vec[len_a][len_b];
}

fn levenshtein_ratio(str_a: &str, str_b: &str) -> f64 {
    let levenshtein_dist = levenshtein_distance(str_a, str_b);
    let greatest_str = max(str_a.len(), str_b.len()).to_f64().unwrap();
    if greatest_str == 0.0 {
        return 1.0;
    }
    return 1.0 - levenshtein_dist.to_f64().unwrap() / greatest_str;
}

#[polars_expr(output_type=Float64)]
fn get_levenshtein_ratio(inputs: &[Series]) -> PolarsResult<Series> {
    // 0th index is the column operated on
    let a = inputs[0].utf8()?;
    // 1 index is the other column passed
    let b = inputs[1].utf8()?;

    let out: Float64Chunked = arity::binary_elementwise_values(a, b, levenshtein_ratio);
    Ok(out.into_series())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("apple", "apple"), 0);
        assert_eq!(levenshtein_distance("Apple", "apple"), 1);
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("", "apple"), 5);
        assert_eq!(levenshtein_distance("apple", "aple"), 1);
        assert_eq!(levenshtein_distance("apple", "appl"), 1);
        assert_eq!(levenshtein_distance("apple", "applle"), 1);
        assert_eq!(levenshtein_distance("apple", "bpple"), 1);
        assert_eq!(levenshtein_distance("apple", "elppa"), 4);
    }

    #[test]
    fn test_levenshtein_ratio() {
        assert_eq!(levenshtein_ratio("apple", "apple"), 1.0);
        assert_eq!(levenshtein_ratio("Apple", "apple"), 0.8);
        assert_eq!(levenshtein_ratio("", ""), 1.0);
        assert_eq!(levenshtein_ratio("", "apple"), 0.0);
    }
}
