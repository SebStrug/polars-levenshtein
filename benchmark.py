"""
"Benchmark" the Rust interop function against a naive implementation and what we currently
do in swim-free: using the rapidfuzz library through the `.map_elements` method.

Rapidfuzz is a set of highly-optimized implementations written in C++, so this is a tough
comparison for our naive Rust code!

`.map_elements` is similar to `.apply` in Pandas.
 """

import time
from math import log10

import matplotlib.pyplot as plt
import polars as pl
from polars.testing import assert_frame_equal
from rapidfuzz.distance.Levenshtein import normalized_distance

from levenshtein_lib import LevenshteinDistance

single_df = pl.DataFrame(
    {
        "places": ["Kingston upon Hull", "Dull", "Sheffield", "Leeds", "Doncaster"],
    }
)

df_rows_tests = (5 * 10**2, 5*10**3, 5*10**4, 5*10**5, 5*10**6, 1*10**7)
rust_times = []
rapidfuzz_times = []

for df_rows in df_rows_tests:
    rows = df_rows // 5
    df = pl.concat([single_df] * rows)

    start_rust = time.time()
    rust_df = df.with_columns(comparison=pl.lit("hull")).with_columns(
        user_input=pl.col("places").dist.get_levenshtein_ratio("comparison")
    )

    time_rust = round(time.time() - start_rust, 2)
    rust_times.append(time_rust)
    print(f"Time taken for interop function, {df_rows:,} rows: {time_rust}")

    start_rapidfuzz = time.time()
    rapidfuzz_df = df.with_columns(comparison=pl.lit("hull")).with_columns(
        user_input=pl.struct(("places", "comparison")).map_elements(
            lambda row: 1 - normalized_distance(row["places"], row["comparison"])
        )
    )
    time_rapidfuzz = round(time.time() - start_rapidfuzz, 2)
    rapidfuzz_times.append(time_rapidfuzz)
    print(f"Time taken for rapidfuzz, {df_rows:,} rows: {time_rapidfuzz}")

    assert_frame_equal(rust_df, rapidfuzz_df)

plt.plot([log10(r) for r in df_rows_tests], rust_times, label="Rust interop")
plt.plot(
    [log10(r) for r in df_rows_tests], rapidfuzz_times, label="Using `.map_elements`"
)
plt.xlabel("Log(10) of the number of dataframe rows")
plt.ylabel("Number of seconds to compute levenshtein ratio")
plt.legend()
plt.show()
