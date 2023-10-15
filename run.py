import polars as pl
from levenshtein_lib import LevenshteinDistance

df = pl.DataFrame({
    "places": ["Kingston upon Hull", "Dull", "Sheffield", "Leeds", "Doncaster"],
})


df = df.with_columns(
        comparison=pl.lit("hull")
    ).with_columns(
        user_input=pl.col("places").dist.get_levenshtein_ratio("comparison")
    )

print(df)