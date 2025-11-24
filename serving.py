from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count
import time
import builtins

spark = SparkSession.builder.appName("GoldLayerQueries").getOrCreate()

# Load SILVER parquet (directly, no encoding)
silver_path = "gs://medallion-dat535/gold/rdd/mental_health_clean.parquet"
df = spark.read.parquet(silver_path)

# ================= RDD VERSION =================
print("\n================= RDD VERSION =================\n")
start_rdd = time.time()

# Filter out rows with null IncreasingStress
rdd = df.rdd.filter(lambda row: row["IncreasingStress"] is not None)

# ---------- COUNTRY ----------
country_map = rdd.map(lambda r: (r["Country"], float(r["IncreasingStress"])))
country_grouped = country_map.groupByKey()

country_stats = country_grouped.mapValues(
    lambda values: {
        "avg_stress": builtins.sum(values) / len(values),
        "count": len(values)
    }
).collect()

print("RDD — Average Stress per Country:")
for country, stats in country_stats:
    print(country, stats)

# ---------- SOCIAL WEAKNESS ----------
sw_map = rdd.map(lambda r: (r["SocialWeakness"], float(r["IncreasingStress"])))
sw_grouped = sw_map.groupByKey()

sw_stats = sw_grouped.mapValues(
    lambda values: {
        "avg_stress": builtins.sum(values) / len(values),
        "count": len(values)
    }
).collect()

print("\nRDD — Stress per Social Weakness:")
for sw, stats in sw_stats:
    print(sw, stats)

print(f"\nRDD completed in {time.time() - start_rdd:.2f} seconds.")


# ================= DATAFRAME VERSION =================
print("\n================= DATAFRAME VERSION =================\n")
start_df = time.time()

df_country = df.groupBy("Country").agg(
    avg("IncreasingStress").alias("avg_stress"),
    count("IncreasingStress").alias("count")
).orderBy("avg_stress", ascending=False)

print("DataFrame — Average Stress per Country:")
df_country.show(truncate=False)

df_sw = df.groupBy("SocialWeakness").agg(
    avg("IncreasingStress").alias("avg_stress"),
    count("IncreasingStress").alias("count")
).orderBy("avg_stress", ascending=False)

print("DataFrame — Stress per Social Weakness:")
df_sw.show(truncate=False)

print(f"\nDataFrame completed in {time.time() - start_df:.2f} seconds.")


# ================= SQL VERSION =================
print("\n================= SQL VERSION =================\n")
start_sql = time.time()

df.createOrReplaceTempView("mh")

sql_country = spark.sql("""
    SELECT Country,
           AVG(IncreasingStress) AS avg_stress,
           COUNT(IncreasingStress) AS count
    FROM mh
    GROUP BY Country
    ORDER BY avg_stress DESC
""")

print("SQL — Average Stress per Country:")
sql_country.show(truncate=False)

sql_sw = spark.sql("""
    SELECT SocialWeakness,
           AVG(IncreasingStress) AS avg_stress,
           COUNT(IncreasingStress) AS count
    FROM mh
    GROUP BY SocialWeakness
    ORDER BY avg_stress DESC
""")

print("SQL — Stress per Social Weakness:")
sql_sw.show(truncate=False)

print(f"\nSQL completed in {time.time() - start_sql:.2f} seconds.")
