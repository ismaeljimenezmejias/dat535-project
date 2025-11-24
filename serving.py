from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
import time

# -----------------------------
# Spark configuration
# -----------------------------
spark = SparkSession.builder.appName("UseCase1_RDD").getOrCreate()

df = spark.read.parquet("gs://medallion-dat535/gold/rdd/mental_health_clean.parquet")
df.createOrReplaceTempView("mental_health")

# Convert DF to RDD and cache it
rdd = df.rdd.repartition(4).cache()
rdd = rdd.sample(False, 0.01, seed=42)

print("\n========== USE CASE 1 (RDD ONLY) ==========\n")
start = time.time()

# ============================
# A) Factor most associated with HIGH stress
# ============================
categorical_cols = [
    "Gender", "Country", "Occupation", "SelfEmployed", "FamilyHistory",
    "Treatment", "DaysIndoors", "HabitsChange", "MentalHealthHistory",
    "MoodSwings", "SocialWeakness", "CopingStruggles", "WorkInterest",
    "MentalHealthInterview", "CareOptions"
]

results = {}
# Check the actual values
vals = rdd.map(lambda r: r["IncreasingStress"]).distinct().collect()
print("Unique values of IncreasingStress:", vals)

for col in categorical_cols:
    # (category_value, isHighStress)
    pairs = rdd.map(lambda row: (
        row[col],
        1 if row["IncreasingStress"].strip().lower() == "yes" else 0
    )).filter(lambda x: x[0] is not None)  # remove nulls

    # (category_value, (sumHigh, count))
    stats = pairs.aggregateByKey(
        (0, 0),
        lambda acc, value: (acc[0] + value, acc[1] + 1),
        lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])
    )

    # proportion of "High" by category
    proportions = stats.mapValues(lambda x: x[0] / x[1] if x[1] > 0 else 0)
    # category with highest High Stress proportion
    best = proportions.reduce(lambda a, b: a if a[1] > b[1] else b)
    results[col] = best

print("\n=== Factor most associated with HIGH stress ===")
for col, (category, prop) in results.items():
    print(f"{col}: '{category}' → {prop:.2f}")

# ============================
# B) Stress distribution by country
# ============================
country_pairs = rdd.map(lambda r: (r["Country"], r["IncreasingStress"])).filter(lambda x: x[0] is not None)

def count_categories(acc, v):
    acc[v] = acc.get(v, 0) + 1
    return acc

def merge_dicts(d1, d2):
    for k, v in d2.items():
        d1[k] = d1.get(k, 0) + v
    return d1

country_stats = country_pairs.aggregateByKey({}, count_categories, merge_dicts).collect()

print("\n=== Stress distribution by country ===")
for country, stats in country_stats:
    total = sum(stats.values())
    proportions = {k: v / total for k, v in stats.items()}
    print(f"{country}: {proportions}")

# ============================
# C) Global distribution of IncreasingStress
# ============================
stress_counts = rdd.map(lambda r: (r["IncreasingStress"], 1)) \
                   .filter(lambda x: x[0] is not None) \
                   .reduceByKey(lambda a, b: a + b) \
                   .collect()
total = sum([c for _, c in stress_counts])
print("\n=== Global distribution of IncreasingStress ===")
for k, v in stress_counts:
    print(f"{k}: {v} ({v/total:.2%})")

# ============================
# D) People with mental health history by country
# ============================
mh_history = rdd.map(lambda r: (r["Country"], 1 if r["MentalHealthHistory"] and r["MentalHealthHistory"].strip().lower() == "yes" else 0)) \
                .filter(lambda x: x[0] is not None)
mh_by_country = mh_history.aggregateByKey((0,0), 
                                          lambda acc, v: (acc[0]+v, acc[1]+1), 
                                          lambda acc1, acc2: (acc1[0]+acc2[0], acc1[1]+acc2[1])) \
                    .collect()
print("\n=== Proportion of people with mental health history by country ===")
for country, (yes_count, total_count) in mh_by_country:
    print(f"{country}: {yes_count}/{total_count} ({yes_count/total_count:.2%})")

# ============================
# E) Proportion of SelfEmployed by Occupation
# ============================
selfemp_pairs = rdd.map(lambda r: (r["Occupation"], 1 if r["SelfEmployed"] and r["SelfEmployed"].strip().lower() == "yes" else 0)) \
                   .filter(lambda x: x[0] is not None)
selfemp_stats = selfemp_pairs.aggregateByKey((0,0), 
                                             lambda acc, v: (acc[0]+v, acc[1]+1), 
                                             lambda acc1, acc2: (acc1[0]+acc2[0], acc1[1]+acc2[1])) \
                         .collect()
print("\n=== Proportion of SelfEmployed by Occupation ===")
for occ, (yes_count, total_count) in selfemp_stats:
    print(f"{occ}: {yes_count}/{total_count} ({yes_count/total_count:.2%})")

print(f"\nRDD completed in {time.time() - start:.2f} seconds.\n")



# ============================
# DataFrame API
# ============================
start_df = time.time()

# A) Factor most associated with HIGH stress
df_results = {}
for col in categorical_cols:
    prop_df = df.filter(F.col(col).isNotNull()) \
                .withColumn("isHigh", F.when(F.lower(F.col("IncreasingStress"))=="yes",1).otherwise(0)) \
                .groupBy(col) \
                .agg(F.avg("isHigh").alias("prop")) \
                .orderBy(F.desc("prop")) \
                .limit(1) \
                .collect()
    df_results[col] = (prop_df[0][col], prop_df[0]["prop"])

# B) Stress distribution by country
df_country = df.groupBy("Country", "IncreasingStress") \
               .count() \
               .withColumn("prop", F.col("count") / F.sum("count").over(Window.partitionBy("Country"))) \
               .orderBy("Country", "IncreasingStress")

# C) Global distribution of IncreasingStress
df_stress_global = df.groupBy("IncreasingStress") \
                     .count() \
                     .withColumn("prop", F.col("count")/F.sum("count").over(Window.partitionBy()))

# D) People with mental health history by country
df_mh = df.filter(F.col("MentalHealthHistory").isNotNull()) \
          .withColumn("hasHistory", F.when(F.lower(F.col("MentalHealthHistory"))=="yes",1).otherwise(0)) \
          .groupBy("Country") \
          .agg(F.sum("hasHistory").alias("yes_count"), F.count("*").alias("total_count")) \
          .withColumn("prop", F.col("yes_count")/F.col("total_count"))

# E) Proportion of SelfEmployed by Occupation
df_selfemp = df.filter(F.col("SelfEmployed").isNotNull() & F.col("Occupation").isNotNull()) \
               .withColumn("isSelf", F.when(F.lower(F.col("SelfEmployed"))=="yes",1).otherwise(0)) \
               .groupBy("Occupation") \
               .agg(F.sum("isSelf").alias("yes_count"), F.count("*").alias("total_count")) \
               .withColumn("prop", F.col("yes_count")/F.col("total_count"))

time_df = time.time() - start_df

# ============================
# SQL
# ============================
start_sql = time.time()

# A) Factor most associated with HIGH stress
sql_results = {}
for col in categorical_cols:
    query = f"""
    SELECT {col}, AVG(CASE WHEN LOWER(IncreasingStress)='yes' THEN 1 ELSE 0 END) AS prop
    FROM mental_health
    WHERE {col} IS NOT NULL
    GROUP BY {col}
    ORDER BY prop DESC
    LIMIT 1
    """
    row = spark.sql(query).collect()[0]
    sql_results[col] = (row[col], row['prop'])

# B) Stress distribution by country
sql_country = spark.sql("""
    SELECT Country, IncreasingStress, COUNT(*) AS count,
           COUNT(*) * 1.0 / SUM(COUNT(*)) OVER(PARTITION BY Country) AS prop
    FROM mental_health
    GROUP BY Country, IncreasingStress
    ORDER BY Country, IncreasingStress
""")

# C) Global distribution of IncreasingStress
sql_stress_global = spark.sql("""
    SELECT IncreasingStress, COUNT(*) AS count,
           COUNT(*) * 1.0 / SUM(COUNT(*)) OVER() AS prop
    FROM mental_health
    GROUP BY IncreasingStress
""")

# D) People with mental health history by country
sql_mh = spark.sql("""
    SELECT Country,
           SUM(CASE WHEN LOWER(MentalHealthHistory)='yes' THEN 1 ELSE 0 END) AS yes_count,
           COUNT(*) AS total_count,
           SUM(CASE WHEN LOWER(MentalHealthHistory)='yes' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS prop
    FROM mental_health
    WHERE MentalHealthHistory IS NOT NULL
    GROUP BY Country
""")

# E) Proportion of SelfEmployed by Occupation
sql_selfemp = spark.sql("""
    SELECT Occupation,
           SUM(CASE WHEN LOWER(SelfEmployed)='yes' THEN 1 ELSE 0 END) AS yes_count,
           COUNT(*) AS total_count,
           SUM(CASE WHEN LOWER(SelfEmployed)='yes' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS prop
    FROM mental_health
    WHERE SelfEmployed IS NOT NULL AND Occupation IS NOT NULL
    GROUP BY Occupation
""")

time_sql = time.time() - start_sql

# Mostrar todas las vistas temporales registradas
spark.catalog.listTables()


# ============================
# Print execution times
# ============================
print("\n=== Execution times (seconds) ===")
print(f"RDD total time: {time.time() - start:.2f}")
print(f"DataFrame API total time: {time_df:.2f}")
print(f"SQL total time: {time_sql:.2f}\n")