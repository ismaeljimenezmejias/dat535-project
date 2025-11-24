from pyspark.sql import SparkSession
import time

# -----------------------------
# Configuración de Spark
# -----------------------------
spark = SparkSession.builder.appName("UseCase1_RDD").getOrCreate()

df = spark.read.parquet("gs://medallion-dat535/gold/rdd/mental_health_clean.parquet")

# Convertimos el DF a RDD y cacheamos

rdd = df.rdd.repartition(4).cache()
rdd = rdd.sample(False, 0.05, seed=42)

print("\n========== USE CASE 1 (RDD ONLY) ==========\n")
start = time.time()

# ============================
# A) ¿Qué factor se asocia más con HIGH stress?
# ============================

categorical_cols = [
    "Gender", "Country", "Occupation", "SelfEmployed", "FamilyHistory",
    "Treatment", "DaysIndoors", "HabitsChange", "MentalHealthHistory",
    "MoodSwings", "SocialWeakness", "CopingStruggles", "WorkInterest",
    "MentalHealthInterview", "CareOptions"
]

results = {}

# Ver qué valores hay realmente
vals = rdd.map(lambda r: r["IncreasingStress"]).distinct().collect()
print(vals)

for col in categorical_cols:
    # (category_value, isHighStress)
    pairs = rdd.map(lambda row: (
        row[col],
        1 if row["IncreasingStress"].strip().lower() == "yes" else 0
    )).filter(lambda x: x[0] is not None)  # eliminamos nulos

    # (category_value, (sumHigh, count))
    stats = pairs.aggregateByKey(
        (0, 0),
        lambda acc, value: (acc[0] + value, acc[1] + 1),
        lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])
    )

    # proporción de "High" por categoría
    proportions = stats.mapValues(lambda x: x[0] / x[1] if x[1] > 0 else 0)

    # categoría con mayor proporción de High Stress
    best = proportions.reduce(lambda a, b: a if a[1] > b[1] else b)

    results[col] = best

print("=== Factor con mayor asociación con HIGH stress ===")
for col, (category, prop) in results.items():
    print(f"{col}: '{category}' → {prop:.2f}")

# ============================
# B) Agrupación por país
# ============================

country_pairs = rdd.map(lambda r: (r["Country"], r["IncreasingStress"])).filter(lambda x: x[0] is not None)

def count_categories(acc, v):
    acc[v] = acc.get(v, 0) + 1
    return acc

def merge_dicts(d1, d2):
    for k, v in d2.items():
        d1[k] = d1.get(k, 0) + v
    return d1

country_stats = country_pairs.aggregateByKey(
    {},
    count_categories,
    merge_dicts
).collect()

print("\n=== Distribución de stress por país ===")
for country, stats in country_stats:
    total = sum(stats.values())
    proportions = {k: v / total for k, v in stats.items()}
    print(f"{country}: {proportions}")

print(f"\nRDD Use Case 1 completed in {time.time() - start:.2f} seconds.\n")

# ============================
# C) Distribución global de IncreasingStress
# ============================
stress_counts = rdd.map(lambda r: (r["IncreasingStress"], 1)) \
                   .filter(lambda x: x[0] is not None) \
                   .reduceByKey(lambda a, b: a + b) \
                   .collect()
total = sum([c for _, c in stress_counts])
print("\n=== Distribución global de IncreasingStress ===")
for k, v in stress_counts:
    print(f"{k}: {v} ({v/total:.2%})")

# ============================
# D) Personas con historial de salud mental por país
# ============================
mh_history = rdd.map(lambda r: (r["Country"], 1 if r["MentalHealthHistory"] and r["MentalHealthHistory"].strip().lower() == "yes" else 0)) \
                .filter(lambda x: x[0] is not None)
mh_by_country = mh_history.aggregateByKey((0,0), 
                                          lambda acc, v: (acc[0]+v, acc[1]+1), 
                                          lambda acc1, acc2: (acc1[0]+acc2[0], acc1[1]+acc2[1])) \
                    .collect()
print("\n=== Proporción de personas con historial de salud mental por país ===")
for country, (yes_count, total_count) in mh_by_country:
    print(f"{country}: {yes_count}/{total_count} ({yes_count/total_count:.2%})")

# ============================
# E) Proporción de SelfEmployed por Occupation
# ============================
selfemp_pairs = rdd.map(lambda r: (r["Occupation"], 1 if r["SelfEmployed"] and r["SelfEmployed"].strip().lower() == "yes" else 0)) \
                   .filter(lambda x: x[0] is not None)
selfemp_stats = selfemp_pairs.aggregateByKey((0,0), 
                                             lambda acc, v: (acc[0]+v, acc[1]+1), 
                                             lambda acc1, acc2: (acc1[0]+acc2[0], acc1[1]+acc2[1])) \
                         .collect()
print("\n=== Proporción de SelfEmployed por Occupation ===")
for occ, (yes_count, total_count) in selfemp_stats:
    print(f"{occ}: {yes_count}/{total_count} ({yes_count/total_count:.2%})")

print(f"\nRDD Use Case 1 completed in {time.time() - start:.2f} seconds.\n")