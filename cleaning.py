
# === 6. Preview ===
print("\nPreview:")
print(df.head(5).to_markdown())

# === 7. Null value summary ===
null_counts = df.isnull().sum()
total_rows = len(df)

print("\n=== Null Value Summary ===")
for col, n in null_counts.items():
    if n > 0:
        print(f"{col}: {n} nulls ({n/total_rows:.2%})")

total_nulls = null_counts.sum()
print(f"\nTotal null values in dataset: {total_nulls} ({total_nulls/(total_rows*len(df.columns)):.2%} of all cells)")



# === X. Check SocialWeakness consistency ===
col1 = "SocialWeaknessPrimary"
col2 = "SocialWeaknessSecondary"
match_ratio = (df[col1] == df[col2]).mean()

print("\n=== Consistency Check ===")
print(f"Share of matching rows ({col1}) and ({col2}): {match_ratio:.2%}")


# === 9. Remove all rows with any null value ===
before_rows = len(df)
df_clean = df.dropna(how="any")
after_rows = len(df_clean)

print("\n=== Null Removal (strict) ===")
print(f"Rows before: {before_rows}")
print(f"Rows after: {after_rows}")
print(f"Rows removed: {before_rows - after_rows} ({(before_rows - after_rows) / before_rows:.2%})")

# === 10. Dataset status after null removal ===
print("\n=== Dataset Status after NULL removal ===")
print(f"Total rows after cleaning: {len(df_clean)}")
print(f"Total columns: {len(df_clean.columns)}")

col1 = "SocialWeaknessPrimary"
col2 = "SocialWeaknessSecondary"
match_ratio = (df_clean[col1] == df_clean[col2]).mean()
print(f"Share of matching rows ({col1}) and ({col2}): {match_ratio:.2%}")

# === 11. Case Sensitivity Check ===
print("\n=== Case Sensitivity Check ===")
text_cols = df_clean.select_dtypes(include="object").columns
case_issues = {}
total_case_diff = 0

for col in text_cols:
    unique_count = df_clean[col].nunique()
    lower_count = df_clean[col].str.lower().nunique()
    if unique_count != lower_count:
        diff = unique_count - lower_count
        total_case_diff += diff
        case_issues[col] = (unique_count, lower_count, diff)

if case_issues:
    for col, (u, l, diff) in case_issues.items():
        print(f"{col}: {diff} values differ only by letter case")
    print(f"\nTotal case differences across all text columns: {total_case_diff}")
else:
    print("No case differences detected in text columns.")


# === 12. Normalize text case: capitalize first letter of each word ===
print("\n=== Text Case Normalization ===")

# Snapshot before change
before_snapshot = df_clean.copy()

# Apply normalization
df_clean = df_clean.map(lambda x: x.title() if isinstance(x, str) else x)

# Compare changes
case_changes_per_col = {}
total_case_changes = 0

for col in text_cols:
    changed = (before_snapshot[col] != df_clean[col]).sum()
    case_changes_per_col[col] = changed
    total_case_changes += changed

# Print summary
for col, changed in case_changes_per_col.items():
    if changed > 0:
        print(f"{col}: {changed} text entries changed")

print(f"\nTotal changed text entries across all text columns: {total_case_changes}")
print("Text case normalization completed.")


# === 13. Post-normalization case recheck ===
print("\n=== Post-Normalization Case Recheck ===")

post_case_diff = 0
for col in text_cols:
    unique_count = df_clean[col].nunique()
    lower_count = df_clean[col].str.lower().nunique()
    if unique_count != lower_count:
        diff = unique_count - lower_count
        post_case_diff += diff
        print(f"{col}: {diff} remaining differences")

if post_case_diff == 0:
    print("✅ All previous case differences resolved. Dataset is now case-consistent.")
else:
    print(f"⚠ {post_case_diff} case differences remain after normalization.")


# === 14. Dataset status after case ===
print("\n=== Dataset Status after NULL removal ===")
print(f"Total rows after cleaning: {len(df_clean)}")
print(f"Total columns: {len(df_clean.columns)}")

col1 = "SocialWeaknessPrimary"
col2 = "SocialWeaknessSecondary"
match_ratio = (df_clean[col1] == df_clean[col2]).mean()
print(f"Share of matching rows ({col1}) and ({col2}): {match_ratio:.2%}")

# === 15. Drop redundant SocialWeaknessSecondary column ===
print("\n=== Column Simplification ===")

if (df_clean["SocialWeaknessPrimary"] == df_clean["SocialWeaknessSecondary"]).all():
    df_clean = df_clean.drop(columns=["SocialWeaknessSecondary"])
    df_clean = df_clean.rename(columns={"SocialWeaknessPrimary": "SocialWeakness"})
    print("✅ SocialWeaknessSecondary dropped.")
    print("✅ SocialWeaknessPrimary renamed to SocialWeakness.")
else:
    print("⚠ Columns are not identical, skipping deletion.")


# === 15. Preview ===
print("\nPreview:")
print(df_clean.head(5).to_markdown())
