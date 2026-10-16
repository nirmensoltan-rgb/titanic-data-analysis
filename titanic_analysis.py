# -*- coding: utf-8 -*-
"""
مشروع تحليل واستكشاف بيانات تايتانيك
Titanic Data Analysis Project

المكتبات المستخدمة: Pandas, Numpy, Matplotlib, Seaborn
طريقة التشغيل:
    python titanic_analysis.py                 # يبحث عن Titanic.csv بجوار الملف
    python titanic_analysis.py path/to/file.csv
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------
# إعدادات عامة
# ----------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="Set2")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

# مجلد حفظ الرسوم البيانية (بجوار ملف الكود)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def section(title: str) -> None:
    """طباعة عنوان قسم بشكل منسق."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ======================================================================
# 1. تحميل واستكشاف البيانات (Data Loading & Exploration)
# ======================================================================
section("1. Data Loading & Exploration | تحميل واستكشاف البيانات")

# مسار الملف: من سطر الأوامر إن وُجد، وإلا Titanic.csv في نفس المجلد
csv_path = sys.argv[1] if len(sys.argv) > 1 else "Titanic.csv"
df = pd.read_csv(csv_path)

# أول 5 أسطر
print("\n--- First 5 rows | أول 5 أسطر ---")
print(df.head())

# حجم الداتا (عدد الصفوف، عدد الأعمدة)
print("\n--- Shape | حجم البيانات ---")
print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

# أنواع البيانات ومعلومات عامة
print("\n--- Info | أنواع البيانات ---")
df.info()

# وصف إحصائي سريع للأعمدة الرقمية
print("\n--- Describe | الوصف الإحصائي ---")
print(df.describe())


# ======================================================================
# 2. تنظيف ومعالجة البيانات (Data Cleaning & Preprocessing)
# ======================================================================
section("2. Data Cleaning & Preprocessing | تنظيف ومعالجة البيانات")

# ----- 2.1 اكتشاف القيم المفقودة -----
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({"Missing": missing, "Missing %": missing_pct})
print("\n--- Missing values before cleaning | القيم المفقودة قبل التنظيف ---")
print(missing_report[missing_report["Missing"] > 0])

# ----- 2.2 معالجة القيم المفقودة -----

# (أ) Age: نعوّض بالوسيط (Median) لكل مجموعة (Sex + Pclass).
#     الوسيط أفضل من المتوسط لأنه لا يتأثر بالقيم الشاذة،
#     والتعويض حسب المجموعة أدق من استخدام رقم واحد لكل الركاب.
df["Age"] = df["Age"].fillna(
    df.groupby(["Sex", "Pclass"])["Age"].transform("median")
)
# احتياط: لو بقيت مجموعة كاملة بلا أعمار نستخدم الوسيط العام
df["Age"] = df["Age"].fillna(df["Age"].median())

# (ب) Fare: قيمة مفقودة واحدة فقط -> نعوّضها بوسيط سعر درجة التذكرة نفسها
df["Fare"] = df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"))

# (ج) Embarked: (إن وُجدت قيم مفقودة) نعوّضها بالمنوال Mode لأنه عمود فئوي
if df["Embarked"].isnull().any():
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# (د) Cabin: أكثر من 75% من القيم مفقودة، والتعويض هنا سيكون تخميناً
#     وليس بيانات حقيقية، لذلك نحذف العمود.
cabin_missing_pct = df["Cabin"].isnull().mean() * 100
print(f"\nCabin is {cabin_missing_pct:.1f}% missing -> column dropped.")
df = df.drop(columns=["Cabin"])

print("\n--- Missing values after cleaning | القيم المفقودة بعد التنظيف ---")
print(df.isnull().sum())

# ----- 2.3 إضافة أعمدة جديدة -----
# FamilySize = الأشقاء/الأزواج + الآباء/الأبناء + الراكب نفسه
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

print("\n--- FamilySize preview | معاينة العمود الجديد ---")
print(df[["SibSp", "Parch", "FamilySize"]].head())
print("\nFamilySize value counts:")
print(df["FamilySize"].value_counts().sort_index())


# ======================================================================
# 3. التحليل الاستكشافي للبيانات (EDA)
# ======================================================================
section("3. Exploratory Data Analysis | التحليل الاستكشافي")

# ----- 3.1 نسبة النجاة العامة -----
overall_survival = df["Survived"].mean() * 100
survivors = int(df["Survived"].sum())
print(f"\nOverall survival rate | نسبة النجاة العامة: {overall_survival:.2f}% "
      f"({survivors} of {len(df)} passengers)")

# ----- 3.2 تأثير الجنس على النجاة -----
survival_by_sex = (
    df.groupby("Sex")["Survived"]
    .agg(Passengers="count", Survivors="sum", SurvivalRate="mean")
)
survival_by_sex["SurvivalRate"] = (survival_by_sex["SurvivalRate"] * 100).round(2)
print("\n--- Survival by Sex | النجاة حسب الجنس ---")
print(survival_by_sex)

# ----- 3.3 تأثير درجة التذكرة على النجاة -----
survival_by_class = (
    df.groupby("Pclass")["Survived"]
    .agg(Passengers="count", Survivors="sum", SurvivalRate="mean")
)
survival_by_class["SurvivalRate"] = (survival_by_class["SurvivalRate"] * 100).round(2)
print("\n--- Survival by Pclass | النجاة حسب درجة التذكرة ---")
print(survival_by_class)

# ----- 3.4 متوسط سعر التذكرة لكل درجة -----
avg_fare_by_class = df.groupby("Pclass")["Fare"].mean().round(2)
print("\n--- Average Fare by Pclass | متوسط سعر التذكرة لكل درجة ---")
print(avg_fare_by_class)

# ----- 3.5 (إضافي) تأثير الجنس والدرجة معاً -----
print("\n--- Survival rate by Sex & Pclass (%) | النجاة حسب الجنس والدرجة معاً ---")
print((df.pivot_table(values="Survived", index="Sex", columns="Pclass",
                      aggfunc="mean") * 100).round(2))


# ======================================================================
# 4. التصور البصري للبيانات (Data Visualization)
# ======================================================================
section("4. Data Visualization | التصور البصري")

# تحويل Survived إلى نص لتظهر أسماء واضحة في الرسوم
df["Survival Status"] = df["Survived"].map({0: "Did not survive", 1: "Survived"})

# ----- 4.1 Count Plot: الناجون مقابل غير الناجين حسب الجنس -----
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x="Survival Status", hue="Sex",
              order=["Did not survive", "Survived"], ax=ax)
ax.set_title("Survivors vs Non-Survivors by Sex", fontsize=14, fontweight="bold")
ax.set_xlabel("Survival Status")
ax.set_ylabel("Number of Passengers")
for container in ax.containers:          # كتابة العدد فوق كل عمود
    ax.bar_label(container, padding=2)
ax.legend(title="Sex")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_countplot_survival_by_sex.png"), dpi=150)
plt.show()

# ----- 4.2 Distribution Plot: توزيع الأعمار للناجين مقابل غير الناجين -----
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(data=df, x="Age", hue="Survival Status", kde=True,
             stat="density", common_norm=False, bins=25,
             element="step", ax=ax)
ax.set_title("Age Distribution: Survivors vs Non-Survivors",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Density")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_age_distribution_by_survival.png"), dpi=150)
plt.show()

# ----- 4.3 Box Plot: درجة التذكرة مقابل سعر التذكرة -----
# المقياس اللوغاريتمي لأن Fare فيه قيم شاذة كبيرة تضغط الرسم
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="Pclass", y="Fare", hue="Pclass",
            palette="Set2", legend=False, ax=ax)
ax.set_yscale("log")
ax.set_title("Ticket Fare by Passenger Class (log scale)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Passenger Class (1 = First, 3 = Third)")
ax.set_ylabel("Fare (log scale)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_boxplot_fare_by_pclass.png"), dpi=150)
plt.show()

# ----- 4.4 Heatmap: مصفوفة الارتباط للأعمدة الرقمية -----
# نستبعد PassengerId لأنه مجرد رقم تسلسلي بلا معنى تحليلي
numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, center=0, square=True,
            linewidths=0.5, cbar_kws={"label": "Correlation"}, ax=ax)
ax.set_title("Correlation Matrix of Numeric Features",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_correlation_heatmap.png"), dpi=150)
plt.show()

print("\n--- Correlation with Survived | الارتباط مع النجاة ---")
print(corr["Survived"].drop("Survived").sort_values(ascending=False).round(3))

print(f"\nDone. Figures saved in: {OUTPUT_DIR}")
