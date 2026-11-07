import sys
import os
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# تنظیم مسیر
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from AMSFS import AMSFS, load_dataset

# تنظیمات کلی
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "Dataset_ASU")
seed = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT_DIR, exist_ok=True)
max_display = 10

# لیست دیتاست‌ها
datasets_to_try = ['SMK_CAN_187', 'CLL_SUB_111', 'GLI_85', 'TOX_171', 'lung', 'Prostate_GE', 'colon']

selected_dataset = None
X_final = None
y_final = None
selected_indices_final = None

# ---------------------------------------------------------
# مرحله 1: پیدا کردن دیتاست مناسب
# ---------------------------------------------------------
print("جستجو برای دیتاستی با تعداد فیچر مناسب و دقت بالا...")

for ds in datasets_to_try:
    print(f"\nدر حال بررسی دیتاست: {ds}...")
    X, y = load_dataset(DATA_DIR, ds)
    if X is None:
        continue

    amsfs = AMSFS(max_drop=0.025, min_features=12, random_state=seed)
    X_selected = amsfs.fit_transform(X, y)
    n_feat = len(amsfs.selected_indices_)

    if n_feat < 8:
        print(f"  -> تعداد فیچرها ({n_feat}) کم است، رد شد.")
        continue

    try:
        X_tr_t, X_te_t, y_tr_t, y_te_t = train_test_split(X_selected, y, test_size=0.3, random_state=seed, stratify=y)
        scaler_t = StandardScaler()
        X_tr_t = scaler_t.fit_transform(X_tr_t)
        X_te_t = scaler_t.transform(X_te_t)

        rf_t = RandomForestClassifier(n_estimators=100, random_state=seed)
        rf_t.fit(X_tr_t, y_tr_t)
        acc = accuracy_score(y_te_t, rf_t.predict(X_te_t))
        print(f"  -> دقت مدل: {acc*100:.2f}% | تعداد فیچر: {n_feat}")

        if acc > 0.80:
            selected_dataset = ds
            X_final = X
            y_final = y
            selected_indices_final = amsfs.selected_indices_
            print(f"  -> دیتاست {ds} انتخاب شد!")
            break
    except Exception as e:
        print(f"  -> خطا در تست مدل: {e}")

if selected_dataset is None:
    print("هیچ دیتاست مناسبی پیدا نشد!")
    exit()

print(f"\nدیتاست نهایی: {selected_dataset}")
print(f"تعداد فیچرهای نهایی: {len(selected_indices_final)}")

# ---------------------------------------------------------
# مرحله 2: آموزش مدل روی کل داده‌ها
# ---------------------------------------------------------
X_selected_final = X_final[:, selected_indices_final]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected_final)

model = RandomForestClassifier(n_estimators=130, max_depth=9, random_state=seed, class_weight='balanced')
model.fit(X_scaled, y_final)

# ---------------------------------------------------------
# مرحله 3: محاسبه SHAP
# ---------------------------------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)

if isinstance(shap_values, list):
    # برای تفسیر، کلاس 1 را انتخاب می‌کنیم
    shap_values_class_1 = np.array(shap_values[1], copy=True)
else:
    shap_values_class_1 = np.array(shap_values, copy=True)

# اصلاح ابعاد اگر آرایه 3 بعدی بود
if shap_values_class_1.ndim == 3:
    shap_values_class_1 = shap_values_class_1[:, :, 0]

feature_names = [f"Gene_{i}" for i in selected_indices_final]

# پیدا کردن 10 ویژگی برتر
top_n = 10
top_indices = np.argsort(np.mean(np.abs(shap_values_class_1), axis=0))[::-1][:top_n]
top_indices_list = top_indices.flatten().tolist()
feature_names_top = [feature_names[i] for i in top_indices_list]

X_top = X_scaled[:, top_indices_list]
shap_top = shap_values_class_1[:, top_indices_list]

# ---------------------------------------------------------
# مرحله 4: رسم نمودارها
# ---------------------------------------------------------

# نمودار (a): Bar Plot (تک‌رنگ و تمیز برای مقاله)
print("Plotting (a) SHAP Feature Importance...")
plt.figure(figsize=(10, 6))
shap.summary_plot(np.abs(shap_values_class_1), X_scaled, feature_names=feature_names, plot_type="bar", max_display=max_display, show=False)
plt.title("(a) SHAP Feature Importance")
plt.savefig(os.path.join(OUT_DIR, "fig3a_importance.png"), dpi=300); plt.close()
plt.tight_layout()
plt.savefig(OUT_DIR + "/fig_" + str(plt.gcf().number) + ".png", dpi=300); plt.close()


# نمودار (b): Summary Plot (Beeswarm)
print("Plotting (b) SHAP Summary Plot...")
plt.figure(figsize=(10, 6))

for i in range(top_n):
    shap_vals = shap_top[:, i]
    feat_vals = X_top[:, i]
    
    # نرمال‌سازی برای رنگ
    norm_vals = (feat_vals - np.min(feat_vals)) / (np.max(feat_vals) - np.min(feat_vals) + 1e-8)
    colors = plt.cm.coolwarm(norm_vals)
    
    y_pos = np.ones_like(shap_vals) * i
    plt.scatter(shap_vals, y_pos, c=colors, s=20, alpha=0.7)

plt.yticks(range(top_n), feature_names_top)
plt.xlabel("SHAP value (impact on model output)")
plt.title("(b) SHAP Summary Plot")
plt.savefig(os.path.join(OUT_DIR, "fig3b_summary.png"), dpi=300); plt.close()
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(OUT_DIR + "/fig_" + str(plt.gcf().number) + ".png", dpi=300); plt.close()

# نمودار (c): SHAP Heatmap (ماتریس همبستگی)
print("Plotting (c) SHAP Heatmap...")

corr_matrix = np.corrcoef(shap_top.T)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
            xticklabels=feature_names_top, yticklabels=feature_names_top,
            cbar_kws={'label': 'Correlation'})
plt.title("(c) Correlation of SHAP values")
plt.savefig(os.path.join(OUT_DIR, "fig3c_correlation.png"), dpi=300); plt.close()
plt.xlabel("Features")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig(OUT_DIR + "/fig_" + str(plt.gcf().number) + ".png", dpi=300); plt.close()

