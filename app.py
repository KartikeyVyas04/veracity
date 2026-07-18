# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import warnings

# warnings.filterwarnings('ignore')

# # Set page configuration
# st.set_page_config(page_title="PharmaLens | Project Astra", page_icon="💊", layout="wide")

# # =========================================================
# # 1. HARDCODED CONFIGURATION (Replaces variables.yaml for single-file deployment)
# # =========================================================
# CONFIG = {
#     "id_columns": ["state", "district", "district_code", "lgd_code"],
#     "pillar_weights": {
#         "overall": {"P1": 0.30, "P2": 0.20, "P3": 0.20, "P4": 0.15, "P5": 0.15},
#         "chronic": {"P1": 0.35, "P2": 0.15, "P3": 0.25, "P4": 0.15, "P5": 0.10},
#         "acute":   {"P1": 0.40, "P2": 0.25, "P3": 0.10, "P4": 0.10, "P5": 0.15}
#     },
#     "opportunity_blend": 0.7,
#     "correlation_flag_threshold": 0.85,
#     "emerging_opportunity_variables": ["urbanization_growth", "income_growth_cagr", "ncd_risk_trend"],
#     "variables": {
#         # Pillar 1: Demand Potential
#         "population_total":       {"pillar": "P1", "indices": ["overall", "chronic", "acute"], "direction": "higher_better"},
#         "pct_pop_45_plus":        {"pillar": "P1", "indices": ["chronic"],                     "direction": "higher_better"},
#         "pct_diabetes_highsugar": {"pillar": "P1", "indices": ["overall", "chronic"],          "direction": "higher_better"},
#         "pct_hypertension":       {"pillar": "P1", "indices": ["overall", "chronic"],          "direction": "higher_better"},
#         "pct_ari_children":       {"pillar": "P1", "indices": ["overall", "acute"],            "direction": "higher_better"},
#         "pct_urban":              {"pillar": "P1", "indices": ["overall", "chronic"],          "direction": "higher_better"},
        
#         # Pillar 2: Access & Provider Landscape
#         "doctors_per_1000":       {"pillar": "P2", "indices": ["overall", "chronic", "acute"], "direction": "opportunity"},
#         "beds_per_1000":          {"pillar": "P2", "indices": ["overall", "chronic", "acute"], "direction": "opportunity"},
        
#         # Pillar 3: Economic Capacity
#         "per_capita_income":      {"pillar": "P3", "indices": ["overall", "chronic", "acute"], "direction": "higher_better"},
#         "pmjay_enrollment_rate":  {"pillar": "P3", "indices": ["overall", "chronic", "acute"], "direction": "higher_better"},
        
#         # Pillar 4: Growth Momentum
#         "urbanization_growth":    {"pillar": "P4", "indices": ["overall", "chronic"],          "direction": "higher_better"},
#         "income_growth_cagr":     {"pillar": "P4", "indices": ["overall", "chronic", "acute"], "direction": "higher_better"},
#         "ncd_risk_trend":         {"pillar": "P4", "indices": ["chronic"],                     "direction": "higher_better"},
        
#         # Pillar 5: Competitive Landscape 
#         "pharmacy_density":       {"pillar": "P5", "indices": ["overall", "chronic", "acute"], "direction": "higher_better"},
#         "jan_aushadhi_density":   {"pillar": "P5", "indices": ["overall", "chronic"],          "direction": "lower_better"}
#     }
# }

# # =========================================================
# # 2. PIPELINE CLASS (Embedded v2)
# # =========================================================
# class MAIPipeline:
#     def __init__(self, df: pd.DataFrame, cfg: dict):
#         self.cfg = cfg
#         self.raw = df.copy()
        
#         # Standardize column names to lowercase for robust matching
#         self.df = df.copy()
#         self.df.columns = [c.lower().strip() for c in self.df.columns]
        
#         # Dynamically map available ID columns
#         self.id_cols = [c for c in self.cfg["id_columns"] if c in self.df.columns]
#         if not self.id_cols:
#             # Fallback if no matching ID columns are found
#             self.df["district"] = [f"District_{i}" for i in range(len(self.df))]
#             self.id_cols = ["district"]
            
#         self.var_cfg = self.cfg["variables"]
#         self.available_vars = [v for v in self.var_cfg if v in self.df.columns]
        
#         self.flags = pd.DataFrame(index=self.df.index)
#         self.weights_log = {}
#         self.corr_redundancy_log = []

#     def impute_missing(self, df=None):
#         target = df if df is not None else self.df
#         for col in self.available_vars:
#             if df is None:
#                 self.flags[col + "_imputed"] = target[col].isna()
#             if target[col].isna().any() and "state" in target.columns:
#                 state_median = target.groupby("state")[col].transform("median")
#                 target[col] = target[col].fillna(state_median)
#             if target[col].isna().any():
#                 target[col] = target[col].fillna(target[col].median())
#         if df is None:
#             self.df = target
#             flag_cols = [c + "_imputed" for c in self.available_vars]
#             if flag_cols:
#                 self.df["data_confidence_score"] = (100 * (1 - self.flags[flag_cols].mean(axis=1))).round(1)
#             else:
#                 self.df["data_confidence_score"] = 100.0
#             return self
#         return target

#     def _normalize_frame(self, target):
#         opp_blend = self.cfg.get("opportunity_blend", 0.7)
#         skewed_vars = ["population_total", "per_capita_income"] 
#         for col in self.available_vars:
#             s = target[col].astype(float)
            
#             if col in skewed_vars:
#                 s = np.log1p(s)
                
#             lo, hi = s.quantile(0.01), s.quantile(0.99)
#             s_w = s.clip(lo, hi)
#             rng = (s_w.max() - s_w.min()) or 1
#             norm = 100 * (s_w - s_w.min()) / rng

#             direction = self.var_cfg[col]["direction"]
#             if direction == "opportunity":
#                 inv = 100 - norm
#                 target[col + "_norm"] = (opp_blend * norm + (1 - opp_blend) * inv).round(2)
#             elif direction == "lower_better":
#                 target[col + "_norm"] = (100 - norm).round(2)
#             else:
#                 target[col + "_norm"] = norm.round(2)
#         return target

#     def normalize(self):
#         self.df = self._normalize_frame(self.df)
#         return self

#     def correlation_pca_check(self, cols):
#         X = self.df[[c + "_norm" for c in cols]].values
#         if X.shape[1] < 2:
#             return {c: 1.0 for c in cols}
#         corr = pd.DataFrame(X, columns=cols).corr().abs()
#         thresh = self.cfg.get("correlation_flag_threshold", 0.85)
#         for i, c1 in enumerate(cols):
#             for c2 in cols[i + 1:]:
#                 if corr.loc[c1, c2] > thresh:
#                     self.corr_redundancy_log.append((c1, c2, round(corr.loc[c1, c2], 3)))

#         Xs = StandardScaler().fit_transform(X)
#         n_comp = min(len(cols), Xs.shape[0])
#         pca = PCA(n_components=n_comp)
#         pca.fit(Xs)
#         var_ratio = pca.explained_variance_ratio_
#         loadings = np.abs(pca.components_)
#         importance = (loadings * var_ratio[:, None]).sum(axis=0)
#         importance = importance / importance.sum()
#         return dict(zip(cols, importance))

#     def entropy_weights(self, cols):
#         X = self.df[[c + "_norm" for c in cols]].values + 1e-9
#         P = X / X.sum(axis=0, keepdims=True)
#         k = 1 / np.log(max(len(self.df), 2)) 
#         e = -k * (P * np.log(P)).sum(axis=0)
#         d = 1 - e
#         w = d / d.sum()
#         return dict(zip(cols, w))

#     def ahp_weights(self, cols):
#         return {c: 1 / len(cols) for c in cols}

#     def build_index(self, index_name):
#         df = self.df
#         vars_in_index = [v for v in self.available_vars if index_name in self.var_cfg[v]["indices"]]
#         if not vars_in_index: return self 
        
#         pca_w = self.correlation_pca_check(vars_in_index)
#         ent_w = self.entropy_weights(vars_in_index)
#         ahp_w = self.ahp_weights(vars_in_index)
#         blended_w = {v: (pca_w[v] + ent_w[v] + ahp_w[v]) / 3 for v in vars_in_index}

#         pw_cfg = self.cfg["pillar_weights"][index_name]
#         pillar_scores = {}
#         pillar_var_weights = {}
#         for pillar in pw_cfg:
#             pvars = [v for v in vars_in_index if self.var_cfg[v]["pillar"] == pillar]
#             if not pvars:
#                 continue
#             w_sum = sum(blended_w[v] for v in pvars)
#             w_norm = {v: blended_w[v] / w_sum for v in pvars}
#             pillar_var_weights[pillar] = w_norm
#             pillar_scores[pillar] = sum(df[v + "_norm"] * w_norm[v] for v in pvars)

#         active_pillars = list(pillar_scores.keys())
#         pw_sum = sum(pw_cfg[p] for p in active_pillars)
#         final = sum(pillar_scores[p] * (pw_cfg[p] / pw_sum) for p in active_pillars)

#         df[f"{index_name}_MAI"] = final.round(1)
#         for p in active_pillars:
#             df[f"{index_name}_{p}_score"] = pillar_scores[p].round(1)

#         self.weights_log[index_name] = {
#             "variable_weights": blended_w,
#             "pca_component": pca_w,
#             "entropy_component": ent_w,
#             "ahp_component": ahp_w,
#             "pillar_variable_weights": pillar_var_weights,
#             "pillar_weights_used": {p: pw_cfg[p] / pw_sum for p in active_pillars},
#         }
#         self.df = df
#         return self

#     def build_all_indices(self):
#         self.impute_missing().normalize()
#         for idx in ["overall", "chronic", "acute"]:
#             self.build_index(idx)
#         return self

#     def opportunity_gap_score(self):
#         df = self.df
#         # Graceful fallback if specific pillars are missing from the uploaded CSV
#         demand = df.get("overall_P1_score", pd.Series(50.0, index=df.index))
#         supply = df.get("overall_P2_score", pd.Series(50.0, index=df.index))
#         competition = df.get("overall_P5_score", pd.Series(50.0, index=df.index))
        
#         openness = (supply + competition) / 2

#         demand_pct = demand.rank(pct=True) * 100
#         openness_pct = openness.rank(pct=True) * 100
#         eps = 1e-6
#         harmonic = 2 * demand_pct * openness_pct / (demand_pct + openness_pct + eps)
#         df["opportunity_gap_score"] = harmonic.round(1)
#         self.df = df
#         return self

#     def emerging_opportunity_index(self):
#         df = self.df
#         cols = [c for c in self.cfg.get("emerging_opportunity_variables", []) if c in self.available_vars]
#         if not cols:
#             # Crash-proof fallback if trend variables are missing
#             df["emerging_opportunity_index"] = 0.0
#             self.df = df
#             return self
            
#         w = self.entropy_weights(cols) 
#         w_sum = sum(w.values())
#         score = sum(df[c + "_norm"] * (w[c] / w_sum) for c in cols)
#         df["emerging_opportunity_index"] = score.round(1)
#         self.emerging_opp_weights = w
#         self.df = df
#         return self

#     DRIVER_MAP = {
#         "population_total": (None, False),         
#         "per_capita_income": ("income_growth_cagr", False),
#         "pct_urban": ("urbanization_growth", True),
#         "pct_hypertension": ("ncd_risk_trend", True),
#         "pct_diabetes_highsugar": ("ncd_risk_trend", True),
#         "pct_pop_45_plus": (None, True),             
#         "pmjay_enrollment_rate": ("insurance_growth_rate", True),
#     }
#     ASSUMED_ANNUAL_POP_GROWTH = 0.012   
#     ASSUMED_ANNUAL_AGING_DRIFT = 0.25   

#     def project_future(self, years=3):
#         proj = self.df.copy()
#         for target_var, (driver_col, is_pct_point) in self.DRIVER_MAP.items():
#             if target_var not in proj.columns: continue
#             if driver_col and driver_col in proj.columns:
#                 rate = proj[driver_col] / 100.0
#             elif target_var == "population_total":
#                 rate = pd.Series(self.ASSUMED_ANNUAL_POP_GROWTH, index=proj.index)
#             elif target_var == "pct_pop_45_plus":
#                 proj[target_var] = (proj[target_var] + self.ASSUMED_ANNUAL_AGING_DRIFT * years).clip(0, 100)
#                 continue
#             else: continue

#             if is_pct_point:
#                 proj[target_var] = (proj[target_var] + rate * years).clip(0, 100)
#             else:
#                 proj[target_var] = proj[target_var] * (1 + rate) ** years

#         proj = self._normalize_frame(proj)
#         for index_name in ["overall", "chronic", "acute"]:
#             if index_name not in self.weights_log: continue
#             vars_in_index = [v for v in self.available_vars if index_name in self.var_cfg[v]["indices"]]
#             log = self.weights_log[index_name]
#             pw_cfg = log["pillar_weights_used"]
#             pillar_scores = {}
#             for pillar, w_norm in log["pillar_variable_weights"].items():
#                 pillar_scores[pillar] = sum(proj[v + "_norm"] * w for v, w in w_norm.items())
#             final = sum(pillar_scores[p] * pw_cfg[p] for p in pw_cfg if p in pillar_scores)
#             self.df[f"{index_name}_MAI_future"] = final.round(1)
#         self.projected_raw = proj
#         return self

#     def segment_districts(self, k=6):
#         df = self.df
#         feat_cols = ["overall_MAI", "chronic_MAI", "acute_MAI",
#                      "overall_P1_score", "overall_P2_score", "overall_P3_score",
#                      "overall_P4_score", "overall_P5_score"]
#         feat_cols = [c for c in feat_cols if c in df.columns]
#         if not feat_cols: return self
        
#         k = min(k, len(df))
#         X = StandardScaler().fit_transform(df[feat_cols])
#         km = KMeans(n_clusters=k, n_init=10, random_state=42)
#         labels = km.fit_predict(X)
#         df["segment_id"] = labels

#         centers = pd.DataFrame(km.cluster_centers_, columns=feat_cols)
#         centers_raw = df.groupby("segment_id")[feat_cols].mean()
#         label_map = self._auto_label_clusters(centers_raw)
#         df["segment_label"] = df["segment_id"].map(label_map)
#         self.cluster_centers = centers_raw
#         self.cluster_labels = label_map
#         self.df = df
#         return self

#     def _auto_label_clusters(self, centers_raw):
#         labels = {}
#         if "overall_MAI" not in centers_raw: return {i: f"Cluster {i}" for i in centers_raw.index}
        
#         overall_rank = centers_raw["overall_MAI"].rank(ascending=False)
#         for seg_id, row in centers_raw.iterrows():
#             if overall_rank[seg_id] == 1:
#                 labels[seg_id] = "Metro Leaders"
#             elif row.get("overall_P4_score", 0) == centers_raw.get("overall_P4_score", pd.Series([0])).max():
#                 labels[seg_id] = "Future Growth Markets"
#             elif row.get("chronic_MAI", 0) - row.get("acute_MAI", 0) > 8:
#                 labels[seg_id] = "Rural Chronic Belt" if row["overall_MAI"] < centers_raw["overall_MAI"].median() else "Emerging Urban"
#             elif row.get("acute_MAI", 0) - row.get("chronic_MAI", 0) > 8:
#                 labels[seg_id] = "Acute Disease Belt"
#             elif row.get("overall_P2_score", 100) == centers_raw.get("overall_P2_score", pd.Series([100])).min():
#                 labels[seg_id] = "Healthcare Desert"
#             else:
#                 labels[seg_id] = "Emerging Urban"
#         return labels

#     def explain_district(self, district_name, index_name="overall", top_n=4):
#         df = self.df
#         row = df[df["district"] == district_name]
#         if row.empty: return None
#         row = row.iloc[0]
#         if index_name not in self.weights_log: return None
        
#         log = self.weights_log[index_name]
#         contributions = {}
#         for pillar, w_norm in log["pillar_variable_weights"].items():
#             pw = log["pillar_weights_used"][pillar]
#             for v, w in w_norm.items():
#                 mean_val = df[v + "_norm"].mean()
#                 contributions[v] = pw * w * (row[v + "_norm"] - mean_val)
#         sorted_c = sorted(contributions.items(), key=lambda x: -x[1])
#         return {
#             "district": district_name,
#             "score": row.get(f"{index_name}_MAI", 0),
#             "avg_score": df[f"{index_name}_MAI"].mean().round(1) if f"{index_name}_MAI" in df.columns else 0,
#             "top_positive": sorted_c[:top_n],
#             "top_negative": sorted_c[-top_n:][::-1],
#         }

#     @staticmethod
#     def recommend(overall_mai, opportunity_gap):
#         if overall_mai >= 70:
#             return "Expand Immediately — core market, scale MR footprint & full portfolio"
#         elif overall_mai >= 55 and opportunity_gap >= 60:
#             return "Increase MR Coverage — strong underlying demand, access gap not yet closed"
#         elif overall_mai >= 55:
#             return "Maintain & Optimize — solid market, focus on share defense"
#         elif overall_mai >= 40 and opportunity_gap >= 65:
#             return "Pilot Market — high latent opportunity despite low current infra/income base"
#         elif overall_mai >= 40:
#             return "Pilot Market — monitor before committing full field force"
#         else:
#             return "Monitor — deprioritize until fundamentals improve"

#     def apply_recommendations(self):
#         df = self.df
#         gap_col = df["opportunity_gap_score"] if "opportunity_gap_score" in df.columns else pd.Series(50, index=df.index)
#         mai_col = df["overall_MAI"] if "overall_MAI" in df.columns else pd.Series(50, index=df.index)
#         df["recommended_strategy"] = [
#             self.recommend(m, g) for m, g in zip(mai_col, gap_col)
#         ]
#         self.df = df
#         return self

#     def get_rankings(self, index_name):
#         col = f"{index_name}_MAI"
#         if col not in self.df.columns: return pd.DataFrame()
#         extra = [c for c in ["opportunity_gap_score", "segment_label", "recommended_strategy",
#                               "data_confidence_score"] if c in self.df.columns]
#         out = self.df[self.id_cols + [col, f"{col}_future"] + extra].copy()
#         out = out.sort_values(col, ascending=False).reset_index(drop=True)
#         out.insert(0, "rank", out.index + 1)
#         return out

# # =========================================================
# # 3. STREAMLIT DASHBOARD UI
# # =========================================================

# st.title("PharmaLens: District Intelligence Platform")
# st.markdown("**Project Astra** - Trilytics Sun Pharma Case Solution")

# st.sidebar.header("Data Upload & Controls")
# st.sidebar.markdown("Upload `master_district_pharma_data.csv` to generate the MAI rankings and visualizations.")

# uploaded_file = st.sidebar.file_uploader("Upload District Data (CSV)", type=["csv"])

# @st.cache_data
# def generate_demo_data():
#     return pd.DataFrame({
#         "state": ["Gujarat", "Maharashtra", "UP", "Kerala", "Bihar", "Karnataka", "Tamil Nadu", "Rajasthan", "MP", "Punjab"],
#         "district": ["Ahmedabad", "Mumbai", "Lucknow", "Kochi", "Patna", "Gadag", "Chennai", "Jaipur", "Indore", "Ludhiana"],
#         "lgd_code": range(1, 11),
#         "population_total": [8000000, 20000000, 4500000, 3000000, 5800000, 1000000, 9000000, 3500000, 2500000, 3100000],
#         "pct_urban": [85, 100, 65, 70, 40, 35, 90, 50, 60, 55],
#         "pct_diabetes_highsugar": [12.5, 14.1, 9.5, 15.2, 7.1, 8.2, 13.5, 8.5, 10.1, 11.2],
#         "pct_hypertension": [15.1, 16.2, 11.0, 18.5, 9.2, 10.1, 16.0, 10.5, 12.0, 14.1],
#         "doctors_per_1000": [2.5, 3.8, 1.5, 3.2, 0.5, 0.8, 3.5, 1.2, 1.8, 2.0],
#         "per_capita_income": [210000, 350000, 120000, 280000, 65000, 85000, 310000, 140000, 180000, 230000],
#         "urbanization_growth": [3.2, 1.1, 4.5, 1.5, 2.1, 1.8, 2.5, 3.8, 4.1, 1.9],
#         "income_growth_cagr": [7.5, 5.2, 8.1, 6.0, 9.2, 6.5, 7.1, 8.5, 8.8, 5.9],
#         "pharmacy_density": [15, 25, 8, 12, 4, 3, 20, 6, 9, 11],
#         "jan_aushadhi_density": [5, 12, 3, 4, 1, 0, 8, 2, 4, 5],
#         "ncd_risk_trend": [1.2, 0.8, 1.5, 0.5, 1.8, 1.1, 0.9, 1.4, 1.6, 1.0]
#     })

# if uploaded_file is not None:
#     raw_df = pd.read_csv(uploaded_file)
#     st.sidebar.success("File uploaded successfully!")
# else:
#     raw_df = generate_demo_data()
#     st.sidebar.info("Using built-in demo dataset. Upload your Master CSV to test your own data.")

# with st.spinner('Running AI Assessment & Mathematical Modeling...'):
#     pipe = MAIPipeline(raw_df, CONFIG)
#     pipe.build_all_indices().opportunity_gap_score().emerging_opportunity_index()
#     pipe.project_future(years=3).segment_districts(k=4).apply_recommendations()
#     final_df = pipe.df

# tab1, tab2, tab3 = st.tabs(["📊 District Rankings", "📈 Strategic Matrices", "📄 Executive Deep-Dive"])

# with tab1:
#     st.header("Overall Market Attractiveness Index (MAI)")
#     st.markdown("Sorted by current potential. Includes future 2031 forecast and AI-generated field recommendations.")
    
#     display_cols = ["rank", "state", "district", "overall_MAI", "overall_MAI_future", "opportunity_gap_score", "segment_label", "recommended_strategy"]
#     rk_df = pipe.get_rankings("overall")
#     if not rk_df.empty:
#         available_cols = [c for c in display_cols if c in rk_df.columns]
#         st.dataframe(rk_df[available_cols], use_container_width=True, hide_index=True)
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.subheader("Emerging Opportunity Index")
#         st.markdown("Independent of current size—pure forward signal.")
        
#         # Safe-check to prevent crash if user forgot trend variables
#         if "emerging_opportunity_index" in final_df.columns and final_df["emerging_opportunity_index"].max() > 0:
#             emerg_rk = final_df.sort_values("emerging_opportunity_index", ascending=False).reset_index(drop=True)
#             emerg_rk.insert(0, "rank", emerg_rk.index + 1)
#             avail_emerg = [c for c in ["rank", "district", "overall_MAI", "emerging_opportunity_index", "recommended_strategy"] if c in emerg_rk.columns]
#             st.dataframe(emerg_rk[avail_emerg], use_container_width=True, hide_index=True)
#         else:
#             st.warning("Upload a dataset containing trend variables (e.g., `urbanization_growth`, `income_growth_cagr`) to activate the Emerging Opportunity Index.")
    
#     with col2:
#         st.subheader("Chronic Therapy Hotspots")
#         chronic_rk = pipe.get_rankings("chronic")
#         if not chronic_rk.empty:
#             avail_chron = [c for c in ["rank", "district", "chronic_MAI", "segment_label"] if c in chronic_rk.columns]
#             st.dataframe(chronic_rk[avail_chron], use_container_width=True, hide_index=True)

# with tab2:
#     st.header("Visual Strategy Matrices")
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("Therapy Skew vs Attractiveness")
#         st.markdown("Identifies whether to deploy specialist MRs (Chronic) or high-volume trade reps (Acute).")
#         if "chronic_MAI" in final_df.columns and "acute_MAI" in final_df.columns:
#             fig, ax = plt.subplots(figsize=(8, 6))
#             sns.set_theme(style="whitegrid")
#             skew = final_df["chronic_MAI"] - final_df["acute_MAI"]
#             sns.scatterplot(data=final_df, x="overall_MAI", y=skew, hue="segment_label" if "segment_label" in final_df.columns else None, s=150, palette="tab10", edgecolor="w", alpha=0.85, ax=ax)
#             ax.axhline(0, color="gray", linestyle="--")
#             ax.set_xlabel("Overall MAI (Current Size)")
#             ax.set_ylabel("Therapy Skew (Positive = Chronic Leaning)")
#             ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#             st.pyplot(fig)
            
#     with col2:
#         st.subheader("Opportunity Gap Analysis")
#         st.markdown("Identifies 'Hidden Gems': High unmet demand + Low Provider Supply/Competition.")
#         if "opportunity_gap_score" in final_df.columns and final_df["opportunity_gap_score"].max() > 0:
#             fig2, ax2 = plt.subplots(figsize=(8, 6))
#             sns.scatterplot(data=final_df, x="overall_MAI", y="opportunity_gap_score", color="#8e44ad", s=150, alpha=0.8, edgecolor="w", ax=ax2)
            
#             top_gap = final_df.nlargest(5, "opportunity_gap_score")
#             for _, row in top_gap.iterrows():
#                 if "district" in row:
#                     ax2.annotate(row["district"], (row["overall_MAI"], row["opportunity_gap_score"]), xytext=(5, 5), textcoords="offset points")
                
#             ax2.set_xlabel("Overall MAI")
#             ax2.set_ylabel("Opportunity Gap Score")
#             st.pyplot(fig2)
#         else:
#             st.info("Upload provider metrics (e.g., `doctors_per_1000`) to unlock the Opportunity Gap Chart.")

# with tab3:
#     st.header("Explainable AI: District Intelligence Reports")
#     st.markdown("Select a district to see the exact linear contribution decomposition driving its score.")
    
#     if "district" in final_df.columns:
#         selected_district = st.selectbox("Select District:", final_df["district"].unique())
        
#         if selected_district:
#             exp = pipe.explain_district(selected_district, "overall")
#             row = final_df[final_df["district"] == selected_district].iloc[0]
            
#             if exp:
#                 st.info(f"**Field Directive:** {row.get('recommended_strategy', 'N/A')}")
                
#                 c1, c2, c3, c4 = st.columns(4)
#                 c1.metric("Overall MAI", exp["score"], delta=f"Projected {row.get('overall_MAI_future', 'N/A')} by 2031")
#                 c2.metric("Opportunity Gap", row.get("opportunity_gap_score", "N/A"))
#                 c3.metric("Data Confidence", f"{row.get('data_confidence_score', 'N/A')}%")
#                 c4.metric("Segment", row.get("segment_label", "N/A"))
                
#                 st.write("---")
#                 st.subheader("Score Decomposition (vs. Portfolio Average)")
                
#                 col_pos, col_neg = st.columns(2)
#                 with col_pos:
#                     st.success("**Strengths (Positive Drivers)**")
#                     for v, c in exp["top_positive"]:
#                         if c > 0: st.markdown(f"+ **{v.replace('_', ' ').title()}**: +{c:.1f} pts")
                        
#                 with col_neg:
#                     st.error("**Weaknesses (Negative Drivers)**")
#                     for v, c in exp["top_negative"]:
#                         if c < 0: st.markdown(f"- **{v.replace('_', ' ').title()}**: {c:.1f} pts")

# st.sidebar.markdown("---")
# st.sidebar.caption("Built for Sun Pharma x Trilytics Challenge")











import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Veracity | District Intelligence", page_icon="📈", layout="wide")

# Custom CSS for the intro box - using rgba so text adapts to dark/light mode automatically
st.markdown("""
<style>
    .info-box {
        background-color: rgba(52, 152, 219, 0.1);
        border-left: 5px solid #3498db;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 1. HARDCODED CONFIGURATION 
# =========================================================
CONFIG = {
    "id_columns": ["state", "district", "district_code", "lgd_code"],
    "pillar_weights": {
        "overall": {"P1": 0.30, "P2": 0.20, "P3": 0.20, "P4": 0.15, "P5": 0.15},
        "chronic": {"P1": 0.35, "P2": 0.15, "P3": 0.25, "P4": 0.15, "P5": 0.10},
        "acute":   {"P1": 0.40, "P2": 0.25, "P3": 0.10, "P4": 0.10, "P5": 0.15}
    },
    "opportunity_blend": 0.7,
    "correlation_flag_threshold": 0.85,
    "emerging_opportunity_variables": ["urbanization_growth", "income_growth_cagr", "ncd_risk_trend"],
    "variables": {
        # Pillar 1: Demand Potential
        "population_total":       {"pillar": "P1", "indices": ["overall", "chronic", "acute"], "direction": "higher_better"},
        "pct_pop_45_plus":        {"pillar": "P1", "indices": ["chronic"],                     "direction": "higher_better"},
        "pct_diabetes_highsugar": {"pillar": "P1", "indices": ["overall", "chronic"],          "direction": "higher_better"},
        "pct_hypertension":       {"pillar": "P1", "indices": ["overall", "chronic"],          "direction": "higher_better"},
        "pct_ari_children":       {"pillar": "P1", "indices": ["overall", "acute"],            "direction": "higher_better"},
        "pct_urban":              {"pillar": "P1", "indices": ["overall", "chronic"],          "direction": "higher_better"},
        
        # Pillar 2: Access & Provider Landscape
        "doctors_per_1000":       {"pillar": "P2", "indices": ["overall", "chronic", "acute"], "direction": "opportunity"},
        "beds_per_1000":          {"pillar": "P2", "indices": ["overall", "chronic", "acute"], "direction": "opportunity"},
        
        # Pillar 3: Economic Capacity
        "per_capita_income":      {"pillar": "P3", "indices": ["overall", "chronic", "acute"], "direction": "higher_better"},
        "pmjay_enrollment_rate":  {"pillar": "P3", "indices": ["overall", "chronic", "acute"], "direction": "higher_better"},
        
        # Pillar 4: Growth Momentum
        "urbanization_growth":    {"pillar": "P4", "indices": ["overall", "chronic"],          "direction": "higher_better"},
        "income_growth_cagr":     {"pillar": "P4", "indices": ["overall", "chronic", "acute"], "direction": "higher_better"},
        "ncd_risk_trend":         {"pillar": "P4", "indices": ["chronic"],                     "direction": "higher_better"},
        
        # Pillar 5: Competitive Landscape 
        "pharmacy_density":       {"pillar": "P5", "indices": ["overall", "chronic", "acute"], "direction": "higher_better"},
        "jan_aushadhi_density":   {"pillar": "P5", "indices": ["overall", "chronic"],          "direction": "lower_better"}
    }
}

# =========================================================
# 2. PIPELINE CLASS
# =========================================================
class MAIPipeline:
    def __init__(self, df: pd.DataFrame, cfg: dict):
        self.cfg = cfg
        self.raw = df.copy()
        
        self.df = df.copy()
        self.df.columns = [c.lower().strip() for c in self.df.columns]
        
        self.id_cols = [c for c in self.cfg["id_columns"] if c in self.df.columns]
        if not self.id_cols:
            self.df["district"] = [f"District_{i}" for i in range(len(self.df))]
            self.id_cols = ["district"]
            
        self.var_cfg = self.cfg["variables"]
        self.available_vars = [v for v in self.var_cfg if v in self.df.columns]
        
        self.flags = pd.DataFrame(index=self.df.index)
        self.weights_log = {}
        self.corr_redundancy_log = []

    def impute_missing(self, df=None):
        target = df if df is not None else self.df
        for col in self.available_vars:
            if df is None:
                self.flags[col + "_imputed"] = target[col].isna()
            if target[col].isna().any() and "state" in target.columns:
                state_median = target.groupby("state")[col].transform("median")
                target[col] = target[col].fillna(state_median)
            if target[col].isna().any():
                target[col] = target[col].fillna(target[col].median())
        if df is None:
            self.df = target
            flag_cols = [c + "_imputed" for c in self.available_vars]
            if flag_cols:
                self.df["data_confidence_score"] = (100 * (1 - self.flags[flag_cols].mean(axis=1))).round(1)
            else:
                self.df["data_confidence_score"] = 100.0
            return self
        return target

    def _normalize_frame(self, target):
        opp_blend = self.cfg.get("opportunity_blend", 0.7)
        skewed_vars = ["population_total", "per_capita_income"] 
        for col in self.available_vars:
            s = target[col].astype(float)
            
            if col in skewed_vars:
                s = np.log1p(s)
                
            lo, hi = s.quantile(0.01), s.quantile(0.99)
            s_w = s.clip(lo, hi)
            rng = (s_w.max() - s_w.min()) or 1
            norm = 100 * (s_w - s_w.min()) / rng

            direction = self.var_cfg[col]["direction"]
            if direction == "opportunity":
                inv = 100 - norm
                target[col + "_norm"] = (opp_blend * norm + (1 - opp_blend) * inv).round(2)
            elif direction == "lower_better":
                target[col + "_norm"] = (100 - norm).round(2)
            else:
                target[col + "_norm"] = norm.round(2)
        return target

    def normalize(self):
        self.df = self._normalize_frame(self.df)
        return self

    def correlation_pca_check(self, cols):
        X = self.df[[c + "_norm" for c in cols]].values
        if X.shape[1] < 2:
            return {c: 1.0 for c in cols}
        corr = pd.DataFrame(X, columns=cols).corr().abs()
        thresh = self.cfg.get("correlation_flag_threshold", 0.85)
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                if corr.loc[c1, c2] > thresh:
                    self.corr_redundancy_log.append((c1, c2, round(corr.loc[c1, c2], 3)))

        Xs = StandardScaler().fit_transform(X)
        n_comp = min(len(cols), Xs.shape[0])
        pca = PCA(n_components=n_comp)
        pca.fit(Xs)
        var_ratio = pca.explained_variance_ratio_
        loadings = np.abs(pca.components_)
        importance = (loadings * var_ratio[:, None]).sum(axis=0)
        importance = importance / importance.sum()
        return dict(zip(cols, importance))

    def entropy_weights(self, cols):
        X = self.df[[c + "_norm" for c in cols]].values + 1e-9
        P = X / X.sum(axis=0, keepdims=True)
        k = 1 / np.log(max(len(self.df), 2)) 
        e = -k * (P * np.log(P)).sum(axis=0)
        d = 1 - e
        w = d / d.sum()
        return dict(zip(cols, w))

    def ahp_weights(self, cols):
        return {c: 1 / len(cols) for c in cols}

    def build_index(self, index_name):
        df = self.df
        vars_in_index = [v for v in self.available_vars if index_name in self.var_cfg[v]["indices"]]
        if not vars_in_index: return self 
        
        pca_w = self.correlation_pca_check(vars_in_index)
        ent_w = self.entropy_weights(vars_in_index)
        ahp_w = self.ahp_weights(vars_in_index)
        blended_w = {v: (pca_w[v] + ent_w[v] + ahp_w[v]) / 3 for v in vars_in_index}

        pw_cfg = self.cfg["pillar_weights"][index_name]
        pillar_scores = {}
        pillar_var_weights = {}
        for pillar in pw_cfg:
            pvars = [v for v in vars_in_index if self.var_cfg[v]["pillar"] == pillar]
            if not pvars:
                continue
            w_sum = sum(blended_w[v] for v in pvars)
            w_norm = {v: blended_w[v] / w_sum for v in pvars}
            pillar_var_weights[pillar] = w_norm
            pillar_scores[pillar] = sum(df[v + "_norm"] * w_norm[v] for v in pvars)

        active_pillars = list(pillar_scores.keys())
        pw_sum = sum(pw_cfg[p] for p in active_pillars)
        final = sum(pillar_scores[p] * (pw_cfg[p] / pw_sum) for p in active_pillars)

        df[f"{index_name}_MAI"] = final.round(1)
        for p in active_pillars:
            df[f"{index_name}_{p}_score"] = pillar_scores[p].round(1)

        self.weights_log[index_name] = {
            "variable_weights": blended_w,
            "pca_component": pca_w,
            "entropy_component": ent_w,
            "ahp_component": ahp_w,
            "pillar_variable_weights": pillar_var_weights,
            "pillar_weights_used": {p: pw_cfg[p] / pw_sum for p in active_pillars},
        }
        self.df = df
        return self

    def build_all_indices(self):
        self.impute_missing().normalize()
        for idx in ["overall", "chronic", "acute"]:
            self.build_index(idx)
        return self

    def opportunity_gap_score(self):
        df = self.df
        demand = df.get("overall_P1_score", pd.Series(50.0, index=df.index))
        supply = df.get("overall_P2_score", pd.Series(50.0, index=df.index))
        competition = df.get("overall_P5_score", pd.Series(50.0, index=df.index))
        
        openness = (supply + competition) / 2

        demand_pct = demand.rank(pct=True) * 100
        openness_pct = openness.rank(pct=True) * 100
        eps = 1e-6
        harmonic = 2 * demand_pct * openness_pct / (demand_pct + openness_pct + eps)
        df["opportunity_gap_score"] = harmonic.round(1)
        self.df = df
        return self

    def emerging_opportunity_index(self):
        df = self.df
        cols = [c for c in self.cfg.get("emerging_opportunity_variables", []) if c in self.available_vars]
        if not cols:
            df["emerging_opportunity_index"] = 0.0
            self.df = df
            return self
            
        w = self.entropy_weights(cols) 
        w_sum = sum(w.values())
        score = sum(df[c + "_norm"] * (w[c] / w_sum) for c in cols)
        df["emerging_opportunity_index"] = score.round(1)
        self.emerging_opp_weights = w
        self.df = df
        return self

    DRIVER_MAP = {
        "population_total": (None, False),         
        "per_capita_income": ("income_growth_cagr", False),
        "pct_urban": ("urbanization_growth", True),
        "pct_hypertension": ("ncd_risk_trend", True),
        "pct_diabetes_highsugar": ("ncd_risk_trend", True),
        "pct_pop_45_plus": (None, True),             
        "pmjay_enrollment_rate": ("insurance_growth_rate", True),
    }
    ASSUMED_ANNUAL_POP_GROWTH = 0.012   
    ASSUMED_ANNUAL_AGING_DRIFT = 0.25   

    def project_future(self, years=3):
        proj = self.df.copy()
        for target_var, (driver_col, is_pct_point) in self.DRIVER_MAP.items():
            if target_var not in proj.columns: continue
            if driver_col and driver_col in proj.columns:
                rate = proj[driver_col] / 100.0
            elif target_var == "population_total":
                rate = pd.Series(self.ASSUMED_ANNUAL_POP_GROWTH, index=proj.index)
            elif target_var == "pct_pop_45_plus":
                proj[target_var] = (proj[target_var] + self.ASSUMED_ANNUAL_AGING_DRIFT * years).clip(0, 100)
                continue
            else: continue

            if is_pct_point:
                proj[target_var] = (proj[target_var] + rate * years).clip(0, 100)
            else:
                proj[target_var] = proj[target_var] * (1 + rate) ** years

        proj = self._normalize_frame(proj)
        for index_name in ["overall", "chronic", "acute"]:
            if index_name not in self.weights_log: continue
            vars_in_index = [v for v in self.available_vars if index_name in self.var_cfg[v]["indices"]]
            log = self.weights_log[index_name]
            pw_cfg = log["pillar_weights_used"]
            pillar_scores = {}
            for pillar, w_norm in log["pillar_variable_weights"].items():
                pillar_scores[pillar] = sum(proj[v + "_norm"] * w for v, w in w_norm.items())
            final = sum(pillar_scores[p] * pw_cfg[p] for p in pw_cfg if p in pillar_scores)
            self.df[f"{index_name}_MAI_future"] = final.round(1)
        self.projected_raw = proj
        return self

    def segment_districts(self, k=6):
        df = self.df
        feat_cols = ["overall_MAI", "chronic_MAI", "acute_MAI",
                     "overall_P1_score", "overall_P2_score", "overall_P3_score",
                     "overall_P4_score", "overall_P5_score"]
        feat_cols = [c for c in feat_cols if c in df.columns]
        if not feat_cols: return self
        
        k = min(k, len(df))
        X = StandardScaler().fit_transform(df[feat_cols])
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        df["segment_id"] = labels

        centers = pd.DataFrame(km.cluster_centers_, columns=feat_cols)
        centers_raw = df.groupby("segment_id")[feat_cols].mean()
        label_map = self._auto_label_clusters(centers_raw)
        df["segment_label"] = df["segment_id"].map(label_map)
        self.cluster_centers = centers_raw
        self.cluster_labels = label_map
        self.df = df
        return self

    def _auto_label_clusters(self, centers_raw):
        labels = {}
        if "overall_MAI" not in centers_raw: return {i: f"Cluster {i}" for i in centers_raw.index}
        
        overall_rank = centers_raw["overall_MAI"].rank(ascending=False)
        for seg_id, row in centers_raw.iterrows():
            if overall_rank[seg_id] == 1:
                labels[seg_id] = "Metro Leaders"
            elif row.get("overall_P4_score", 0) == centers_raw.get("overall_P4_score", pd.Series([0])).max():
                labels[seg_id] = "Future Growth Markets"
            elif row.get("chronic_MAI", 0) - row.get("acute_MAI", 0) > 8:
                labels[seg_id] = "Rural Chronic Belt" if row["overall_MAI"] < centers_raw["overall_MAI"].median() else "Emerging Urban"
            elif row.get("acute_MAI", 0) - row.get("chronic_MAI", 0) > 8:
                labels[seg_id] = "Acute Disease Belt"
            elif row.get("overall_P2_score", 100) == centers_raw.get("overall_P2_score", pd.Series([100])).min():
                labels[seg_id] = "Healthcare Desert"
            else:
                labels[seg_id] = "Emerging Urban"
        return labels

    def explain_district(self, district_name, index_name="overall", top_n=4):
        df = self.df
        row = df[df["district"] == district_name]
        if row.empty: return None
        row = row.iloc[0]
        if index_name not in self.weights_log: return None
        
        log = self.weights_log[index_name]
        contributions = {}
        for pillar, w_norm in log["pillar_variable_weights"].items():
            pw = log["pillar_weights_used"][pillar]
            for v, w in w_norm.items():
                mean_val = df[v + "_norm"].mean()
                contributions[v] = pw * w * (row[v + "_norm"] - mean_val)
        sorted_c = sorted(contributions.items(), key=lambda x: -x[1])
        return {
            "district": district_name,
            "score": row.get(f"{index_name}_MAI", 0),
            "avg_score": df[f"{index_name}_MAI"].mean().round(1) if f"{index_name}_MAI" in df.columns else 0,
            "top_positive": sorted_c[:top_n],
            "top_negative": sorted_c[-top_n:][::-1],
        }

    @staticmethod
    def recommend(overall_mai, opportunity_gap):
        if overall_mai >= 70:
            return "Expand Immediately — core market, scale MR footprint & full portfolio"
        elif overall_mai >= 55 and opportunity_gap >= 60:
            return "Increase MR Coverage — strong underlying demand, access gap not yet closed"
        elif overall_mai >= 55:
            return "Maintain & Optimize — solid market, focus on share defense"
        elif overall_mai >= 40 and opportunity_gap >= 65:
            return "Pilot Market — high latent opportunity despite low current infra/income base"
        elif overall_mai >= 40:
            return "Pilot Market — monitor before committing full field force"
        else:
            return "Monitor — deprioritize until fundamentals improve"

    def apply_recommendations(self):
        df = self.df
        gap_col = df["opportunity_gap_score"] if "opportunity_gap_score" in df.columns else pd.Series(50, index=df.index)
        mai_col = df["overall_MAI"] if "overall_MAI" in df.columns else pd.Series(50, index=df.index)
        df["recommended_strategy"] = [
            self.recommend(m, g) for m, g in zip(mai_col, gap_col)
        ]
        self.df = df
        return self

    def get_rankings(self, index_name):
        col = f"{index_name}_MAI"
        if col not in self.df.columns: return pd.DataFrame()
        extra = [c for c in ["opportunity_gap_score", "segment_label", "recommended_strategy",
                              "data_confidence_score"] if c in self.df.columns]
        out = self.df[self.id_cols + [col, f"{col}_future"] + extra].copy()
        out = out.sort_values(col, ascending=False).reset_index(drop=True)
        out.insert(0, "rank", out.index + 1)
        return out

# =========================================================
# 3. STREAMLIT DASHBOARD UI
# =========================================================

st.title("Veracity: District Intelligence Platform")
st.markdown("**Sun Pharma Market Attractiveness Framework**")

st.sidebar.header("Data Upload & Controls")
st.sidebar.markdown("Upload your `master_district_pharma_data.csv` to generate the MAI rankings and visualizations.")

uploaded_file = st.sidebar.file_uploader("Upload District Data (CSV)", type=["csv"])

# ---------------------------------------------------------
# NO FILE UPLOADED STATE: Show Methodology & Guidelines
# ---------------------------------------------------------
if uploaded_file is None:
    st.markdown("""
    <div class="info-box">
        <h3>👋 Welcome to Veracity</h3>
        <p>This platform evaluates over 700 Indian districts to identify pharmaceutical market attractiveness. 
        It moves beyond static rankings by incorporating predictive growth models, objective entropy weighting, and competitive whitespace analysis.</p>
        <p><b>Please upload your CSV file in the sidebar to activate the intelligence engine.</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("Expected Data Parameters")
    st.markdown("Your uploaded CSV should ideally contain the following columns to generate the most accurate insights. *Note: The model will automatically adapt and mathematically impute any missing data based on regional medians.*")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Demand & Demographics")
        st.markdown("""
        *   **`population_total`**: Absolute district population (defines total addressable volume).
        *   **`pct_urban`**: Urbanization percentage.
        *   **`pct_diabetes_highsugar`**: NCD prevalence (Drives Chronic MAI).
        *   **`pct_hypertension`**: Cardiovascular risk prevalence.
        *   **`pct_ari_children`**: Acute respiratory infections (Drives Acute MAI).
        """)
        
        st.subheader("2. Access & Infrastructure")
        st.markdown("""
        *   **`doctors_per_1000`**: Clinical provider density. *(Scored dynamically: low density + high demand = high opportunity).*
        *   **`beds_per_1000`**: Hospital bed capacity.
        """)

    with col2:
        st.subheader("3. Economics & Competition")
        st.markdown("""
        *   **`per_capita_income`**: Defines out-of-pocket spending capability.
        *   **`pmjay_enrollment_rate`**: Ayushman Bharat coverage (institutional spending proxy).
        *   **`pharmacy_density`**: Retail channel maturity.
        *   **`jan_aushadhi_density`**: Government generic competition *(Scored negatively: higher density = lower attractiveness).*
        """)
        
        st.subheader("4. Growth Trends (Future Forecasts)")
        st.markdown("""
        *   **`urbanization_growth`**: 5-year urban shift trend.
        *   **`income_growth_cagr`**: Income growth velocity.
        *   **`ncd_risk_trend`**: Rate of increase in chronic diseases.
        """)
        
    st.stop() # Halts execution here until a file is uploaded

# ---------------------------------------------------------
# FILE UPLOADED STATE: Run Engine and Show Dashboard
# ---------------------------------------------------------
raw_df = pd.read_csv(uploaded_file)
st.sidebar.success("File uploaded successfully!")

with st.spinner('Running AI Assessment & Mathematical Modeling...'):
    pipe = MAIPipeline(raw_df, CONFIG)
    pipe.build_all_indices().opportunity_gap_score().emerging_opportunity_index()
    pipe.project_future(years=3).segment_districts(k=4).apply_recommendations()
    final_df = pipe.df

tab1, tab2, tab3 = st.tabs(["📊 District Rankings", "📈 Strategic Matrices", "📄 Executive Deep-Dive"])

with tab1:
    st.header("Overall Market Attractiveness Index (MAI)")
    st.markdown("""
    **What is this?** A blended macro-economic index that considers current population scale, total wealth, and existing healthcare infrastructure. Use this to guide broad, national-level territory planning.
    """)
    
    display_cols = ["rank", "state", "district", "overall_MAI", "overall_MAI_future", "opportunity_gap_score", "segment_label", "recommended_strategy"]
    rk_df = pipe.get_rankings("overall")
    if not rk_df.empty:
        available_cols = [c for c in display_cols if c in rk_df.columns]
        st.dataframe(rk_df[available_cols], use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Emerging Opportunity Index")
        st.markdown("**Objective:** Identifies 'tomorrow's markets.' Independent of current size—pure forward signal.")
        
        if "emerging_opportunity_index" in final_df.columns and final_df["emerging_opportunity_index"].max() > 0:
            emerg_rk = final_df.sort_values("emerging_opportunity_index", ascending=False).reset_index(drop=True)
            emerg_rk.insert(0, "rank", emerg_rk.index + 1)
            avail_emerg = [c for c in ["rank", "district", "overall_MAI", "emerging_opportunity_index", "recommended_strategy"] if c in emerg_rk.columns]
            st.dataframe(emerg_rk[avail_emerg], use_container_width=True, hide_index=True)
        else:
            st.warning("Upload a dataset containing trend variables (e.g., `urbanization_growth`, `income_growth_cagr`) to activate the Emerging Opportunity Index.")
    
    with col2:
        st.subheader("Chronic Therapy Hotspots")
        st.markdown("**Objective:** Ranks markets heavily weighted towards NFHS-5 lifestyle diseases (Diabetes/Hypertension) and high-income bands.")
        chronic_rk = pipe.get_rankings("chronic")
        if not chronic_rk.empty:
            avail_chron = [c for c in ["rank", "district", "chronic_MAI", "segment_label"] if c in chronic_rk.columns]
            st.dataframe(chronic_rk[avail_chron], use_container_width=True, hide_index=True)

with tab2:
    st.header("Visual Strategy Matrices")
    st.markdown("Hover over any dot on the matrices below to view specific district details and strategic directives.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Therapy Skew vs Attractiveness")
        st.markdown("""
        * **Above 0:** District leans Chronic. Deploy specialist MRs.
        * **Below 0:** District leans Acute. Deploy high-volume trade reps.
        """)
        
        if "chronic_MAI" in final_df.columns and "acute_MAI" in final_df.columns:
            final_df["therapy_skew"] = final_df["chronic_MAI"] - final_df["acute_MAI"]
            fig1 = px.scatter(
                final_df,
                x="overall_MAI",
                y="therapy_skew",
                color="segment_label" if "segment_label" in final_df.columns else None,
                hover_name="district",
                hover_data={
                    "overall_MAI": True,
                    "therapy_skew": True,
                    "chronic_MAI": True,
                    "acute_MAI": True,
                    "segment_label": False
                },
                labels={
                    "overall_MAI": "Overall MAI (Market Scale)",
                    "therapy_skew": "Therapy Skew (Chronic - Acute)",
                    "segment_label": "Segment"
                }
            )
            fig1.add_hline(y=0, line_dash="dash", line_color="gray")
            fig1.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color="DarkSlateGrey")))
            fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig1, use_container_width=True)
            
    with col2:
        st.subheader("Opportunity Gap Analysis")
        st.markdown("""
        * **Top-Right:** 'Hidden Gems'. High patient demand, but very low provider supply or competitor density.
        """)
        
        if "opportunity_gap_score" in final_df.columns and final_df["opportunity_gap_score"].max() > 0:
            fig2 = px.scatter(
                final_df,
                x="overall_MAI",
                y="opportunity_gap_score",
                color="segment_label" if "segment_label" in final_df.columns else None,
                hover_name="district",
                hover_data={
                    "overall_MAI": True,
                    "opportunity_gap_score": True,
                    "recommended_strategy": True,
                    "segment_label": False
                },
                labels={
                    "overall_MAI": "Overall MAI (Market Scale)",
                    "opportunity_gap_score": "Opportunity Gap Score",
                    "segment_label": "Segment"
                }
            )
            fig2.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color="DarkSlateGrey")))
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Upload provider metrics (e.g., `doctors_per_1000`) to unlock the Opportunity Gap Chart.")

with tab3:
    st.header("Explainable AI: District Intelligence Reports")
    st.markdown("Veracity does not use black-box scoring. Select any district to see the exact linear contribution decomposition driving its rank.")
    
    if "district" in final_df.columns:
        selected_district = st.selectbox("Select Target District:", final_df["district"].unique())
        
        if selected_district:
            exp = pipe.explain_district(selected_district, "overall")
            row = final_df[final_df["district"] == selected_district].iloc[0]
            
            if exp:
                st.info(f"**📋 Strategic Field Directive:** {row.get('recommended_strategy', 'N/A')}")
                
                # Using native st.metric to guarantee contrast and readability across Light/Dark modes
                c1, c2, c3, c4 = st.columns(4)
                
                future_delta = round(row.get('overall_MAI_future', exp['score']) - exp['score'], 1)
                
                c1.metric("Overall MAI", exp["score"], delta=f"{future_delta} Projected by 2031")
                c2.metric("Opportunity Gap Score", row.get("opportunity_gap_score", "N/A"))
                c3.metric("Data Confidence", f"{row.get('data_confidence_score', 'N/A')}%")
                c4.metric("Persona Segment", row.get("segment_label", "N/A"))
                
                st.write("---")
                st.subheader("Algorithmic Score Decomposition")
                st.markdown(f"*Showing how specific variables pushed this district above or below the national portfolio average score of **{exp['avg_score']}**.*")
                
                col_pos, col_neg = st.columns(2)
                with col_pos:
                    st.success("**✅ Structural Strengths (Positive Drivers)**")
                    for v, c in exp["top_positive"]:
                        if c > 0: 
                            clean_name = v.replace('_', ' ').title()
                            st.markdown(f"**{clean_name}**: +{c:.1f} points")
                        
                with col_neg:
                    st.error("**⚠️ Structural Weaknesses (Negative Drivers)**")
                    for v, c in exp["top_negative"]:
                        if c < 0: 
                            clean_name = v.replace('_', ' ').title()
                            st.markdown(f"**{clean_name}**: {c:.1f} points")
                            
                with st.expander("📖 Data Dictionary: What do these parameters mean?"):
                    st.markdown("""
                    * **Population Total**: The absolute demographic scale. High values indicate a large addressable market for high-volume acute therapies.
                    * **Pct Urban**: Percentage of population living in cities. Highly correlated with retail pharmacy penetration and modern trade access.
                    * **Pct Diabetes Highsugar**: The prevalence of metabolic syndrome. A primary driver for chronic and cardiovascular therapeutic portfolios.
                    * **Doctors per 1000**: Clinical provider density. Scored uniquely—extremely low density flags an underdeveloped market, but *moderate* density flags an underserved "white space" opportunity.
                    * **Per Capita Income**: Economic purchasing power. Essential for predicting uptake of premium branded generics and out-of-pocket expenditure capacity.
                    * **PMJAY Enrollment Rate**: Institutional insurance coverage, which drives hospital-based bulk procurement and generic utilization.
                    """)
