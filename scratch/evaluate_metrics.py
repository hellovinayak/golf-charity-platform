import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")
from src.data_loader import load_sdg_data, STATES_AND_UTS
from src.risk_classifier import classify_score

def compute_all_metrics():
    df = load_sdg_data()
    SDG_TARGETS = ["SDG3_Health", "SDG4_Education", "SDG13_Climate"]
    
    all_entities = list(df["State"].unique())
    
    train_df = df[df["Year"] <= 2021]
    test_df = df[df["Year"] >= 2022]
    
    holdout_records = []
    full_records = []
    
    tier_actuals = []
    tier_preds = []
    
    # Store all point predictions for overall pooling
    all_y_true_holdout = []
    all_y_pred_holdout = []
    all_y_naive_holdout = []
    
    for state in all_entities:
        state_train = train_df[train_df["State"] == state]
        state_test = test_df[test_df["State"] == state]
        state_full = df[df["State"] == state]
        
        for sdg in SDG_TARGETS:
            # --- 1. Holdout Model (2018-2021 train, 2022-2023 test) ---
            X_tr = state_train[["Year"]].values
            y_tr = state_train[sdg].values
            
            X_te = state_test[["Year"]].values
            y_te = state_test[sdg].values
            
            val_model = LinearRegression()
            val_model.fit(X_tr, y_tr)
            y_pred_te = val_model.predict(X_te)
            
            # Prediction Interval on Holdout
            residuals_tr = y_tr - val_model.predict(X_tr)
            n_tr = len(y_tr)
            s_err_tr = np.sqrt(np.sum(residuals_tr**2) / max(1, n_tr - 2))
            x_mean_tr = np.mean(X_tr)
            ss_x_tr = np.sum((X_tr - x_mean_tr)**2)
            t_crit_tr = stats.t.ppf(0.975, df=max(1, n_tr - 2))
            
            ci_covered = []
            for yr, actual, pred in zip(X_te.flatten(), y_te, y_pred_te):
                se_pred = s_err_tr * np.sqrt(1 + (1 / n_tr) + ((yr - x_mean_tr)**2 / max(ss_x_tr, 1e-4)))
                ci_l = max(0.0, pred - t_crit_tr * se_pred)
                ci_u = min(100.0, pred + t_crit_tr * se_pred)
                ci_covered.append(ci_l <= actual <= ci_u)
                
                # Tier classification
                act_tier = classify_score(actual)["Risk_Category"]
                prd_tier = classify_score(pred)["Risk_Category"]
                tier_actuals.append(act_tier)
                tier_preds.append(prd_tier)
                
                all_y_true_holdout.append(actual)
                all_y_pred_holdout.append(pred)
                all_y_naive_holdout.append(y_tr[-1])
            
            rmse_val = np.sqrt(mean_squared_error(y_te, y_pred_te))
            mae_val = mean_absolute_error(y_te, y_pred_te)
            mape_val = np.mean(np.abs((y_te - y_pred_te) / y_te)) * 100
            
            naive_pred = np.full_like(y_te, y_tr[-1])
            rmse_naive = np.sqrt(mean_squared_error(y_te, naive_pred))
            mae_naive = mean_absolute_error(y_te, naive_pred)
            
            holdout_records.append({
                "State": state,
                "SDG": sdg,
                "Holdout_RMSE": rmse_val,
                "Holdout_MAE": mae_val,
                "Holdout_MAPE": mape_val,
                "Naive_RMSE": rmse_naive,
                "Naive_MAE": mae_naive,
                "Naive_Beaten": rmse_val <= rmse_naive,
                "CI_Coverage_Pct": np.mean(ci_covered) * 100
            })
            
            # --- 2. Full Model (2018-2023) ---
            X_f = state_full[["Year"]].values
            y_f = state_full[sdg].values
            
            full_model = LinearRegression()
            full_model.fit(X_f, y_f)
            y_pred_f = full_model.predict(X_f)
            
            r2_f = r2_score(y_f, y_pred_f)
            rmse_f = np.sqrt(mean_squared_error(y_f, y_pred_f))
            mae_f = mean_absolute_error(y_f, y_pred_f)
            mape_f = np.mean(np.abs((y_f - y_pred_f) / y_f)) * 100
            
            n_f = len(y_f)
            p_f = 1
            adj_r2_f = 1 - (1 - r2_f) * (n_f - 1) / (n_f - p_f - 1) if n_f > 2 else r2_f
            
            # Pearson correlation & p-value
            pearson_r, p_val = stats.pearsonr(X_f.flatten(), y_f)
            
            # Mean Bias Error (MBE)
            mbe = np.mean(y_pred_f - y_f)
            
            full_records.append({
                "State": state,
                "SDG": sdg,
                "Slope_Annual_Rate": full_model.coef_[0],
                "R2": r2_f,
                "Adj_R2": adj_r2_f,
                "RMSE": rmse_f,
                "MAE": mae_f,
                "MAPE": mape_f,
                "Pearson_r": pearson_r,
                "p_value": p_val,
                "MBE": mbe
            })
            
    df_holdout = pd.DataFrame(holdout_records)
    df_full = pd.DataFrame(full_records)
    
    # Global regression evaluation on holdout test set (222 test points: 37 entities x 3 SDGs x 2 years)
    all_y_true = np.array(all_y_true_holdout)
    all_y_pred = np.array(all_y_pred_holdout)
    all_y_naive = np.array(all_y_naive_holdout)
    
    global_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    global_mae = mean_absolute_error(all_y_true, all_y_pred)
    global_mape = np.mean(np.abs((all_y_true - all_y_pred) / all_y_true)) * 100
    global_r2 = r2_score(all_y_true, all_y_pred)
    global_max_err = np.max(np.abs(all_y_true - all_y_pred))
    
    naive_global_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_naive))
    naive_global_mae = mean_absolute_error(all_y_true, all_y_naive)
    naive_global_mape = np.mean(np.abs((all_y_true - all_y_naive) / all_y_true)) * 100
    
    # Classification metrics
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    acc = accuracy_score(tier_actuals, tier_preds)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(tier_actuals, tier_preds, average='macro', zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(tier_actuals, tier_preds, average='weighted', zero_division=0)
    prec_per_c, rec_per_c, f1_per_c, sup_per_c = precision_recall_fscore_support(tier_actuals, tier_preds, labels=labels, zero_division=0)
    cm = confusion_matrix(tier_actuals, tier_preds, labels=labels)
    
    return {
        "df_holdout": df_holdout,
        "df_full": df_full,
        "global_regression": {
            "RMSE": global_rmse,
            "MAE": global_mae,
            "MAPE": global_mape,
            "R2": global_r2,
            "Max_Error": global_max_err,
            "Naive_RMSE": naive_global_rmse,
            "Naive_MAE": naive_global_mae,
            "Naive_MAPE": naive_global_mape,
            "RMSE_Improvement_pct": (naive_global_rmse - global_rmse) / naive_global_rmse * 100,
            "MAE_Improvement_pct": (naive_global_mae - global_mae) / naive_global_mae * 100
        },
        "classification": {
            "Accuracy": acc,
            "Macro_Precision": prec_m,
            "Macro_Recall": rec_m,
            "Macro_F1": f1_m,
            "Weighted_Precision": prec_w,
            "Weighted_Recall": rec_w,
            "Weighted_F1": f1_w,
            "Per_Class": {
                labels[i]: {
                    "Precision": prec_per_c[i],
                    "Recall": rec_per_c[i],
                    "F1_Score": f1_per_c[i],
                    "Support": int(sup_per_c[i])
                } for i in range(len(labels))
            },
            "Confusion_Matrix": cm.tolist(),
            "Labels": labels
        }
    }

if __name__ == "__main__":
    res = compute_all_metrics()
    import pprint
    print("=== GLOBAL REGRESSION METRICS (HOLDOUT 2022-2023) ===")
    pprint.pprint(res["global_regression"])
    print("\n=== CLASSIFICATION & RISK TIER METRICS ===")
    pprint.pprint(res["classification"])
    print("\n=== SDG-WISE HOLDOUT BREAKDOWN ===")
    print(res["df_holdout"].groupby("SDG")[["Holdout_RMSE", "Holdout_MAE", "Holdout_MAPE", "Naive_RMSE", "Naive_MAE", "CI_Coverage_Pct"]].mean())
    print("\n=== SDG-WISE FULL HORIZON FIT (2018-2023) ===")
    print(res["df_full"].groupby("SDG")[["R2", "Adj_R2", "RMSE", "MAE", "MAPE", "Pearson_r"]].mean())
    print("\n=== NATIONAL INDIA MODEL METRICS ===")
    print(res["df_full"][res["df_full"]["State"] == "India"])
