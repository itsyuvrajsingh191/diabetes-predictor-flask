import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "diabetes_model.pkl"
)
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
METADATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "model_metadata.json"
)

FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

FEATURE_DISPLAY = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose Level",
    "BloodPressure": "Blood Pressure",
    "SkinThickness": "Skin Thickness",
    "Insulin": "Insulin",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Diabetes Pedigree",
    "Age": "Age",
}

NORMAL_RANGES = {
    "Glucose": {"low": 70, "normal": 99, "prediab": 125, "unit": "mg/dL"},
    "BloodPressure": {"low": 60, "normal": 79, "prediab": 89, "unit": "mmHg"},
    "BMI": {"low": 18.5, "normal": 24.9, "prediab": 29.9, "unit": "kg/m²"},
    "Insulin": {"low": 16, "normal": 166, "prediab": 200, "unit": "μU/mL"},
    "Age": {"low": 0, "normal": 44, "prediab": 60, "unit": "years"},
}


class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_importances = {}
        self.model_metrics = {}
        self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self._load_metadata()
            if not self.model_metrics or not self.feature_importances:
                self._rebuild_metadata_from_loaded_model()
                self._save_metadata()
            print("[GlucoSense] Model loaded from disk.")
        else:
            print("[GlucoSense] Training new model...")
            self._train()

    def _get_dataset(self):
        """
        Load PIMA Indians Diabetes Dataset.
        Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
        Save as: data/diabetes.csv
        Falls back to synthetic data if not found.
        """
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "diabetes.csv"
        )
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print(f"[GlucoSense] Loaded Kaggle dataset: {len(df)} rows")
        else:
            print("[GlucoSense] Kaggle dataset not found — generating synthetic data.")
            print(
                "[GlucoSense] Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"
            )
            np.random.seed(42)
            n = 768
            df = pd.DataFrame(
                {
                    "Pregnancies": np.random.poisson(3, n),
                    "Glucose": np.concatenate(
                        [np.random.normal(100, 15, 500), np.random.normal(150, 25, 268)]
                    ),
                    "BloodPressure": np.random.normal(72, 12, n),
                    "SkinThickness": np.random.normal(26, 10, n),
                    "Insulin": np.random.exponential(80, n),
                    "BMI": np.concatenate(
                        [np.random.normal(27, 4, 500), np.random.normal(35, 6, 268)]
                    ),
                    "DiabetesPedigreeFunction": np.random.exponential(0.45, n),
                    "Age": np.concatenate(
                        [np.random.normal(30, 8, 500), np.random.normal(40, 12, 268)]
                    ),
                }
            )
            df = df.abs()
            glucose_norm = (df["Glucose"] - df["Glucose"].min()) / (
                df["Glucose"].max() - df["Glucose"].min()
            )
            bmi_norm = (df["BMI"] - df["BMI"].min()) / (
                df["BMI"].max() - df["BMI"].min()
            )
            prob = 0.15 + 0.5 * glucose_norm + 0.3 * bmi_norm
            df["Outcome"] = (np.random.random(n) < prob).astype(int)
        return df

    def _prepare_dataset(self, df):
        zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in zero_cols:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
                df[col] = df[col].fillna(df[col].median())
        return df[FEATURE_COLS], df["Outcome"]

    def _load_metadata(self):
        if not os.path.exists(METADATA_PATH):
            return
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.model_metrics = payload.get("metrics") or {}
            self.feature_importances = payload.get("feature_importances") or {}
        except Exception:
            self.model_metrics = {}
            self.feature_importances = {}

    def _save_metadata(self):
        payload = {
            "metrics": self.model_metrics,
            "feature_importances": self.feature_importances,
        }
        os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _rebuild_metadata_from_loaded_model(self):
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances = dict(
                zip(
                    FEATURE_COLS,
                    [round(float(x), 4) for x in self.model.feature_importances_],
                )
            )

        try:
            df = self._get_dataset()
            X, y = self._prepare_dataset(df)

            X_train, X_test, _, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            X_test_scaled = self.scaler.transform(X_test)
            y_pred = self.model.predict(X_test_scaled)
            y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            cv_scores = cross_val_score(
                self.model, self.scaler.transform(X), y, cv=5, scoring="accuracy"
            )

            self.model_metrics = {
                "accuracy": round(float(acc * 100), 2),
                "roc_auc": round(float(auc), 4),
                "cv_mean": round(float(cv_scores.mean() * 100), 2),
                "cv_std": round(float(cv_scores.std() * 100), 2),
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                "algorithm": "Random Forest (200 trees)",
            }
            print("[GlucoSense] Model metadata rebuilt.")
        except Exception:
            pass

    def _train(self):
        df = self._get_dataset()
        X, y = self._prepare_dataset(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest (best for PIMA)
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cv_scores = cross_val_score(
            self.model, self.scaler.transform(X), y, cv=5, scoring="accuracy"
        )

        self.model_metrics = {
            "accuracy": round(float(acc * 100), 2),
            "roc_auc": round(float(auc), 4),
            "cv_mean": round(float(cv_scores.mean() * 100), 2),
            "cv_std": round(float(cv_scores.std() * 100), 2),
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "algorithm": "Random Forest (200 trees)",
        }

        self.feature_importances = dict(
            zip(
                FEATURE_COLS,
                [round(float(x), 4) for x in self.model.feature_importances_],
            )
        )

        # Save models
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        self._save_metadata()
        print(
            f"[GlucoSense] Model trained — Accuracy: {acc * 100:.1f}%, AUC: {auc:.4f}"
        )

    def predict(self, data: dict) -> dict:
        # Build feature vector
        features = {
            "Pregnancies": float(data.get("Pregnancies", data.get("preg", 0))),
            "Glucose": float(data.get("Glucose", data.get("glucose", 100))),
            "BloodPressure": float(data.get("BloodPressure", data.get("bp", 72))),
            "SkinThickness": float(data.get("SkinThickness", data.get("skin", 20))),
            "Insulin": float(data.get("Insulin", data.get("insulin", 80))),
            "BMI": float(data.get("BMI", data.get("bmi", 25))),
            "DiabetesPedigreeFunction": float(
                data.get("DiabetesPedigreeFunction", data.get("dpf", 0.35))
            ),
            "Age": float(data.get("Age", data.get("age", 30))),
        }

        X = np.array([[features[f] for f in FEATURE_COLS]])
        X_scaled = self.scaler.transform(X)

        prob = self.model.predict_proba(X_scaled)[0][1]
        risk_score = round(prob * 100, 1)

        if risk_score < 30:
            risk_level = "Low"
            risk_color = "#00C896"
        elif risk_score < 60:
            risk_level = "Moderate"
            risk_color = "#FFB830"
        else:
            risk_level = "High"
            risk_color = "#FF6B6B"

        # Per-feature contribution (manual SHAP-like via permutation)
        contributions = self._compute_contributions(X_scaled, features)

        # What-if scenarios
        whatif = self._compute_whatif(features, prob)

        # Status for each biomarker
        statuses = self._compute_statuses(features)

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "probability": round(prob, 4),
            "features": features,
            "contributions": contributions,
            "whatif": whatif,
            "statuses": statuses,
            "timestamp": pd.Timestamp.now().isoformat(),
            "model_accuracy": self.model_metrics.get("accuracy"),
        }

    def _compute_contributions(self, X_scaled, features):
        """Permutation-based feature importance for this sample."""
        base_prob = self.model.predict_proba(X_scaled)[0][1]
        contribs = {}
        for i, feat in enumerate(FEATURE_COLS):
            X_perm = X_scaled.copy()
            X_perm[0, i] = 0  # zero out (mean in scaled space)
            perm_prob = self.model.predict_proba(X_perm)[0][1]
            contribs[feat] = round((base_prob - perm_prob) * 100, 2)
        return contribs

    def _compute_whatif(self, features, base_prob):
        scenarios = []
        for feat, delta, label in [
            ("Glucose", -20, "Reduce glucose by 20 mg/dL"),
            ("BMI", -2, "Lower BMI by 2 kg/m²"),
            ("BloodPressure", -10, "Reduce BP by 10 mmHg"),
            ("Insulin", -30, "Normalize insulin by 30 μU/mL"),
            ("Age", 0, "Age (fixed)"),
        ]:
            if delta == 0:
                continue
            mod_features = features.copy()
            mod_features[feat] = max(0, features[feat] + delta)
            X_mod = np.array([[mod_features[f] for f in FEATURE_COLS]])
            X_mod_scaled = self.scaler.transform(X_mod)
            new_prob = self.model.predict_proba(X_mod_scaled)[0][1]
            scenarios.append(
                {
                    "label": label,
                    "original_score": round(base_prob * 100, 1),
                    "new_score": round(new_prob * 100, 1),
                    "delta": round((base_prob - new_prob) * 100, 1),
                }
            )
        return scenarios

    def _compute_statuses(self, features):
        statuses = {}
        for feat, ranges in NORMAL_RANGES.items():
            val = features.get(feat, 0)
            if val <= ranges["normal"]:
                status = "Normal"
                color = "#00C896"
            elif val <= ranges["prediab"]:
                status = "Borderline"
                color = "#FFB830"
            else:
                status = "High"
                color = "#FF6B6B"
            statuses[feat] = {
                "value": val,
                "status": status,
                "color": color,
                "unit": ranges["unit"],
            }
        return statuses

    def get_model_info(self):
        return {
            "metrics": self.model_metrics,
            "feature_importances": self.feature_importances,
            "algorithm": "Random Forest Classifier",
            "dataset": "PIMA Indians Diabetes Database (Kaggle)",
            "features": FEATURE_DISPLAY,
        }
