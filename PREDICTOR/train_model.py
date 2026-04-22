"""
Run this ONCE after placing diabetes.csv in the data/ folder.
  python train_model.py
Trains the Random Forest model and saves it to models/
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from utils.predictor import DiabetesPredictor

print("=" * 50)
print("  GlucoSense AI — Model Training Script")
print("=" * 50)

# Check for dataset
data_path = os.path.join("data", "diabetes.csv")
if not os.path.exists(data_path):
    print("\n⚠  PIMA dataset not found at data/diabetes.csv")
    print("   Download it from:")
    print("   https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
    print("   Place the file at:  glucosense/data/diabetes.csv")
    print("\n   Falling back to synthetic training data for now...\n")
else:
    print(f"\n✓ Dataset found: {data_path}")

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Remove old model files so a fresh train happens
for f in ["models/diabetes_model.pkl", "models/scaler.pkl"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"  Removed old: {f}")

print("\n  Training model...\n")
predictor = DiabetesPredictor()

print("\n" + "=" * 50)
print("  Training Complete!")
print("=" * 50)
metrics = predictor.model_metrics
print(f"  Accuracy  : {metrics.get('accuracy')}%")
print(f"  ROC-AUC   : {metrics.get('roc_auc')}")
print(f"  CV Mean   : {metrics.get('cv_mean')}% ± {metrics.get('cv_std')}%")
print(f"  Algorithm : {metrics.get('algorithm')}")
print("\n  Model saved to models/")
print("  Run 'python app.py' to start the server.\n")
