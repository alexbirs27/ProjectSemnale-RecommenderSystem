from data import load_data
from metrics import rmse, mae
from baseline import BaselinePredictor
from user_cf import UserBasedCF

def evaluate(model, test_ratings):
    preds = []
    actuals = []
    for user, item, rating in test_ratings:
        preds.append(model.predict(user, item))
        actuals.append(rating)
    return rmse(preds, actuals), mae(preds, actuals)

def main():
    print("Loading data...")
    train, test, user_to_idx, item_to_idx, n_users, n_items = load_data()
    print(f"Train: {len(train)}, Test: {len(test)}")

    print("\nTraining Baseline...")
    baseline = BaselinePredictor()
    baseline.fit(train)
    r, m = evaluate(baseline, test)
    print(f"Baseline - RMSE: {r:.4f}, MAE: {m:.4f}")

    print("\nTraining User-Based CF...")
    user_cf = UserBasedCF(k=20)
    user_cf.fit(train)
    r, m = evaluate(user_cf, test)
    print(f"User-Based CF - RMSE: {r:.4f}, MAE: {m:.4f}")

if __name__ == "__main__":
    main()
