import sys
from data import load_data
from models import BaselinePredictor, MatrixFactorizationSVD, MatrixFactorizationSGD

def main():
    print("Loading data...")
    train, test, user_to_idx, item_to_idx, n_users, n_items = load_data()

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    # train models
    print("Training models...\n")

    baseline = BaselinePredictor()
    baseline.fit(train)

    mf_svd = MatrixFactorizationSVD(n_factors=50)
    mf_svd.fit(train, n_users, n_items, user_to_idx, item_to_idx)

    mf_sgd = MatrixFactorizationSGD(n_factors=20, lr=0.005, epochs=30, reg=0.02)
    mf_sgd.fit(train, n_users, n_items, user_to_idx, item_to_idx)

    # predictions
    print("\n" + "="*60)
    print(f"{'User':<8} {'Item':<8} {'Baseline':<10} {'SVD':<10} {'SGD':<10} {'Actual':<10}")
    print("="*60)

    for i in range(n):
        user, item, actual = test[i]
        pred_base = baseline.predict(user, item)
        pred_svd = mf_svd.predict(user, item)
        pred_sgd = mf_sgd.predict(user, item)
        print(f"{user:<8} {item:<8} {pred_base:<10.2f} {pred_svd:<10.2f} {pred_sgd:<10.2f} {actual:<10.2f}")

    print("="*60)

if __name__ == "__main__":
    main()
