import numpy as np

class MatrixFactorizationSGD:
    def __init__(self, n_factors=20, lr=0.005, epochs=30, reg=0.02):
        self.n_factors = n_factors
        self.lr = lr
        self.epochs = epochs
        self.reg = reg

    def fit(self, ratings, n_users, n_items, user_to_idx, item_to_idx):
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.global_mean = np.mean([r for _, _, r in ratings])

        # latent factors
        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # biases
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)

        ratings_list = list(ratings)

        for epoch in range(self.epochs):
            np.random.shuffle(ratings_list)

            for user, item, rating in ratings_list:
                u = user_to_idx[user]
                i = item_to_idx[item]

                pred = self.global_mean + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
                error = rating - pred

                # update biases
                self.bu[u] += self.lr * (error - self.reg * self.bu[u])
                self.bi[i] += self.lr * (error - self.reg * self.bi[i])

                # update factors
                P_old = self.P[u].copy()
                self.P[u] += self.lr * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (error * P_old - self.reg * self.Q[i])

            if epoch % 5 == 0:
                train_rmse = self._compute_rmse(ratings_list, user_to_idx, item_to_idx)
                print(f"  Epoch {epoch}, RMSE: {train_rmse:.4f}")

    def _compute_rmse(self, ratings, user_to_idx, item_to_idx):
        errors = []
        for user, item, rating in ratings:
            pred = self.predict(user, item)
            errors.append((rating - pred) ** 2)
        return np.sqrt(np.mean(errors))

    def predict(self, user, item):
        if user not in self.user_to_idx or item not in self.item_to_idx:
            return self.global_mean

        u = self.user_to_idx[user]
        i = self.item_to_idx[item]

        pred = self.global_mean + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
        return np.clip(pred, 0.5, 5.0)
