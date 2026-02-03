from collections import defaultdict

class BaselinePredictor:
    def __init__(self):
        self.global_mean = 0.0
        self.user_means = {}
        self.item_means = {}

    def fit(self, train_ratings):
        user_sums = defaultdict(float)
        user_counts = defaultdict(int)
        item_sums = defaultdict(float)
        item_counts = defaultdict(int)
        total = 0.0

        for user, item, rating in train_ratings:
            total += rating
            user_sums[user] += rating
            user_counts[user] += 1
            item_sums[item] += rating
            item_counts[item] += 1

        self.global_mean = total / len(train_ratings)
        self.user_means = {u: user_sums[u] / user_counts[u] for u in user_sums}
        self.item_means = {i: item_sums[i] / item_counts[i] for i in item_sums}

    def predict(self, user, item):
        user_bias = self.user_means.get(user, self.global_mean) - self.global_mean
        item_bias = self.item_means.get(item, self.global_mean) - self.global_mean
        pred = self.global_mean + user_bias + item_bias
        return max(0.5, min(5.0, pred))
