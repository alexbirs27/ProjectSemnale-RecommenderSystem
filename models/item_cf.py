import numpy as np
from collections import defaultdict

class ItemBasedCF:
    def __init__(self, k=20):
        self.k = k

    def fit(self, ratings):
        self.user_ratings = defaultdict(dict)
        self.item_ratings = defaultdict(dict)
        self.global_mean = np.mean([r for _, _, r in ratings])

        for user, item, rating in ratings:
            self.user_ratings[user][item] = rating
            self.item_ratings[item][user] = rating

        self.item_means = {}
        for item, users in self.item_ratings.items():
            self.item_means[item] = np.mean(list(users.values()))

    def similarity(self, i1, i2):
        users1 = set(self.item_ratings[i1].keys())
        users2 = set(self.item_ratings[i2].keys())
        common = users1 & users2

        if len(common) < 3:
            return 0

        mean1 = self.item_means[i1]
        mean2 = self.item_means[i2]

        num = sum((self.item_ratings[i1][u] - mean1) * (self.item_ratings[i2][u] - mean2) for u in common)
        den1 = np.sqrt(sum((self.item_ratings[i1][u] - mean1)**2 for u in common))
        den2 = np.sqrt(sum((self.item_ratings[i2][u] - mean2)**2 for u in common))

        if den1 == 0 or den2 == 0:
            return 0
        return num / (den1 * den2)

    def predict(self, user, item):
        if user not in self.user_ratings:
            return self.global_mean

        if item in self.user_ratings[user]:
            return self.user_ratings[user][item]

        sims = []
        for other_item in self.user_ratings[user]:
            sim = self.similarity(item, other_item)
            if sim > 0:
                sims.append((sim, other_item))

        sims.sort(reverse=True)
        sims = sims[:self.k]

        if not sims:
            return self.global_mean

        num = sum(sim * self.user_ratings[user][other] for sim, other in sims)
        den = sum(sim for sim, _ in sims)

        if den == 0:
            return self.global_mean

        return num / den
