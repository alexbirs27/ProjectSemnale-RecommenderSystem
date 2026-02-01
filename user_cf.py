import numpy as np
from collections import defaultdict

class UserBasedCF:
    def __init__(self, k=20):
        self.k = k

    def fit(self, ratings):
        self.user_ratings = defaultdict(dict)
        self.global_mean = np.mean([r for _, _, r in ratings])

        for user, item, rating in ratings:
            self.user_ratings[user][item] = rating

        self.user_means = {}
        for user, items in self.user_ratings.items():
            self.user_means[user] = np.mean(list(items.values()))

    def similarity(self, u1, u2):
        items1 = set(self.user_ratings[u1].keys())
        items2 = set(self.user_ratings[u2].keys())
        common = items1 & items2

        if len(common) < 3:
            return 0

        mean1 = self.user_means[u1]
        mean2 = self.user_means[u2]

        num = sum((self.user_ratings[u1][i] - mean1) * (self.user_ratings[u2][i] - mean2) for i in common)
        den1 = np.sqrt(sum((self.user_ratings[u1][i] - mean1)**2 for i in common))
        den2 = np.sqrt(sum((self.user_ratings[u2][i] - mean2)**2 for i in common))

        if den1 == 0 or den2 == 0:
            return 0
        return num / (den1 * den2)

    def predict(self, user, item):
        if user not in self.user_ratings:
            return self.global_mean

        if item in self.user_ratings[user]:
            return self.user_ratings[user][item]

        sims = []
        for other in self.user_ratings:
            if other != user and item in self.user_ratings[other]:
                sim = self.similarity(user, other)
                if sim > 0:
                    sims.append((sim, other))

        sims.sort(reverse=True)
        sims = sims[:self.k]

        if not sims:
            return self.user_means.get(user, self.global_mean)

        num = sum(sim * (self.user_ratings[other][item] - self.user_means[other]) for sim, other in sims)
        den = sum(abs(sim) for sim, _ in sims)

        if den == 0:
            return self.user_means.get(user, self.global_mean)

        return self.user_means[user] + num / den
