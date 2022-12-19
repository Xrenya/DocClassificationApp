def dot_metric(a, b):
    return -np.dot(a, b)


def recall(retrieved, relevant):
    return float(len(set(relevant) & set(retrieved))) \
        / float(len(set(relevant)))


n_points = len(train_predictions)
n_points_test = len(test_predictions)
n_trees = 10
n = 100
total_recall = []


for j in range(n_points_test):
    # create random points at distance x
    f = 256
    idx = AnnoyIndex(f, 'dot')

    embed = test_predictions[j]

    expected_results = [
        sorted(
            range(n_points),
            key=lambda j: dot_metric(train_predictions[i], embed)
        )[:n]
        for i in range(n_points)
    ]

    for i, vec in enumerate(train_predictions):
        idx.add_item(i, vec)

    idx.build(n_trees)

    nns = idx.get_nns_by_vector(embed, n)
    total_recall.append(recall(nns, expected_results[0]))
