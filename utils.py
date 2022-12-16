def split_data(X):
    l = len(X)
    train = X[0 : l // 2]
    valid = X[l // 2 : l * 3 // 4]
    test = X[l * 3 // 4 : l]
    assert l == len(train) + len(valid) + len(test)
    return train, valid, test