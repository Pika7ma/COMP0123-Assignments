from LinearRegression import LinearRegression

if __name__ == '__main__':
    lr = LinearRegression(12345, 1e-2)
    # lr.cv([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2], 2)
    lr.fit(1e-6, 2)