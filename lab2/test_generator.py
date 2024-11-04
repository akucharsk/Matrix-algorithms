if __name__ == "__main__":
    sizes = range(2, 150, 12)
    func = "invert_matrix"

    # A test(ref, matrix_size) function is required in the testing module
    # Pass "self" as the ref

    for size in sizes:
        print(f"def test_{size}x{size}(self):\ntest(self, {size})\n")
