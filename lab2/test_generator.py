if __name__ == "__main__":
    powers = range(1, 11)
    func = "invert_matrix"

    # A test(ref, matrix_size) function is required in the testing module
    # Pass "self" as the ref

    for power in powers:
        print(f"def test_{2 ** power}x{2 ** power}(self):\ntest(self, {2 ** power})\n")
