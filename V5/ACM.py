def sum_of_digits(n):
    return sum(int(char) for char in n)


def process_number(n):
    # If n is a single digit, no spells are needed.
    if len(n) == 1:
        return 0

    count = 0
    while len(n) > 1:
        n = str(sum_of_digits(n))
        count += 1

    return count


# Read input
n = input().strip()

# Calculate the number of times the operation is performed
result = process_number(n)

# Output the result
print(result)