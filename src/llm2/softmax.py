import math


def softmax(logits):
    # Step 1: exponentiate every logit. e^x makes everything positive and amplifies the differences between big and small.
    exps = [math.exp(x) for x in logits]

    # Step 2: sum them all up. This is the denominator.
    total = sum(exps)

    # Step 3: divide each one by the total. Now they sum to 1.
    return [e / total for e in exps]


print(softmax([1.122, 1.23, 1.41]))
