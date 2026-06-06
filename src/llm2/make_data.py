import random
import os

# 1. Define the 50-word vocabulary (including the period)
vocab = {
    "articles": ["the", "a", "an"],
    "subjects": ["cat", "kitten", "lion", "tiger", "dog", "puppy"],
    "verbs_pres": ["sleeps", "naps", "jumps", "rests", "is"],
    "verbs_past": ["sat", "lounged", "was", "jumped"],
    "preps": ["on", "under", "beside", "atop", "near", "by", "over", "behind"],
    "adjectives": [
        "small",
        "large",
        "big",
        "tiny",
        "soft",
        "fluffy",
        "sleepy",
        "lazy",
        "orange",
        "black",
        "white",
        "gray",
    ],
    "conjunctions": ["and", "but", "while", "then"],
    "punctuation": ["."],
}


def get_article(next_word):
    """Ensures grammar: 'an' before 'orange', 'a' otherwise."""
    if next_word == "orange":
        return "an"
    return random.choice(["the", "a"])


def generate_simple_sentence():
    """Builds: [Article] [Adj] [Subject] [Verb] [Prep] [Article] [Adj] [Object]."""
    # Choose core components
    adj1 = random.choice(vocab["adjectives"])
    subj = random.choice(vocab["subjects"])
    verb = random.choice(vocab["verbs_pres"] + vocab["verbs_past"])
    prep = random.choice(vocab["preps"])
    adj2 = random.choice(vocab["adjectives"])
    obj = random.choice(["mat", "rug", "carpet", "floor", "sofa", "couch", "chair"])

    # Construct with 'a/an' logic
    art1 = get_article(adj1)
    art2 = get_article(adj2)

    return f"{art1} {adj1} {subj} {verb} {prep} {art2} {adj2} {obj}."


def generate_complex_sentence():
    """Builds: While [Sentence], [Sentence] (but without the comma to stay in vocab)."""
    s1 = generate_simple_sentence().replace(".", "")
    s2 = generate_simple_sentence().replace(".", "")
    conj = random.choice(["and", "but", "then"])
    return f"{s1} {conj} {s2}."


# 2. File Generation Loop
target_size_mb = 2
target_size_bytes = target_size_mb * 1024 * 1024
filename = "universe_50_data.txt"

print(f"Generating {target_size_mb}MB of feline training data...")

with open(filename, "w") as f:
    current_size = 0
    while current_size < target_size_bytes:
        # Mix simple and complex sentences
        if random.random() > 0.7:
            sentence = generate_complex_sentence()
        else:
            sentence = generate_simple_sentence()

        f.write(sentence + " ")
        current_size = os.path.getsize(filename)

print(f"Done! Created '{filename}' ({current_size / (1024 * 1024):.2f} MB)")
