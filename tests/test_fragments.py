import random

from copyspan.fragments import (
    Fragment,
    fragments,
    fragments_bruteforce,
    fragments_newsroom,
    longest_matches,
)


def test_hand_worked_example():
    source = "the cat sat on the mat and the dog sat on the rug".split()
    output = "the dog sat on the mat today".split()
    # position 0: "the dog sat on the" occurs at source[7:12] (5 tokens);
    # position 5: "mat" at source[5]; "today" is not in the source.
    assert fragments(source, output) == [Fragment(0, 7, 5), Fragment(5, 5, 1)]


def test_leftmost_source_position_on_ties():
    source = "a b x a b".split()
    output = "a b".split()
    assert fragments(source, output) == [Fragment(0, 0, 2)]


def test_longest_matches_values():
    source = "a b c a b d".split()
    output = "a b d z".split()
    lengths, starts = longest_matches(source, output)
    assert lengths == [3, 2, 1, 0]
    assert starts == [3, 4, 5, -1]


def test_empty_and_one_token():
    assert fragments([], ["a"]) == []
    assert fragments(["a"], []) == []
    assert fragments(["a"], ["a"]) == [Fragment(0, 0, 1)]
    assert fragments(["a"], ["b"]) == []
    assert longest_matches([], []) == ([], [])


def test_newsroom_pseudocode_can_miss_overlapping_longer_match():
    source = "a a a b".split()
    output = "a a b".split()
    # The longest match is source[1:4]; the printed procedure jumps past
    # source[0:2] and finds it only from source[2].
    assert fragments(source, output) == [Fragment(0, 1, 3)]
    assert fragments_newsroom(source, output) == [Fragment(0, 0, 2), Fragment(2, 3, 1)]


def _random_pair(rng: random.Random):
    vocab = [str(k) for k in range(rng.randint(1, 6))]
    n = rng.randint(0, 40)
    source = [rng.choice(vocab) for _ in range(n)]
    output = []
    while len(output) < rng.randint(0, 30):
        if source and rng.random() < 0.5:
            a = rng.randrange(n)
            output.extend(source[a : a + rng.randint(1, 8)])
        else:
            output.append(rng.choice(vocab + ["new"]))
    return source, output


def test_fast_equals_bruteforce_on_random_texts():
    rng = random.Random(12345)
    for _ in range(3000):
        source, output = _random_pair(rng)
        assert fragments(source, output) == fragments_bruteforce(source, output), (source, output)


def test_fragments_are_real_copies_and_disjoint():
    rng = random.Random(7)
    for _ in range(500):
        source, output = _random_pair(rng)
        prev_end = 0
        for f in fragments(source, output):
            assert f.length > 0
            assert f.out_start >= prev_end
            seg = output[f.out_start : f.out_start + f.length]
            assert seg == source[f.src_start : f.src_start + f.length]
            prev_end = f.out_start + f.length


def test_newsroom_fragments_are_real_copies():
    rng = random.Random(99)
    for _ in range(500):
        source, output = _random_pair(rng)
        for f in fragments_newsroom(source, output):
            seg = output[f.out_start : f.out_start + f.length]
            assert seg == source[f.src_start : f.src_start + f.length]


def test_longest_matches_against_direct_search():
    rng = random.Random(3)
    for _ in range(1000):
        source, output = _random_pair(rng)
        lengths, starts = longest_matches(source, output)
        for i in range(len(output)):
            best, best_j = 0, -1
            for j in range(len(source)):
                k = 0
                limit = min(len(output) - i, len(source) - j)
                while k < limit and output[i + k] == source[j + k]:
                    k += 1
                if k > best:
                    best, best_j = k, j
            assert lengths[i] == best
            assert starts[i] == best_j


def test_to_dict():
    assert Fragment(1, 2, 3).to_dict() == {"out_start": 1, "src_start": 2, "length": 3}
