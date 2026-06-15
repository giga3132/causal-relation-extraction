from statistics import median


def _tokenize_entity_span(span):
    return span.strip().split()


def entity_span_lengths(sentence):
    e1_start = sentence.index("<e1>") + len("<e1>")
    e1_end = sentence.index("</e1>")
    e2_start = sentence.index("<e2>") + len("<e2>")
    e2_end = sentence.index("</e2>")

    e1_len = len(_tokenize_entity_span(sentence[e1_start:e1_end]))
    e2_len = len(_tokenize_entity_span(sentence[e2_start:e2_end]))
    return e1_len, e2_len


def entity_span_score(sentence, metric="max"):
    e1_len, e2_len = entity_span_lengths(sentence)
    if metric == "max":
        return max(e1_len, e2_len)
    if metric == "sum":
        return e1_len + e2_len
    if metric == "mean":
        return (e1_len + e2_len) / 2
    raise ValueError(f"Unknown span metric: {metric}")


def split_by_entity_span_length(dataset, threshold=None, metric="max"):
    scores = [entity_span_score(sentence, metric=metric) for sentence in dataset["sentence"]]
    if threshold is None:
        threshold = median(scores)

    short_indexes = [idx for idx, score in enumerate(scores) if score <= threshold]
    long_indexes = [idx for idx, score in enumerate(scores) if score > threshold]

    return {
        "short": dataset.select(short_indexes),
        "long": dataset.select(long_indexes),
    }, threshold, {
        "short": len(short_indexes),
        "long": len(long_indexes),
        "metric": metric,
    }
