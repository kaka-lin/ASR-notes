from evaluate import load


def get_metric(metric_name="wer"):
    """
    Load a metric by name.

    Args:
        metric_name (str): The name of the metric to load.

    Returns:
        Metric: The loaded metric.
    """
    return load(metric_name)


def compute_metric(metric, references, predictions):
    """
    Compute the metric for given references and predictions.

    Args:
        metric (Metric): The loaded metric.
        references (list): List of reference texts.
        predictions (list): List of predicted texts.

    Returns:
        float: The computed metric value.
    """
    return metric.compute(references=references, predictions=predictions)


def main():
    references = ["the cat sat on the mat"]
    predictions = ["the cat sit on the"]

    # Word Error Rate (WER) metric
    metric_name = "wer"
    wer_metric = get_metric(metric_name)

    result = compute_metric(wer_metric, references, predictions)
    print(f"{metric_name} result: {result}")

    # Character Error Rate (CER) metric
    metric_name = "cer"
    cer_metric = get_metric(metric_name)
    # CER 預設是以包含空格的完整字串長度為母數計算
    result = compute_metric(cer_metric, references, predictions)
    print(f"{metric_name} result: {result}")


if __name__ == "__main__":
    main()
