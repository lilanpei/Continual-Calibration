from avalanche.evaluation import Metric

class ECE_metrics(Metric[float]):
    """
    This metric will return a `float` value
    """
    def __init__(self):
        """
        Initialize your metric here
        """
        super().__init__()
        pass

    def update(self):
        """
        Update metric value here
        """
        pass

    def result(self) -> float:
        """
        Emit the metric result here
        """
        return 0

    def reset(self):
        """
        Reset your metric here
        """
        pass
