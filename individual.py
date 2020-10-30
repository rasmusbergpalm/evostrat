class Individual:
    """
    An individual which has a fitness.
    """

    def fitness(self) -> float:
        """
        :return: The fitness of the individual. Does NOT need to be differentiable.
        """
        raise NotImplementedError
