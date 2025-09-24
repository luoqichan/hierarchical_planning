from abc import ABC, abstractmethod


class BasePlanner(ABC):
    def __init__(
        self,
    ) -> None:
        pass

    @abstractmethod
    def initial_plan(self):
        pass

    @abstractmethod
    def replan(self, observations, rewards, terminations, truncations, infos):
        pass
