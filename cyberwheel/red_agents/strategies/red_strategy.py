from abc import ABC, abstractmethod
from cyberwheel.network.host import Host


class RedStrategy(ABC):
    """
    Defines Base class for implementing Red Strategies.
    """
    @classmethod
    @abstractmethod
    def select_target(cls, agent_obj) -> Host | None:
        return None

    @classmethod
    @abstractmethod
    def get_reward_map(cls, config) -> dict[str, tuple[int, int]]:
        return {k: (v["reward"]["immediate"], v["reward"]["recurring"]) for k, v in config['actions'].items()}
