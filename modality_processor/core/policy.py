from typing import List, Dict, Any


class PolicyManager:
    """
    Manages the data processing policy, such as which modalities are enabled.
    """

    def __init__(self, policy: Dict[str, Any] = None):  # type: ignore
        """
        Initializes the PolicyManager.

        Args:
            policy: A dictionary defining the policy. For now, it's expected
                    to have a key 'enabled_modalities' with a list of strings.
                    If None, all modalities are considered enabled by default.
        """
        self.policy = policy if policy is not None else {}
        self._enabled_modalities: List[str] = (
            self.policy.get("enabled_modalities") or []
        )

    def set_policy(self, policy: Dict[str, Any]):
        """
        Updates the current policy.
        """
        self.policy = policy
        self._enabled_modalities = self.policy.get("enabled_modalities") or []

    def is_enabled(self, modality: str) -> bool:
        """
        Checks if a given modality is enabled by the current policy.

        If no 'enabled_modalities' list is defined in the policy, all
        modalities are considered enabled.

        Args:
            modality: The name of the modality to check (e.g., 'video', 'audio').

        Returns:
            True if the modality is enabled, False otherwise.
        """
        if self._enabled_modalities is None:
            return True  # Default to enabled if the list is not specified

        return modality in self._enabled_modalities
