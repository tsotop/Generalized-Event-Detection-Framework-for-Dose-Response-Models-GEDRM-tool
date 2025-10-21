import numpy as np
# Import the contract from your framework
from gedrm.model_interface import DoseResponseModel

class SEVModel(DoseResponseModel):
    """
    Implementation of the Newcombe & Jensen (1996) SEV model.
    Model: SEV = a + b * log(Duration) + c * log(Stressor)
    """
    
    # Static data for this model
    _coeffs = {
        1: (1.0642, 0.6068, 0.7384),
        2: (1.6814, 0.4769, 0.7565),
        3: (0.7262, 0.7034, 0.7144),
        4: (3.7466, 1.0946, 0.3117),
        5: (3.4969, 1.9647, 0.2669),
        6: (4.0815, 0.7126, 0.2829),
    }

    def validate_params(self):
        """Check for the 'group' parameter."""
        if 'group' not in self.params:
            raise ValueError("SEVModel requires a 'group' parameter (1-6).")
        self.group = self.params['group']
        self.a, self.b, self.c = self._coeffs.get(self.group, self._coeffs[1])

    def compute_stressor_threshold(self, response_level: float, duration_hours: np.ndarray) -> np.ndarray:
        """
        Solves the SEV model for Stressor (SSC).
        SSC = exp((SEV - a - b*log(Duration)) / c)
        """
        # This is your compute_min_ssc logic, now generalized
        return np.exp((response_level - self.a - self.b * np.log(duration_hours)) / self.c)

    @property
    def name(self) -> str:
        return f"SEV (Group {self.group})"