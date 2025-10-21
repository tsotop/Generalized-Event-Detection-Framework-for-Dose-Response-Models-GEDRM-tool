from abc import ABC, abstractmethod
import numpy as np

class DoseResponseModel(ABC):
    """
    Abstract Base Class for a dose-response model.
    
    A model must be able to compute the stressor threshold
    given a desired response level and an exposure duration.
    """
    
    def __init__(self, **model_params):
        """
        Stores any model-specific parameters (e.g., fish group, coefficients).
        """
        self.params = model_params
        self.validate_params()

    @abstractmethod
    def validate_params(self):
        """
        Check if all required parameters (e.g., 'group' for SEV) 
        were provided in self.params.
        """
        pass

    @abstractmethod
    def compute_stressor_threshold(self, response_level: float, duration_hours: float) -> float:
        """
        Inverts the dose-response model to solve for stressor magnitude.
        
        Args:
            response_level (float): The target response effect (e.g., SEV 5).
            duration_hours (float): The exposure duration in hours.
            
        Returns:
            float: The minimum continuous stressor magnitude required to
                   produce the response_level over the duration_hours.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """A short name for the model (e.g., 'SEV Group 1')."""
        pass