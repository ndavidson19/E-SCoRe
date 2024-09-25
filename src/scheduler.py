# src/scheduler.py

class InverseScheduler:
    def __init__(self, initial_temperature=1.0, min_temperature=0.5, max_temperature=2.0, adjustment_factor=0.1):
        """
        Initializes the Inverse Scheduler.

        :param initial_temperature: Starting temperature value.
        :param min_temperature: Minimum allowable temperature.
        :param max_temperature: Maximum allowable temperature.
        :param adjustment_factor: Step size for temperature adjustments.
        """
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.adjustment_factor = adjustment_factor

    def update_temperature(self, success_rate):
        """
        Updates the temperature based on the success rate.

        :param success_rate: Proportion of correct answers (0 <= success_rate <= 1).
        """
        if success_rate < 0.5:
            # Increase temperature to promote diversity
            self.temperature = min(self.temperature + self.adjustment_factor, self.max_temperature)
        elif success_rate > 0.8:
            # Decrease temperature to focus on correctness
            self.temperature = max(self.temperature - self.adjustment_factor, self.min_temperature)
        # Else, keep the temperature unchanged
        return self.temperature
