import re

class ComputeModule:
    """
    A simple tool for handling numerical calculations.
    """
    def run(self, operation: str, values: list) -> float:
        """
        Performs a calculation on a list of numerical values.
        
        Args:
            operation: The operation to perform (e.g., 'sum', 'average').
            values: A list of numerical values (can be strings).
            
        Returns:
            The result of the calculation.
        """
        numeric_values = []
        for v in values:
            try:
                cleaned_v = re.sub(r'[$,]', '', str(v))
                numeric_values.append(float(cleaned_v))
            except (ValueError, TypeError):
                continue
        
        if not numeric_values:
            return 0.0

        if operation == 'sum':
            return sum(numeric_values)
        elif operation == 'average':
            return sum(numeric_values) / len(numeric_values)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
