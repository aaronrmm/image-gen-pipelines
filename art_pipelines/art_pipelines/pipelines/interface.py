import abc


class AbstractPipeline(abc.ABC):
    @abc.abstractmethod
    def transform(self, input_path, output_path):
        """
        Loads a resource from input_path
        Performs one or more transformations on the resource
        Saves the result to output_path
        """
