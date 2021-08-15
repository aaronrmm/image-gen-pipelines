import os
import unittest
from functools import partial

from art_pipelines.image_generator import ImageGenerator


class MockGenerator(ImageGenerator):
    @staticmethod
    def print_args(*args, **kwargs):
        print("POSITIONAL ARGS:")
        for arg_index, arg in enumerate(args):
            print(f"{arg_index}: {arg}")
        print("KEYWORD ARGS:")
        for keyword in kwargs.keys():
            print(f"{keyword}: {kwargs[keyword]}")


class ExampleTest(unittest.TestCase):
    def test_load(self):
        print("TEST_LOAD")
        loading_func = partial(print, "MockGenerator loaded")
        generator_func = MockGenerator.print_args
        self.generator = MockGenerator(
            generator_func=generator_func, loading_func=loading_func
        )

    def test_generate_without_init(self):
        print("TEST_GEN_NO_INIT")
        output_dir = "test_output"
        loading_func = partial(print, "MockGenerator loaded")
        generator_func = MockGenerator.print_args
        self.generator = MockGenerator(
            generator_func=generator_func, loading_func=loading_func
        )
        initial_image = None
        self.generator.generate(
            generation_config={
                "text": "testing_config",
                "initial_image": initial_image,
            },
            output_dir=output_dir,
        )
        assert os.path.isdir(output_dir)
        assert os.path.isfile(os.path.join(output_dir, "config.json"))

        def test_generate_without_init(self):
            print("TEST_GEN_NO_INIT")
            output_dir = "test_output"
            loading_func = partial(print, "MockGenerator loaded")
            generator_func = MockGenerator.print_args
            self.generator = MockGenerator(
                generator_func=generator_func, loading_func=loading_func
            )
            initial_image = "./test_input/four_shapes.png"
            self.generator.generate(
                generation_config={
                    "text": "testing_config",
                    "initial_image": initial_image,
                },
                output_dir="test_output",
            )
            assert os.path.isdir(output_dir)
            assert os.path.isfile(os.path.join(output_dir, "config.json"))


if __name__ == "__main__":
    unittest.main()
