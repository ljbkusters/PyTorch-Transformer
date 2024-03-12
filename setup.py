import setuptools


setuptools.setup(
    name="nlptorch",
    version="0.0.1",
    description="Some examples of natural language processing in pytorch",
    packages=("nlptorch",),
    requires=["torch", "torchtext", "tqdm", "torchvision", "spacy"]
)
