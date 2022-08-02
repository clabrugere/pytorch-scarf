from setuptools import find_packages, setup

setup(
    name="pytorch-scarf",
    packages=find_packages(where="scarf"),
    version="0.1",
    license="MIT",
    description="Self-supervised contrastive learning using feature corruptions on tabular data- Pytorch",
    long_description_content_type="text/markdown",
    author="Clément Labrugere",
    url="https://github.com/clabrugere/pytorch-scarf",
    keywords=[
        "artificial intelligence",
        "contrastive learning",
        "self-supervised learning",
    ],
    python_requires=">=3.7",
    install_requires=["torch==1.12", "tqdm==4.64"],
    tests_require=["pytest"],
)
