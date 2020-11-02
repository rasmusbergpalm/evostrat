import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='evostrat',
    version='1.0.0',
    author="Rasmus Berg Palm",
    author_email="rasmusbergpalm@gmail.com",
    description="A library that makes Evolutionary Strategies (ES) simple to use.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rasmusbergpalm/evostrat",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.6.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
