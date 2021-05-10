import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="convblock",
    version="0.0.1",
    author="Emelyanov Kirill",
    author_email="kirillemilio@gmail.com",
    description="Deep Neural Networks implemented via pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kirillemilio/convblock.git",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    zip_safe=False,
    platforms='any'
)
