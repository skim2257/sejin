import setuptools

# with open("README.md", "r") as rm:
#     long_description = rm.read()

setuptools.setup(
    name="sejin",
    packages=setuptools.find_packages(),
    version="0.0.1",
    author="Sekim Jin",
    author_email="hello@sejin.kim",
    description="Personal utils and stuff",
#     long_description=long_description,
#     long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.4',
)
