from setuptools import setup, find_packages

# Function to read the requirements.txt file
def read_requirements():
    with open('requirements.txt') as req_file:
        return req_file.read().splitlines()


setup(
    name='convexlocalisometry',
    version='0.1',
    packages=find_packages(),
    install_requires=read_requirements(),
    author='Samson Koelle',
    author_email='sjkoelle@gmail.com',
    description='Paper and code for Isometry Pursuit',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sjkoelle/convexlocalisometry',  # Replace with your repository URL
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.10',
)

