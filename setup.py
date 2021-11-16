from setuptools import setup
from setuptools import find_packages
with open("readme.md","r") as fh:
    long_description = fh.read()
setup(
    name='NICEx',
    version = '0.1.1',
    description ='Nearest Instance Counterfactual explanations',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url ='https://github.com/DBrughmans/NICE',
    author ='Dieter Brughmans',
    author_email='dieter.brughmans@uantwerpen.be',
    packages = find_packages(),
    license = 'Apache 2.0',
    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3',
    ],
    keywords = 'Counterfactual Explanations XAI',
    install_requires = [
        'numpy>=1.16.2, <2.0.0',
        'pandas>=0.23.3, <2.0.0',
        'scikit-learn>=0.20.2, <0.25.0',
        'tensorflow>=2.0.0, <2.5.0',
    ],
    python_requires= '>=3.6'
)