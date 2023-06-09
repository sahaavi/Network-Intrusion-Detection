from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements] # Replace the \n with blank

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements


setup(
    name='network_intrusion_detection',
    version='0.0.1',
    author='Avishek, Noman and Vimal',
    author_email='avisheksaha123@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    license='MIT',
    description="An Anomaly-based intrusion detection system using Deep Learning",
    url='https://github.com/sahaavi/Network-Intrusion-Detection'
)