from os import path

import setuptools
from setuptools import setup

extras = {
    'test': ['pytest', 'pytest_cases'],
}
# Meta dependency groups.
extras['all'] = [item for group in extras.values() for item in group]

setup(name='cque',
      version='0.0.1',
      description='',
      long_description_content_type='text/markdown',
      long_description=open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8').read(),
      url='https://github.com/koulanurag/cque',
      author='Anurag Koul',
      author_email='koulanurag@gmail.com',
      license=open(path.join(path.abspath(path.dirname(__file__)), 'LICENSE'), encoding='utf-8').read(),
      packages=setuptools.find_packages(),
      install_requires=['wheel',
                        'wandb>=0.10',
                        'gym>=0.17.0',
                        'torch>=1.8.0',
                        'dm-control==0.0.364896371',
                        'pandas==1.3.5',
                        'plotly==5.5.0',
                        # 'policybazaar @ git+https://github.com/koulanurag/policybazaar@main#egg=policybazaar'
                        ],
      extras_require=extras,
      tests_require=extras['test'],
      python_requires='>=3.7',
      classifiers=[
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
      ],
      )
