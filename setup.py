from os import path

import setuptools
from setuptools import setup

extras = {
    'test': ['pytest', 'pytest_cases', 'pytest-cov'],
    'dev': ['pandas==1.3.5', 'plotly==5.5.0', 'wandb'],
    'docs': ['sphinx', 'furo', 'sphinxcontrib-katex',
             'sphinx-copybutton', 'sphinx_design', 'myst-parser']
}
# Meta dependency groups.
extras['all'] = [item for group in extras.values() for item in group]

setup(name='opcc',
      version='0.0.1',
      description="It's a benchmark comprising queries to evaluate "
                  "uncertainty estimation in offline reinforcement learning.",
      long_description_content_type='text/markdown',
      long_description=open(path.join(path.abspath(path.dirname(__file__)),
                                      'README.md'), encoding='utf-8').read(),
      url='https://github.com/koulanurag/opcc',
      author='Anurag Koul',
      author_email='koulanurag@gmail.com',
      license=open(path.join(path.abspath(path.dirname(__file__)),
                             'LICENSE'), encoding='utf-8').read(),
      packages=setuptools.find_packages(),
      install_requires=['absl-py==1.0.0',
                        'numpy==1.21.5',
                        'scikit-learn',
                        'gym==0.21.0',
                        # We use a fork of d4rl to resolve dm_control install
                        # issues arising due to lack of versioning across d4rl
                        # and dm_control packages.
                        # 'd4rl @ git+https://github.com/koulanurag/d4rl@master#egg=d4rl'
                        'd4rl @ git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl',
                        'cython<3',
                        'entry_points<5'
                        ],
      include_package_data=True,
      extras_require=extras,
      tests_require=extras['test'],
      python_requires='>=3.7',
      classifiers=['Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8'])
