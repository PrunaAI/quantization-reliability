from setuptools import setup, find_packages

setup(name='project_name',
      version='0.1',
      description='',
      author='Anonymous',
      author_email='anonymous@mail.com',
      packages=['src'],
      install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'torch', 'tqdm',
                        'sacred', 'deprecation', 'pymongo', 'pytorch-lightning>=0.9.0rc2', 'seml'],
      zip_safe=False)