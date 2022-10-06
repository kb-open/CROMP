from setuptools import setup, find_packages

setup(name='cromp', 
      #license='MIT',
      #author='Kaushik Bar', author_email='kb.opendev@gmail.com', 
      #url='https://github.com/kb-open/cromp',
      #packages=['cromp'], package_dir={'cromp': 'pkgs'},
      packages=find_packages('pkgs'), package_dir={'': 'pkgs'},
      description="The official implementation of CROMP (Constrained Regression with Ordered and Margin-sensitive Parameters)",)

