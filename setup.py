from setuptools import setup, Extension


def get_version_number():
    main_ns = {}
    for line in open('pointgroup/__init__.py', 'r').readlines():
        if not(line.find('__version__')):
            exec(line, main_ns)
            return main_ns['__version__']


setup(name='pointgroup',
      version=get_version_number(),
      description='pointgroup module',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      author='Abel Carreras',
      url='https://github.com/abelcarreras/pointgroup',
      author_email='abelcarreras83@gmail.com',
      packages=['pointgroup'],
      install_requires=['numpy'],
      license='MIT License')
