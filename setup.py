from distutils.command.build_ext import build_ext as build_ext_orig
from setuptools import find_packages, setup, Extension


class CTypesExtension(Extension): pass


class build_ext(build_ext_orig):

    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypesExtension)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + '.so'
        return super().get_ext_filename(ext_name)


setup(
      name='pytdigest',
      version='0.0.1',
      description='Python package for *fast* TDigest calculation.',
      py_modules=['pytdigest'],
      ext_modules=[CTypesExtension(
            name='tdigest',
            sources=['pytdigest/tdigest.c'],
      )],
      cmdclass={'build_ext': build_ext},
      author='Tomas Protivinsky',
      author_email='tomas.protivinsky@gmail.com',
      url='https://github.com/protivinsky/pytdigest',
      keywords='tdigest, distribution, statistics',
      python_requires=">=3.7, <4",
      install_requires=['numpy>=1.19.0', 'pandas>=1.1.0'],
      classifiers=[
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Data Scientists",
            "License :: OSI Approved :: APL",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
      ],
      package_dir={"": "pytdigest"},
      #packages=find_packages(where="pytdigest"),  # Required
      project_urls={  # Optional
            "Source": "https://github.com/protivinsky/pytdigest",
      },
)

