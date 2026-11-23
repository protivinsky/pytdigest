import os
from distutils.command.build_ext import build_ext as _build_ext_distutils

from setuptools import Extension, setup


class CTypesExtension(Extension): pass


class _build_ext_ctypes(_build_ext_distutils):

    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypesExtension)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            if os.name == 'nt':
                return ext_name + '.dll'
            else:
                return ext_name + '.so'
        return super().get_ext_filename(ext_name)


setup(
    ext_modules=[
        CTypesExtension(
            name=os.path.join("pytdigest", "tdigest"),
            sources=["pytdigest/tdigest.c"],
        )
    ],
    cmdclass={"build_ext": _build_ext_ctypes},
)

