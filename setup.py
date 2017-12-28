from setuptools import setup


PACKAGES = [
    'forest_kernels',
    'forest_kernels.tests',
]


def setup_package():
    setup(
        name="Forest Kernels",
        version='0.1.0',
        description='Kernel Functions Derived from Ensembles of Trees.',
        author='Joshua D. Loyal',
        url='https://github.com/joshloyal/ForestKernels',
        license='MIT',
        install_requires=['numpy', 'scipy', 'scikit-learn', 'joblib'],
        packages=PACKAGES
    )


if __name__ == '__main__':
    setup_package()
