from setuptools import setup, find_packages
setup(
    name="modules",  # Nombre del paquete
    version="1.0",
    packages=find_packages(),  # Encuentra automáticamente los subpaquetes
    install_requires=[
      
    ],
    author="Héctor Gerardo Martínez Fuentes",
    author_email="guegueger@gmail.com",
    description="utils",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tu_usuario/tu_proyecto",  # Opcional
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)