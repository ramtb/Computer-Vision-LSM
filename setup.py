from setuptools import setup, find_packages

setup(
    name="keet_database",  # Nombre de tu paquete
    version="0.1",  # Versión inicial del paquete
    packages=find_packages(where='modules'),  # Buscar paquetes en la carpeta modules
    author="Hector Gerardo Martinez Fuentes",
    author_email="guegueger@gmail.com",
)
