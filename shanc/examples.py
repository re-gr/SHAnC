"""
Script that imports the examples from the package
"""

from importlib import resources as impresources
from shanc.demo_system import SiO2 as SiO2_files
from shanc.demo_system import Au as Au_files


SiO2 = (impresources.files(SiO2_files) / "beta_quartz.data").__str__()
quartz = (impresources.files(SiO2_files) / "beta_quartz.data").__str__()
beta_quartz = (impresources.files(SiO2_files) / "beta_quartz.data").__str__()


Au = (impresources.files(Au_files) / "Au.data").__str__()
gold = (impresources.files(Au_files) / "Au.data").__str__()
