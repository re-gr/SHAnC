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


SiO2_last_timestep = (impresources.files(SiO2_files) / "last_timestep.lammpstrj").__str__()
SiO2_last_t = (impresources.files(SiO2_files) / "last_timestep.lammpstrj").__str__()
SiO2_lt = (impresources.files(SiO2_files) / "last_timestep.lammpstrj").__str__()

SiO2_helix_data = (impresources.files(SiO2_files) / "helix.data").__str__()
SiO2_helix = (impresources.files(SiO2_files) / "helix.data").__str__()

Au_twist_last_timestep = (impresources.files(Au_files) / "au_twist_relaxed.lammpstrj").__str__()
Au_twist_last_t = (impresources.files(Au_files) / "au_twist_relaxed.lammpstrj").__str__()
Au_twist_lt = (impresources.files(Au_files) / "au_twist_relaxed.lammpstrj").__str__()

Au_helix_last_timestep = (impresources.files(Au_files) / "au_helix_relaxed.lammpstrj").__str__()
Au_helix_last_t = (impresources.files(Au_files) / "au_helix_relaxed.lammpstrj").__str__()
Au_helix_lt = (impresources.files(Au_files) / "au_helix_relaxed.lammpstrj").__str__()

Au_twist_data = (impresources.files(Au_files) / "au_twist.data").__str__()
Au_helix_data = (impresources.files(Au_files) / "au_helix.data").__str__()
Au_twist= (impresources.files(Au_files) / "au_twist.data").__str__()
Au_helix = (impresources.files(Au_files) / "au_helix.data").__str__()
