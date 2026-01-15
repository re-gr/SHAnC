#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Notes
-----

Yes
"""

import numpy as np
# import matplotlib.pyplot as plt
import os
from distorsion import *
import lammps

def create_in(file_name,template,params={"min_x":0,"max_x":10,"min_y":0,"max_y":10,"min_z":0,"max_z":10,"step_read":0,"step_calc":1000,"dump_read":"dump_in.lammpstrj","dump_write":"dump_out.lammpstrj","repl_x":1,"repl_y":1,"repl_z":1,"deformation":"deformation.data","dump_dt":100}):

    with open(file_name,"w") as file:
        for line in open(template,'r'):
            if "PY" in line:
                line = line.replace("PY_MINX",str(params["min_x"]))
                line = line.replace("PY_MINY",str(params["min_y"]))
                line = line.replace("PY_MINZ",str(params["min_z"]))
                line = line.replace("PY_MAXX",str(params["max_x"]))
                line = line.replace("PY_MAXY",str(params["max_y"]))
                line = line.replace("PY_MAXZ",str(params["max_z"]))

                line = line.replace("PY_STEP_READ",str(params["step_read"]))
                line = line.replace("PY_STEP_CALC",str(params["step_calc"]))

                line = line.replace("PY_DUMP_READ",str(params["dump_read"]))
                line = line.replace("PY_DUMP_WRITE",str(params["dump_write"]))

                line = line.replace("PY_REPLX",str(params["repl_x"]))
                line = line.replace("PY_REPLY",str(params["repl_y"]))
                line = line.replace("PY_REPLZ",str(params["repl_z"]))
                line = line.replace("PY_DEFORMATION",str(params["deformation"]))
                line = line.replace("PY_DUMP_DT",str(params["dump_dt"]))

            file.write(line)


def compil_dump(dumps,compiled_name="dumps_compiled.lammpstrj"):
    with open(compiled_name,"w") as comp:
        for file in dumps:
            dump_file = open(file)

            comp.write(dump_file.read())
            dump_file.close()


if __name__ == "__main__":
    do_create_syst = True
    do_create_relax = False
    do_compute_relax = False
    do_create_relax_after = False
    do_compute_relax_after = False

    N_step_relax = 50
    N_step_torsion = 3000

    rota = 1.0
    D = 167
    pitch = 600
    width = 200
    thickness = 110
    int_thick = 35

    D = 40
    pitch = 300
    width = 80
    thickness = 40
    int_thick = 10

    # D = 10
    # pitch = 200
    # width = 30
    # thickness = 30
    # int_thick = 5



    params = {"min_x":-57,
            "max_x":57,
            "min_y":-57,
            "max_y":57,
            "min_z":0.000000,
            "max_z":283.9432,
            "dump_dt":1}



    #DO NOT CHANGE
    params["step_read"] = 0
    params["step_calc"] = 0
    params["dump_read"] = "plcholder"
    params["dump_write"] = "plcholder"
    params["deformation"] = "plcholder"
    params["repl_x"] = 1
    params["repl_y"] = 1
    params["repl_z"] = 1

    if do_create_syst:
        create_syst(rota,D,pitch,width,thickness,int_thick)

    if do_create_relax:
        params_relax = params
        params_relax["step_calc"] = N_step_relax
        params_relax["dump_write"] = "dump_relax.lammpstrj"
        create_in("in_relax.lmp","in_template.lmp",params_relax)

    if do_compute_relax:
        lmp = lammps.lammps()
        lmp.file("in_relax.lmp")

    if do_create_relax_after:
        params_relax_after = params
        params_relax_after["step_read"] = N_step_relax
        params_relax_after["step_calc"] = N_step_relax
        params_relax_after["dump_write"] = "dump_relax_after.lammpstrj"
        params_relax_after["dump_read"] = "dump_relax.lammpstrj"
        create_in("in_relax_after.lmp","in_after_template.lmp",params_relax_after)

    if do_compute_relax_after:
        lmp = lammps.lammps()
        lmp.file("in_relax_after.lmp")
