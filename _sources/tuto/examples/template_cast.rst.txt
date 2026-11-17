TEMPLATE WITH CAST
==================

The following script is a template for a LAMMPS input file. For more details on the procedure look into :doc:`here<../lammps>`.
And for more details on the LAMMPS input file keywords, look into the `LAMMPS manual <https://docs.lammps.org/Manual.html>`_.
In the template, one needs to modify the terms marked with a $ with no bracket (i.e. $N_TIMESTEPS_RELAXATION).

    units metal
    boundary p p p
    dimension 3

    atom_style full

    #INITIAL RELAXATION TIMESTEPS
    variable step_calc_rel equal $N_TIMESTEPS_RELAXATION
    variable dt_rel equal $DT_RELAXATION

    #DYNAMIC TIMESTEPS
    variable step_calc_dyn equal $N_TIMESTEPS_DYNAMIC
    variable dt_dyn equal $DT_DYNAMIC

    #TEMPERATURE GRADIENT
    variable T_init equal $TEMP_INIT
    variable T_final equal $TEMP_FINAL

    #DUMP VARIABLES
    variable dump_write string $DUMP_FILE_NAME
    variable dump_dt equal $N_TIMESTEPS_DUMP

    ### READ FILES
    read_data $INPUT_FILE extra/atom/types 2
    read_data $INPUT_FILE_CAST add append offset $N_TYPES 0 0 0 0

    #MASSES
    mass 1 $MASS
    ...

    #CHARGES
    set type 1 charge $CHARGE
    ...

    #FORCE FIELD
    pair_style $PAIR_STYLE
    pair_coeff $PAIR_COEFF

    #IF SYSTEM IS SATURATED, BONDS OH
    bond_style harmonic
    bond_coeff 1 23.98 0.95

    neighbor 2.0 bin
    kspace_style pppm 1.0e-4

    #THE DUMP, THE FILE GENERATED CONTAINING THE POSITION OF THE ATOMS
    dump dummp all atom 1 ${dump_write}
    dump_modify dummp sort id

    #GROUP ATOMS TOGETHER
    group $NAME_GROUP_1 type $TYPES
    ...

    #CREATE VELOCITY FOR CERTAIN ATOMS
    velocity $NAME_GROUP_1 create $TEMPERATURE 123

    #FIRST MINIMIZATION OF ENERGY
    minimize $ETOL $FTOL $MAXITER $MAXEVAL

    dump_modify dummp every ${dump_dt}

    #SLOW GRADIENT FROM T_init to T_final
    fix T $NAME_GROUP_1 nvt temp ${T_init} ${T_FINAL} 0.05

    #IF THERE ARE HYDROGENS DONT DO THE GRADIENT ON THEM
    fix Th $NAME_GROUP_H nvt temp ${T_init} ${T_init} 0.05

    thermo_style custom step temp press etotal vol lx ly lz
    thermo 100
    timestep ${dt_rel}

    run ${step_calc_rel}

    velocity sio create ${T_final} 321

    #DYNAMIC AT HIGH TEMPERATURE TO RELAX THE SILICA ON THE CAST
    fix T $NAME_GROUP_1 nvt temp ${T_final} ${T_final} 0.05
    fix Th $NAME_GROUP_H nvt temp ${T_init} ${T_init} 0.05
    thermo_style custom step temp press etotal vol lx ly lz
    thermo 100
    timestep ${dt_dyn}

    run ${step_calc_dyn}

    #GET BACK TO THE INITIAL TEMPERATURE TO REMOVE THE CAST
    velocity sio create ${T_final} 213

    fix T $NAME_GROUP_1 nvt temp ${T_final} ${T_init} 0.05
    fix Th $NAME_GROUP_H nvt temp ${T_init} ${T_init} 0.05
    thermo_style custom step temp press etotal vol lx ly lz
    thermo 100
    timestep ${dt_rel}

    run ${step_calc_rel}

    #REMOVE THE CAST
    #GROUP THE ATOMS OF THE CAST
    group del_pot type $CAST_TYPE
    ...
    delete_atoms group del_pot

    #REDO THE GRADIENT

    velocity $NAME_GROUP_1 create ${T_init} 312

    unfix T
    fix T $NAME_GROUP_1 nvt temp ${T_init} ${T_final} 0.05
    thermo_style custom step temp press etotal vol lx ly lz
    thermo 100
    timestep ${dt_rel}

    run ${step_calc_rel}

    #SECOND DYNAMIC TO RELAX
    velocity $NAME_GROUP_1 create ${T_final} 1023

    unfix T
    fix T $NAME_GROUP_1 nvt temp ${T_final} ${T_final} 0.1
    thermo_style custom step temp press etotal vol lx ly lz
    thermo 100
    timestep ${dt_dyn}

    run ${step_calc_dyn}

    #SLOW RELAXATION IN NPT TO ALLOW CHANGE IN VOLUME
    velocity $NAME_GROUP_1 create ${T_final} 1203

    unfix T
    fix T $NAME_GROUP_1 npt temp ${T_final} ${T_final} 0.05 iso 1 1 0.5
    thermo_style custom step temp press etotal vol lx ly lz
    thermo 100
    timestep ${dt_rel}

    run ${step_calc_rel}

    #LONG DYNAMIC TO RELAX IN NPT
    velocity $NAME_GROUP_1 create ${T_final} 1302
    0 1
    unfix T
    fix T $NAME_GROUP_1 npt temp ${T_final} ${T_final} 0.05 iso 1 1 0.5
    thermo_style custom step temp press etotal vol lx ly lz
    thermo 100
    timestep ${dt_dyn}

    run ${step_calc_dyn}


    #FINALLY RELAX TO TEMPERATURE 0.1
    velocity $NAME_GROUP_1 create ${T_final} 1230

    unfix T
    fix T $NAME_GROUP_1 npt temp ${T_final} 0.1 0.05 iso 1 1 0.5
    thermo_style custom step temp press etotal vol lx ly lz
    thermo 100
    timestep ${dt_rel}

    run ${step_calc_2}

    #MINIMIZE THE STRUCTURE TO FINISH
    dump_modify dummp every 1
    minimize 1.0e-6 1.0e-6 1000 10000
