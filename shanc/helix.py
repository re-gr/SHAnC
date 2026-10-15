"""
Helix object definition for SHAnC
"""


class helix(object):
    """
    the
    """
    def __init__(self, Pos, Types, Lims, Bonds=[], Angles_OH=[], Pos_cast=[], Types_cast=[], Lims_cast=[]):
        """
        helix(Pos, Types, Lims, Angles_OH, Pos_cast, Types_cast, Lims_cast)

        Constructs the helix object containing the atoms and the casts

        Attributes
        ----------
            Pos : list
                The position of the atoms
            Types : list
                The types of the atoms
            Lims : list
                The coordinate limits of the system
            Bonds : list, optional
                The list of the bonds typically used for analysis
            Angles_OH : list
                The angles of the OH bonds used to complete the system
            Pos_cast : list, optional
                The position of the atoms of the cast
            Types_cast : list, optional
                The type of the atoms of the cast
            Lims_cast : list, optional
                The coordinate limits of the cast

        Parameters
        ----------
            Pos : list
                The position of the atoms
            Types : list
                The types of the atoms
            Lims : list
                The coordinate limits of the system
            Bonds : list, optional
                The list of the bonds typically used for analysis
            Angles_OH : list
                The angles of the OH bonds used to complete the system
            Pos_cast : list, optional
                The position of the atoms of the cast
            Types_cast : list, optional
                The type of the atoms of the cast
            Lims_cast : list, optional
                The coordinate limits of the cast

        Returns
        -------
            helix : helix
                The helix object.
        """
        self.Pos = Pos
        self.Types = Types
        self.Lims = Lims
        self.Bonds = Bonds
        self.Angles_OH = Angles_OH,
        self.Pos_cast = Pos_cast
        self.Types_cast = Types_cast
        self.Lims_cast = Lims_cast
