from compass.testgroup import TestGroup

from compass.ocean.tests.hurricane.mesh import Mesh
from compass.ocean.tests.hurricane.init import Init
from compass.ocean.tests.hurricane.forward import Forward


class Hurricane(TestGroup):
    """
    A test group for hurricane simulations with MPAS-Ocean
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.ocean.Ocean
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core,
                         name='hurricane')

        storm = 'sandy'
        mesh_name = 'DEQU120at30cr10rr2'
        mesh = Mesh(test_group=self, mesh_name=mesh_name)
        self.add_test_case(mesh)

        init = Init(test_group=self, mesh=mesh, storm=storm)
        self.add_test_case(init)

        self.add_test_case(Forward(test_group=self,
                                   mesh=mesh,
                                   storm=storm,
                                   init=init))
