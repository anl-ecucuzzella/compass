# from skopt.sampler import Grid
# from skopt.space import Space

from compass.ocean.tests.soma.analysis import Analysis
from compass.ocean.tests.soma.forward import Forward
from compass.ocean.tests.soma.initial_state import InitialState
from compass.testcase import TestCase
from compass.validate import compare_timers, compare_variables


class SomaEnsemble(TestCase):
    """
    The class for all test cases in the SOMA test group.  The test case creates
    the mesh and initial condition, then performs a forward run with analysis
    members.  An analysis step is included if the simulation includes
    particles.

    Attributes
    ----------
    resolution : str
        The resolution of the test case

    with_particles : bool
        Whether particles are include in the simulation

    with_surface_restoring : bool
        Whether surface restoring is included in the simulation

    long : bool
        Whether to run a long (3-year) simulation to quasi-equilibrium

    three_layer : bool
        Whether to use only 3 vertical layers and no continental shelf
    """

    def __init__(self, test_group, resolution, with_particles,
                 with_surface_restoring, long, three_layer):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.soma.Soma
            The test group that this test case belongs to

        resolution : str
            The resolution of the test case

        with_particles : bool
            Whether particles are include in the simulation

        with_surface_restoring : bool
            Whether surface restoring is included in the simulation

        long : bool
            Whether to run a long (3-year) simulation to quasi-equilibrium

        three_layer : bool
            Whether to use only 3 vertical layers and no continental shelf
        """
        self.resolution = resolution
        self.with_particles = with_particles
        self.with_surface_restoring = with_surface_restoring
        self.long = long
        self.three_layer = three_layer

        name = 'ensemble'
        if three_layer:
            name = 'three_layer'
        if long:
            if name is None:
                name = 'long'
            else:
                name = f'{name}_long'

        if with_surface_restoring:
            if name is None:
                name = 'surface_restoring'
            else:
                name = f'{name}_with_surface_restoring'

        if with_particles:
            if name is None:
                name = 'particles'
            else:
                name = f'{name}_with_particles'

        if name is None:
            name = 'default'

        subdir = f'{resolution}/{name}'

        super().__init__(test_group=test_group, name=name, subdir=subdir)

        self.add_step(InitialState(
            test_case=self, resolution=resolution,
            with_surface_restoring=with_surface_restoring,
            three_layer=three_layer))

        # space = Space([(0.0, 1e-4, "uniform")])      # Redi_constant_kappa
        #               (200.0, 2000.0, "uniform"),        # GM_constant_kappa
        #         (0.0, 3000.0, "uniform"),          # cvmix_background_diff
        #              (1e-4, 1e-2, "uniform"),         # implicit_bottom_drag
        #              (0.04, 0.08, "uniform"),         # submesoscale_Ce
        #              (200.0, 5000.0, "uniform"),      # submesoscale_Lfmin
        #              (86400.0, 864000.0, "uniform"),  # submesoscale_tau
        #              ])
        # n_samples = 1000

        # grid = Grid(border="include", use_full_layout=False)
        # x = grid.generate(space.dimensions, n_samples)
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=7)
        sample = sampler.random(n=1000)
        l_bounds = [200.0, 0.0, 0.0, 1e-4, 0.04, 200.0, 86400.0]
        u_bounds = [2000.0, 3000.0, 1e-4, 1e-2, 0.08, 5000.0, 864000.0]
        scaled_sample = qmc.scale(sample, l_bounds, u_bounds)

        option_list = []
        option_list.append({})
        for opt in scaled_sample:
            options = {}
            options['config_GM_constant_kappa'] = str(opt[0])
            options['config_Redi_constant_kappa'] = str(opt[1])
            options['config_cvmix_background_diffusion'] = str(opt[2])
            options['config_implicit_bottom_drag_coeff'] = str(opt[3])
            options['config_submesoscale_Ce'] = str(opt[4])
            options['config_submesoscale_Lfmin'] = str(opt[5])
            options['config_submesoscale_tau'] = str(opt[6])
            options['config_use_GM'] = '.true.'
            options['config_use_Redi'] = '.true.'
            options['config_submesoscale_enable'] = '.true.'
            option_list.append(options)

        options = {}
        options['config_GM_constant_kappa'] = '900.0'
        options['config_Redi_constant_kappa'] = '400.0'
        options['config_cvmix_background_diffusion'] = '0.0'
        options['config_implicit_bottom_drag_coeff'] = '1e-3'
        options['config_submesoscale_Ce'] = '0.06'
        options['config_submesoscale_Lfmin'] = '1000.0'
        options['config_submesoscale_tau'] = '172800.0'
        options['config_use_GM'] = '.true.'
        options['config_use_Redi'] = '.true.'
        options['config_submesoscale_enable'] = '.true.'
        option_list.append(options)

        for i, options in enumerate(option_list):
            self.add_step(Forward(
                test_case=self, resolution=resolution,
                with_particles=with_particles,
                with_surface_restoring=with_surface_restoring, long=long,
                three_layer=three_layer, options=options, step_num=i))

        if with_particles:
            self.add_step(Analysis(test_case=self, resolution=resolution))

    def configure(self):
        """
        Set config options that are different from the defaults
        """
        if self.three_layer:
            config = self.config
            # remove the continental shelf
            config.set('soma', 'phi', '1e-16')
            config.set('soma', 'shelf_depth', '0.0')

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        """
        variables = ['bottomDepth', 'layerThickness', 'maxLevelCell',
                     'temperature', 'salinity']
        compare_variables(
            test_case=self, variables=variables,
            filename1='initial_state/initial_state.nc')

        variables = ['temperature', 'layerThickness']
        compare_variables(
            test_case=self, variables=variables,
            filename1='forward/output/output.0001-01-01_00.00.00.nc')

        if self.with_particles:
            # just do particle validation at coarse res
            variables = [
                'xParticle', 'yParticle', 'zParticle', 'zLevelParticle',
                'buoyancyParticle', 'indexToParticleID', 'currentCell',
                'transfered', 'numTimesReset']
            compare_variables(test_case=self, variables=variables,
                              filename1='forward/analysis_members/'
                                        'lagrPartTrack.0001-01-01_00.00.00.nc')

            timers = ['init_lagrPartTrack', 'compute_lagrPartTrack',
                      'write_lagrPartTrack', 'restart_lagrPartTrack',
                      'finalize_lagrPartTrack']
            compare_timers(self, timers, rundir1='forward')
