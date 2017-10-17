"""
Provides consistent interface for ODE integrators in several variables
"""
import numpy as np
import differentiator

class Integrator:
    def __init__(self, problem):
        """
        """
        self.problem = problem
        # make a copy of the list so the problem is not disturbed
        self._current_data = list(self.problem.initial_data)
        self._current_time = self._current_data[-1]

    def current_data(self):
        """
        Returns a copy of the current state
        """
        return self._current_data[:]
        
    def step(self, delta = None):
        """
        If delta is None, then the default current step size is used
        """
        pass

class ForwardEuler(Integrator):
    """
    The forward Euler method freezes the current values of all variables and
    then updates every variable using the derivatives based on the current
    variable values
    """
    def __init__(self, problem):

        super().__init__(problem)

    def step(self, delta = None):
        """
        If delta is None, does nothing
        """
        if delta is None:
            pass
        else:
            derivs = []
            for i in range(self.problem.n_vars):
                derivs.append(self.problem.value_n_deriv(i,self._current_data))
            self._current_time += delta
            self._current_data[-1] += delta
            assert(len(derivs) == self.problem.n_vars)
            for i, d in enumerate(derivs):
                self._current_data[i] += delta * d

class RungeKutta(Integrator):
    """
    Implements RK4
    """
    def __init__(self, problem):
        super().__init__(problem)

    def step(self, delta = None):
        if delta is None:
            pass
        else:
            k1 = []
            for i in range(self.problem.n_vars):
                k1.append(self.problem.value_n_deriv(i, self._current_data))
            k2 = []
            k1_arr = [self._current_data[i] + k * delta / 2
                      for i, k in enumerate(k1)] \
                          + [self._current_time + delta / 2]
            for i in range(self.problem.n_vars):
                # add in the time separately, note that these need to be
                # normal (not numpy) arrays to work properly
                k2.append(self.problem.value_n_deriv(i, k1_arr))
            k3 = []
            k2_arr = [self._current_data[i] + k * delta / 2
                      for i, k in enumerate(k2)] \
                          + [self._current_time + delta / 2]
            for i in range(self.problem.n_vars):
                k3.append(self.problem.value_n_deriv(i, k2_arr))
            k4 = []
            k3_arr = [self._current_data[i] + k * delta
                      for i, k in enumerate(k3)] \
                          + [self._current_time + delta]
            for i in range(self.problem.n_vars):
                k4.append(self.problem.value_n_deriv(i, k3_arr))

            change = []
            # everything except time
            for i in range(self.problem.n_vars):
                change.append(delta / 6. * (k1[i] + 2 * k2[i]
                                            + 2 * k3[i] + k4[i]))
            # time
            change.append(delta)

            # add the changed value to the current data
            for i, val in enumerate(change):
                self._current_data[i] += val
            self._current_time += delta

class DormandPrince(Integrator):
    """
    Implements the Dormand Prince adaptive Runge Kutta routine
    """
    def __init__(self, problem, step_size = 1e-3, tolerance = 1e-12,
                 adaptive = True, growth_factor = 1.5):
        super().__init__(problem)
        self.step_size = step_size
        self.tolerance = tolerance
        self.adaptive = adaptive
        self.growth_factor = growth_factor

    def step(self, delta = None):
        h = self.step_size
        # if not adaptive, use the provided value for delta
        if self.adaptive == False:
            if delta is not None:
                h = delta
        current_data_np = np.array(self._current_data)
        def value_all_deriv(data):
            """
            Always returns 0 for the time part
            """
            return np.array([self.problem.value_n_deriv(i, data)
                             for i in range(self.problem.n_vars)] + [0])
        unit_time = np.array([0] * self.problem.n_vars + [1])
        k1 = h * value_all_deriv(current_data_np)
        k2_arg = (current_data_np + 1/5. * k1 + 1/5.*h * unit_time)
        k2 = h * value_all_deriv(k2_arg)
        k3_arg = (current_data_np + 3/40. * k1 + 9/40. * k2
                  + 3/10.*h * unit_time)
        k3 = h * value_all_deriv(k3_arg)
        k4_arg = (current_data_np + 44/45. * k1 - 56/15. * k2
                  + 32/9. * k3 + 4/5.*h * unit_time)
        k4 = h * value_all_deriv(k4_arg)
        k5_arg = (current_data_np + 19372/6561. * k1 - 25360/2187. * k2
                  + 64448/6561. * k3 - 212/729. * k4 + 8/9.*h * unit_time)
        k5 = h * value_all_deriv(k5_arg)
        k6_arg = (current_data_np + 9017/3168. * k1 - 355/33. * k2
                  + 46732/5247. * k3 + 49/176. * k4 - 5103 / 18656. * k5
                  + h * unit_time)
        k6 = h * value_all_deriv(k6_arg)
        k7_arg = (current_data_np + 35/384. * k1 + 500/1113. * k3
                  + 125 / 192. * k4 - 2187/6784. * k5 + 11/84. * k6
                  + h * unit_time)
        k7 = h * value_all_deriv(k7_arg)
        z = (current_data_np + 5179 / 57600. * k1 + 7571 / 16695. * k3
             + 393/640. * k4 - 92097 / 339200. * k5 + 187 / 2100. * k6
             + 1/40. * k7 + h * unit_time)
        self._current_data = k7_arg[:]
        if self.adaptive:
            # update the step size
            err = (sum([k**2 for k in [z[i] - k7_arg[i]
                                       for i, v in enumerate(z)]]))**0.5
            # prevents the step size from growing too much
            # restricts each successive step to
            # [(growth factor)^-1, (growth factor)] * (current step size)
            self.step_size = min(max((self.tolerance * h
                                      / (2 * err))**(1.0 / 5.0),
                                     self.growth_factor**-1),
                                 self.growth_factor) * self.step_size
