"""
Generates plots for various test problems
"""
import numpy as np

# if called as a script disable plt.show()
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

import problem
import integrator

def dp_adaptive_step_plot(filename, problem,
                          title = "Dormand Prince Step Size",
                          iterations = 10000, **kwargs):
    """
    Use kwargs to specify custom values for the Dormand Prince integrator
    """
    dp = integrator.DormandPrince(problem, **kwargs)
    steps = []
    for i in range(iterations):
        steps.append(dp.step_size)
        dp.step()
    plt.xlabel("Step Number")
    plt.ylabel("Step Size")
    plt.title(title)
    plt.plot(steps)
    plt.gca().set_yscale("log")
    plt.locator_params(numticks=6)
    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    return steps

def lyapunov_plot(filename):
    separation = [10**-13] * 3
    iterations = [10**n for n in range(1, 7)]
    lyapunov = [max_lyapunov(separation, iterations = iteration)
                for iteration in iterations]
    plt.xlabel("Number of iterations")
    plt.ylabel("Maximum Lyapunov exponent")
    plt.scatter(iterations, lyapunov)
    plt.gca().set_xscale("log")
    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

def max_lyapunov(initial_sep, initial_position = [0, 0, 0],
                 step_size = 1e-3, iterations = 10**6):
    """
    Applies the initial separation and tries to measure the maximum
    lyapunov exponent
    """
    initial_sep_magnitude = (initial_sep[0]**2 + initial_sep[1]**2
                             + initial_sep[2]**2)**0.5
    def separation(data1, data2):
        return ( (data1[0] - data2[0])**2 + (data1[1] - data2[1])**2
                 + (data1[2] - data2[2])**2)**0.5
    def current_lyapunov(data1, data2, time):
        return 1 / time * np.log(separation(data1, data2) /
                                 initial_sep_magnitude)
    # if 0 is returned, either it was actually 0 or it was negative
    ans = 0
    initial_data = initial_position + [0]
    lorenz1 = problem.Lorenz(initial_data = initial_data)
    lorenz2 = problem.Lorenz(initial_data = [initial_data[0] + initial_sep[0],
                                             initial_data[1] + initial_sep[1],
                                             initial_data[2] + initial_sep[2],
                                             0])
    rk1 = integrator.RungeKutta(lorenz1)
    rk2 = integrator.RungeKutta(lorenz2)
    for i in range(iterations):
        rk1.step(step_size)
        rk2.step(step_size)
        ly = current_lyapunov(rk1.current_data(), rk2.current_data(),
                              rk1.current_data()[-1])
        if ly > ans:
            ans = ly
    return ans

def harmonic_convergence_plot(filename, largest_step = 5e-1,
                              smallest_step = 5e-5, **kwargs):
    step_sizes_prime = np.logspace(np.log10(largest_step),
                                   np.log10(smallest_step),
                                   num = 1 + abs(int(np.log10(smallest_step
                                                              / largest_step))
                                   ))
    #print(step_sizes)
    # add intermediate step sizes
    step_sizes = []
    for i, v in enumerate(step_sizes_prime):
        step_sizes.append(v)
        if i < len(step_sizes_prime) - 1:
            step_sizes.append(5 * step_sizes_prime[i + 1])
    #print(step_sizes)
    positions_euler = []
    velocities_euler = []
    positions_rk = []
    velocities_rk = []
    positions_dp = []
    velocities_dp = []
    for step in step_sizes:
        # set up a fresh problem each time
        simple_mass_spring = problem.Harmonic(k = 1, m = 1)
        euler = integrator.ForwardEuler(simple_mass_spring)
        rk = integrator.RungeKutta(simple_mass_spring)
        dp = integrator.DormandPrince(simple_mass_spring, step, 1,
                                      adaptive = False)
        
        num_steps = int(np.round(10 * largest_step / step))
        for i in range(num_steps):
            euler.step(delta = step)
            rk.step(delta = step)
            dp.step(delta = step)
        final_euler = euler.current_data()
        final_rk = rk.current_data()
        final_dp = dp.current_data()
        # the analytic solution is x(t) = cos(t), v(t) = -sin(t)
        # TODO: make sure this isn't off by one step size
        # if correct, will look like exponential decay
        # if wrong, might "flatline", indicating systematic offset error
        # final_analytic = [np.cos(10 * largest_step),
        #                   -np.sin(10 * largest_step),
        #                   10 * largest_step]
        positions_euler.append(final_euler[0])
        velocities_euler.append(final_euler[1])
        positions_rk.append(final_rk[0])
        velocities_rk.append(final_rk[1])
        positions_dp.append(final_dp[0])
        velocities_dp.append(final_dp[1])
    error_position_euler = np.array([np.abs(x - np.cos(10 * largest_step))
                                     for x in positions_euler])
    error_velocity_euler = np.array([np.abs(v + np.sin(10 * largest_step))
                                     for v in velocities_euler])
    error_position_rk = np.array([np.abs(x - np.cos(10 * largest_step))
                                  for x in positions_rk])
    error_velocity_rk = np.array([np.abs(v + np.sin(10 * largest_step))
                                  for v in velocities_rk])
    error_position_dp = np.array([np.abs(x - np.cos(10 * largest_step))
                                  for x in positions_dp])
    error_velocity_dp = np.array([np.abs(v + np.sin(10 * largest_step))
                                  for v in velocities_dp])
    plt.title("Euler, RK4, Dormand Prince Convergence")
    plt.xlabel("Step size")
    plt.ylabel("Absolute Error")
    plt.loglog(step_sizes, error_position_euler, label="Euler x")
    plt.loglog(step_sizes, error_velocity_euler, label="Euler v")
    plt.loglog(step_sizes, error_position_rk, label="RK4 x")
    plt.loglog(step_sizes, error_velocity_rk, label="RK4 v")
    plt.loglog(step_sizes, error_position_dp, label="DP x")
    plt.loglog(step_sizes, error_velocity_dp, label="DP v")
    plt.legend()
    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    return [step_sizes, error_position_euler, error_velocity_euler,
            error_position_rk, error_velocity_rk, error_position_dp,
            error_velocity_dp]

def lorenz_xyz(total_time = 100, step_size = 1e-3):
    """
    Computes trajectories of the default Lorenz Attractor using RK4
    """
    lorenz = problem.Lorenz()
    rk = integrator.RungeKutta(lorenz)
    x = []
    y = []
    z = []
    x.append(rk.current_data()[0])
    y.append(rk.current_data()[1])
    z.append(rk.current_data()[2])
    for i in range(int(total_time / step_size)):
        rk.step(step_size)
        x.append(rk.current_data()[0])
        y.append(rk.current_data()[1])
        z.append(rk.current_data()[2])
    return [x, y, z]
    
def mass_spring_time_plot_euler(filename, step = 1e-3, num_steps = 10**4,
                                **kwargs):
    """
    To not save or close the figure after plotting, set filename = None
    """
    simple_mass_spring = problem.Harmonic(k = 1, m = 1)
    euler = integrator.ForwardEuler(simple_mass_spring)
    history = []
    history.append(euler.current_data())
    for i in range(num_steps):
        euler.step(delta = step)
        history.append(euler.current_data())
    times      = np.array([k[-1] for k in history])
    positions  = np.array([k[0] for k in history])
    velocities = np.array([k[1] for k in history])
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Harmonic Oscillator Euler")
    plt.plot(times, positions, label="Position")
    plt.plot(times, velocities, label="Velocity")
    plt.legend()
    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

def mass_spring_time_plot_rk(filename, step = 1e-3, num_steps = 10**4,
                             **kwargs):
    """
    This plot includes a linear velocity dependent drag

    To not save or close the figure after plotting, set filename = None
    """
    drag = lambda v: 0.3
    simple_mass_spring = problem.Harmonic(k = 1, m = 1, velocity_drag = drag)
    rk = integrator.RungeKutta(simple_mass_spring)
    history = []
    history.append(rk.current_data())
    for i in range(num_steps):
        rk.step(delta = step)
        history.append(rk.current_data())
    times      = np.array([k[-1] for k in history])
    positions  = np.array([k[0] for k in history])
    velocities = np.array([k[1] for k in history])
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Harmonic Oscillator RK4")
    plt.plot(times, positions, label="Position")
    plt.plot(times, velocities, label="Velocity")
    plt.legend()
    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

def double_pendulum_plot(filename, pendulum, dp, num_iterations = 10000):
    position_history = []
    for i in range(num_iterations):
        position_history.append(pendulum.locations(dp.current_data()))
        dp.step()
    rod2x = np.array([k[2] for k in position_history])
    rod2y = np.array([k[3] for k in position_history])
    plt.plot(rod2x, rod2y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Second Pendulum Location")
    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

def poincare_section(filename, pendulum, iterations, **kwargs):
    dp = integrator.DormandPrince(pendulum)
    history = []
    for i in range(iterations):
        val = pendulum.poincare(dp.current_data())
        if val is not None:
            history.append(val)
        dp.step()
    theta = np.array([h[0] for h in history])
    p = np.array([h[1] for h in history])
    plt.xlabel(r"$\theta_2$")
    plt.ylabel(r"$p_2$")
    plt.title("Double Pendulum Poincare Section")
    plt.scatter(theta, p, s=2)
    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    print("Note: generating all plots can take a long time")
    print("""A "plots/" directory is required""")
    print("This function should work but has not been directly tested")
    print("")
    print("Making simple spring plots")
    mass_spring_time_plot_euler(filename = "plots/spring_euler.pdf")
    mass_spring_time_plot_rk(filename = "plots/spring_rk.pdf")
    harmonic_convergence_plot(filename = "plots/euler_rk_convergence.pdf")

    print("Making Lorenz attractor plots")
    # Lorenz attractor plots
    [x, y, z] = lorenz_xyz()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x[:int(len(x)/10)], y[:int(len(y)/10)])
    plt.tight_layout()
    plt.savefig("plots/lorenz_xy_partial.pdf")
    plt.close()

    plt.xlabel("x")
    plt.ylabel("z")
    plt.plot(x[:int(len(x)/10)], z[:int(len(z)/10)])
    plt.tight_layout()
    plt.savefig("plots/lorenz_xz_partial.pdf")
    plt.close()

    plt.xlabel("y")
    plt.ylabel("z")
    plt.plot(y[:int(len(y)/10)], z[:int(len(z)/10)])
    plt.tight_layout()
    plt.savefig("plots/lorenz_yz_partial.pdf")
    plt.close()
    
    # full plots
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, y)
    plt.tight_layout()
    plt.savefig("plots/lorenz_xy.pdf")
    plt.close()

    plt.xlabel("x")
    plt.ylabel("z")
    plt.plot(x, z)
    plt.tight_layout()
    plt.savefig("plots/lorenz_xz.pdf")
    plt.close()

    plt.xlabel("y")
    plt.ylabel("z")
    plt.plot(y, z)
    plt.tight_layout()
    plt.savefig("plots/lorenz_yz.pdf")
    plt.close()

    # lyapunov plot
    lyapunov_plot("plots/lyapunov.pdf")

    print("Making Dormand Prince adaptive step size plot")
    # adaptive step plots
    for i in [9, 12, 12.838]:
        dp_adaptive_step_plot("plots/adaptive_" + str(i) + ".pdf",
                              problem.Harmonic(),
                              title = r"Dormand Prince Step Size, $\epsilon = "
                              + r"10^{-" + str(i) + r"}$",
                              step_size = 1e-2, tolerance = 10**(-i))

    print("Double pendulum plots")
    # double pendulum
    pendulum = problem.DoublePendulum(initial_data = [np.pi / 6., np.pi / 2.,
                                                      0., 1., 0.])
    dp = integrator.DormandPrince(pendulum, tolerance = 1e-9)
    double_pendulum_plot("plots/pendulum.pdf", pendulum, dp,
                         num_iterations = 10000)
    double_pendulum_plot("plots/pendulum_long.pdf", pendulum, dp,
                         num_iterations = 40000)
    # proof that the step sizes are reasonable
    dp_adaptive_step_plot("plots/pendulum_steps.pdf",
                          pendulum, iterations = 40000,
                          title = "Double Pendulum Step Size",
                          tolerance = 1e-9)

    print("Poincare section (this will take a while)")
    # poincare section
    poincare_section("plots/pendulum_poincare.pdf",
                     problem.DoublePendulum(initial_data = [np.pi / 3.,
                                                            3 * np.pi / 4.,
                                                            0.003, -0.01, 0.]),
                     1000000,
                     tolerance = 1e-9)
    
    print("Poincare section (this one will take a very long time)")
    poincare_section("plots/pendulum_poincare_extended.pdf",
                     problem.DoublePendulum(initial_data = [np.pi / 3.,
                                                            3 * np.pi / 4.,
                                                            0.003, -0.01, 0.]),
                     5000000,
                     tolerance = 1e-9)
