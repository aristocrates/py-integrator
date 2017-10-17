py-integrator
=============

A modular and extensible numerical differential equation solving framework
written in python.

Works with problems which may be expressed as a system of coupled first
order ordinary differential equations in time.

Example usage:
--------------

```python
import integrator, problem
spring_mass = problem.Harmonic(m = 1, k = 1, initial_position = 0,
                               initial_velocity = 1, initial_time = 0)
runge_kutta = integrator.RungeKutta(spring_mass)
runge_kutta.step(delta = 1e-3)
# view the current 
```

To add new integration methods:
-------------------------------

Extend the class ```integrator.Integrator``` and override the ```step()```
function.

To add a new problem:
---------------------

Override the class ```problem.Problem``` and override the constructor to
request the appropriate parameters and provide the superconstructor with
the appropriate first order differential equations.
