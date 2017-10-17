"""
Utility to take numerical partial derivatives
"""

# TODO: put the miscellaneous parameters in **kwargs
def derivative(function, point, step = None, start = None,
               threshold = 1e-10, timeout = 1e5,
               method = "forward_secant"):
    """
    Numerically calculate the partial derivative of a function with respect to
    a variable at a particular value using a specified method and step size.

    function should accept one real variable as an argument

    If step is None, then the step size is set to start and iteratively
    decreased until the absolute change in the derivative is less than
    threshold.
    If start is also None, the first step size is 1

    If convergence does not occur within timeout number of steps, then
    the error is logged to stderr and the value is returned.
    If timeout is set to None, then timeout will never occur.

    If step is not None, then start, threshold, timeout are ignored

    Valid methods: "forward_secant", "symmetric_secant", "backward_secant"
    """
    def evaluation_points(step_size):
        if method == "forward_secant":
            return (point, point + step_size)
        elif method == "symmetric_secant":
            return (point - step_size / 2, point + step_size / 2)
        elif method == "backward_secant":
            return (point - step_size, point)
        else:
            raise(ValueError("uknown method: %s" % method))
    if step is None:
        if start is None:
            current_step = 1
        else:
            current_step = start
        counter = 0
        last_value = None
        while (timeout is None or counter < timeout):
            points = evaluation_points(current_step)
            # TODO: instead of halving the range, let the user specify a
            # custom geometric decay with default of 0.5
            current_value = (function(points[1])
                             - function(points[0])) / current_step
            print(current_step)
            if last_value is not None:
                diff = abs(current_value - last_value)
                print(str(diff) + " " + str(threshold))
                if diff < threshold:
                    return current_value
            last_value = current_value
            current_step *= 0.5
    else:
        points = evaluation_points(step)
        return (function(points[1]) - function(points[0])) / step
