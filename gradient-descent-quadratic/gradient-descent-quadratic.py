def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    linear_model = a * x0**2 + b * x0 + c

    x = x0
    for i in range(steps):
        dx = 2 * x * a + b
        x = x - lr*dx

    return x