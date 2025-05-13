import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def hyperchaotic_system(t, state, a, b, c, d, r):
    x, y, z, w = state
    dxdt = a * (y - x) + w
    dydt = (c - a) * x - x * z + c * y
    dzdt = x * y - b * z
    dwdt = -d * z + r * w
    return [dxdt, dydt, dzdt, dwdt]

a, b, c, d, r = 36, 3, 28, 0.7, 0.5

initial_state = [0.1, 0.2, 0.3, 0.4]

t_start, t_end, dt = 0, 512*512, 0.01
t_eval = np.arange(t_start, t_end, dt)

solution = solve_ivp(hyperchaotic_system, [t_start, t_end], initial_state, args=(a, b, c, d, r), t_eval=t_eval)


x_vals, y_vals, z_vals, w_vals = solution.y

print("Select the view:")
print("1: x vs y")
print("2: y vs x")
print("3: x vs z")
print("4: z vs y")
choice = int(input("Enter your choice (1-4): "))

plt.figure(figsize=(10, 6))

if choice == 1:
    plt.plot(x_vals, y_vals, color='blue', linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Hyperchaotic Sequence (x vs y)")
elif choice == 2:
    plt.plot(y_vals, x_vals, color='blue', linewidth=0.5)
    plt.xlabel("y")
    plt.ylabel("x")
    plt.title("Hyperchaotic Sequence (y vs x)")
elif choice == 3:
    plt.plot(x_vals, z_vals, color='blue', linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Hyperchaotic Sequence (x vs z)")
elif choice == 4:
    plt.plot(z_vals, y_vals, color='blue', linewidth=0.5)
    plt.xlabel("z")
    plt.ylabel("y")
    plt.title("Hyperchaotic Sequence (z vs y)")
else:
    print("Invalid choice. Defaulting to x vs y.")
    plt.plot(x_vals, w_vals, color='blue', linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("w")
    plt.title("Hyperchaotic Sequence (x vs w)")

plt.grid()
plt.show()
