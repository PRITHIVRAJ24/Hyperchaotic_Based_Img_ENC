import numpy as np
import pickle

def hyperchaotic_chen(x0, y0, z0, w0, a, b, c, d, e, dt, steps):
    x, y, z, w = x0, y0, z0, w0
    sequence = []
    
    for _ in range(steps):
        dx = a * (y - x) + w
        dy = (c - a) * x - x * z + c * y
        dz = x * y - b * z
        dw = -d * x - e * w
        
        x += dx * dt
        y += dy * dt
        z += dz * dt
        w += dw * dt
        
        sequence.append((x, y, z, w))
    
    return sequence

# Initial conditions
x0, y0, z0, w0 = 0.1, 0.2, 0.3, 0.4
a, b, c, d, e = 35, 3, 28, 0.7, 0.5
dt = 0.01
steps = 512 * 512

# Generate the sequence
sequence = hyperchaotic_chen(x0, y0, z0, w0, a, b, c, d, e, dt, steps)

# Save to pkl file
with open("hyperchaotic_sequence.pkl", "wb") as f:
    pickle.dump(sequence, f)

print("Hyperchaotic sequence saved to hyperchaotic_sequence.pkl")
