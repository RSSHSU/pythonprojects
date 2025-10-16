# ==================================================
# Signal and Systems Project - Final Version
# Romanized Persian for outputs and titles
# ==================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from tabulate import tabulate

# -----------------------------
# Continuous Signals
# -----------------------------
t = np.linspace(-2, 5, 1000)
x_exp = np.exp(2*t) * (t >= 0)
x_sin = np.sin(2*np.pi*1*t)
x_step = (t >= 0).astype(float)
x_impulse = signal.unit_impulse(len(t), idx=len(t)//2)

plt.figure(figsize=(14,10))
plt.subplot(2,2,1)
plt.plot(t, x_exp, color='crimson', linewidth=2)
plt.title("Namayeh Piyvaste: e^(2t) u(t)")
plt.grid(True)

plt.subplot(2,2,2)
plt.plot(t, x_sin, color='darkblue', linewidth=2)
plt.title("Sinusoid Piyvaste: sin(2πt)")
plt.grid(True)

plt.subplot(2,2,3)
plt.plot(t, x_step, color='green', linewidth=2)
plt.title("Tabe Pele Piyvaste: u(t)")
plt.grid(True)

plt.subplot(2,2,4)
plt.plot(t, x_impulse, color='purple', linewidth=2)
plt.title("Tabe Zarbeh Piyvaste: δ(t)")
plt.grid(True)

plt.tight_layout()
plt.show()

# -----------------------------
# Discrete Signals
# -----------------------------
n = np.arange(0, 20)
x_exp_discrete = 0.7**n
x_sin_discrete = np.sin(2*np.pi*0.1*n)
x_step_discrete = (n >= 0).astype(float)
x_impulse_discrete = np.zeros(len(n))
x_impulse_discrete[0] = 1

plt.figure(figsize=(14,6))
plt.subplot(2,2,1)
plt.stem(n, x_exp_discrete, linefmt='crimson', markerfmt='ro', basefmt='k')
plt.title("Namayeh Gosaste: 0.7^n")
plt.grid(True)

plt.subplot(2,2,2)
plt.stem(n, x_sin_discrete, linefmt='darkblue', markerfmt='bo', basefmt='k')
plt.title("Sinusoid Gosaste: sin(2π0.1n)")
plt.grid(True)

plt.subplot(2,2,3)
plt.stem(n, x_step_discrete, linefmt='green', markerfmt='go', basefmt='k')
plt.title("Tabe Pele Gosaste: u[n]")
plt.grid(True)

plt.subplot(2,2,4)
plt.stem(n, x_impulse_discrete, linefmt='purple', markerfmt='mo', basefmt='k')
plt.title("Tabe Zarbeh Gosaste: δ[n]")
plt.grid(True)

plt.tight_layout()
plt.show()

# -----------------------------
# Energy & Power
# -----------------------------
def energy_continuous(x, t):
    return np.trapz(np.abs(x)**2, t)

def power_continuous(x, t):
    T = t[-1] - t[0]
    return (1/(2*T)) * np.trapz(np.abs(x)**2, t)

def energy_discrete(x):
    return np.sum(np.abs(x)**2)

def power_discrete(x):
    return np.mean(np.abs(x)**2)

energies = {
    "x_exp": energy_continuous(x_exp, t),
    "x_sin": energy_continuous(x_sin, t),
    "x_exp_discrete": energy_discrete(x_exp_discrete),
    "x_sin_discrete": energy_discrete(x_sin_discrete)
}

powers = {
    "x_exp": power_continuous(x_exp, t),
    "x_sin": power_continuous(x_sin, t),
    "x_exp_discrete": power_discrete(x_exp_discrete),
    "x_sin_discrete": power_discrete(x_sin_discrete)
}

print("📊 Enerzhiye Signalha:", energies)
print("📊 Tavan Signalha:", powers)

# -----------------------------
# Shift & Reversal
# -----------------------------
x_exp_shifted = np.exp(2*(t-1)) * (t >= 1)
x_exp_inv = x_exp[::-1]

plt.figure(figsize=(10,4))
plt.plot(t, x_exp, label="x(t)", linewidth=2)
plt.plot(t, x_exp_shifted, '--', label="x(t-1) shift", linewidth=2)
plt.plot(t, x_exp_inv, ':', label="x(-t) reversed", linewidth=2)
plt.title("Shift va Reversal Namayeh Piyvaste")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Complex Exponential
# -----------------------------
x_complex = np.exp((1+1j*2*np.pi*0.5)*t) * (t>=0)

plt.figure(figsize=(10,4))
plt.plot(t, x_complex.real, label="Real Part", color='blue', linewidth=2)
plt.plot(t, x_complex.imag, label="Imag Part", color='red', linewidth=2)
plt.title("Namayeh Gosaste Mokhtalet Piyvaste")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# System Properties
# -----------------------------
print("\n📌 System Properties Examples:")
print("Memoryless System: y[n] = 2 x[n]")
print("Invertible System: y[n] = x[n-2] -> x[n] = y[n+2]")
print("Causal System: y(t) = x(t-1) (depends on past only)")
print("Stable System: y[n] = 0.5^n x[n] (bounded input -> bounded output)")
print("Time-Invariant System: y(t) = x(t-1) (input shift -> output shift)")

# -----------------------------
# Summary Table (Romanized Persian)
# -----------------------------
data = {
    "Topic": [
        "Namayeh Piyvaste",
        "Namayeh Gosaste",
        "Shift & Reversal",
        "Energy & Power",
        "Complex Exponential",
        "Sinusoid",
        "Tabe Pele & Zarbeh",
        "Memoryless System",
        "Invertible System",
        "Causal System",
        "Stable System",
        "Time-Invariant System"
    ],
    "Description": [
        "Continuous signals like exponential and sinusoid",
        "Discrete signals like 0.7^n and discrete sinusoid",
        "Shifting and reversing signals in time",
        "Compute energy and power of signals",
        "Real & imaginary parts of complex exponential",
        "Single frequency sinusoidal signal",
        "Step & impulse functions",
        "Output depends only on current input",
        "System can be reversed to get input",
        "Output depends on past input values",
        "Bounded input gives bounded output",
        "System output shifts same as input"
    ]
}

print("\n📌 Summary Table of Topics and System Properties:\n")
print(tabulate(data, headers='keys', tablefmt='grid'))

