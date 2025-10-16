# ==================================================
# پروژه سیگنال‌ها و سیستم‌ها - نسخه نهایی
# جدول جمع‌بندی فارسی
# ==================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from tabulate import tabulate

# -----------------------------
# سیگنال‌های پیوسته
# -----------------------------
t = np.linspace(-2, 5, 1000)
x_exp = np.exp(2*t) * (t >= 0)
x_sin = np.sin(2*np.pi*1*t)
x_step = (t >= 0).astype(float)
x_impulse = signal.unit_impulse(len(t), idx=len(t)//2)

plt.figure(figsize=(14,10))
plt.subplot(2,2,1)
plt.plot(t, x_exp, color='crimson', linewidth=2)
plt.title("سیگنال پیوسته: e^(2t) u(t)")
plt.grid(True)

plt.subplot(2,2,2)
plt.plot(t, x_sin, color='darkblue', linewidth=2)
plt.title("سیگنال سینوسی پیوسته: sin(2πt)")
plt.grid(True)

plt.subplot(2,2,3)
plt.plot(t, x_step, color='green', linewidth=2)
plt.title("تابه پله پیوسته: u(t)")
plt.grid(True)

plt.subplot(2,2,4)
plt.plot(t, x_impulse, color='purple', linewidth=2)
plt.title("تابه ضربه پیوسته: δ(t)")
plt.grid(True)

plt.tight_layout()
plt.show()

# -----------------------------
# سیگنال‌های گسسته
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
plt.title("سیگنال گسسته: 0.7^n")
plt.grid(True)

plt.subplot(2,2,2)
plt.stem(n, x_sin_discrete, linefmt='darkblue', markerfmt='bo', basefmt='k')
plt.title("سیگنال سینوسی گسسته: sin(2π0.1n)")
plt.grid(True)

plt.subplot(2,2,3)
plt.stem(n, x_step_discrete, linefmt='green', markerfmt='go', basefmt='k')
plt.title("تابه پله گسسته: u[n]")
plt.grid(True)

plt.subplot(2,2,4)
plt.stem(n, x_impulse_discrete, linefmt='purple', markerfmt='mo', basefmt='k')
plt.title("تابه ضربه گسسته: δ[n]")
plt.grid(True)

plt.tight_layout()
plt.show()

# -----------------------------
# انرژی و توان
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

print("📊 انرژی سیگنال‌ها:", energies)
print("📊 توان سیگنال‌ها:", powers)

# -----------------------------
# شیفت و معکوس
# -----------------------------
x_exp_shifted = np.exp(2*(t-1)) * (t >= 1)
x_exp_inv = x_exp[::-1]

plt.figure(figsize=(10,4))
plt.plot(t, x_exp, label="x(t)", linewidth=2)
plt.plot(t, x_exp_shifted, '--', label="x(t-1) شیفت", linewidth=2)
plt.plot(t, x_exp_inv, ':', label="x(-t) معکوس", linewidth=2)
plt.title("شیفت و معکوس سیگنال پیوسته")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# نمایی مختلط
# -----------------------------
x_complex = np.exp((1+1j*2*np.pi*0.5)*t) * (t>=0)

plt.figure(figsize=(10,4))
plt.plot(t, x_complex.real, label="قسمت حقیقی", color='blue', linewidth=2)
plt.plot(t, x_complex.imag, label="قسمت موهومی", color='red', linewidth=2)
plt.title("سیگنال نمایی مختلط پیوسته")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# خواص سیستم‌ها
# -----------------------------
print("\n📌 مثال خواص سیستم‌ها:")
print("سیستم بدون حافظه: y[n] = 2 x[n]")
print("سیستم معکوس‌پذیر: y[n] = x[n-2] -> x[n] = y[n+2]")
print("سیستم علی: y(t) = x(t-1) (وابسته به گذشته)")
print("سیستم پایدار: y[n] = 0.5^n x[n] (ورودی محدود -> خروجی محدود)")
print("سیستم تغییرناپذیر در زمان: y(t) = x(t-1)")

# -----------------------------
# جدول جمع‌بندی نهایی
# -----------------------------
data = {
    "موضوع": [
        "سیگنال پیوسته",
        "سیگنال گسسته",
        "شیفت و معکوس",
        "انرژی و توان",
        "نمایی مختلط",
        "سیگنال سینوسی",
        "تابه پله و ضربه",
        "سیستم بدون حافظه",
        "سیستم معکوس‌پذیر",
        "سیستم علی",
        "سیستم پایدار",
        "سیستم تغییرناپذیر در زمان"
    ],
    "توضیح": [
        "سیگنال‌های پیوسته مثل نمایی و سینوسی",
        "سیگنال‌های گسسته مثل 0.7^n و سینوس گسسته",
        "شیفت و معکوس سیگنال‌ها در زمان",
        "محاسبه انرژی و توان سیگنال‌ها",
        "قسمت حقیقی و موهومی نمایی مختلط",
        "سیگنال سینوسی تک فرکانس",
        "تابه پله و ضربه",
        "خروجی فقط به ورودی فعلی وابسته است",
        "سیستم قابل بازگشت به ورودی",
        "خروجی فقط به ورودی‌های گذشته وابسته است",
        "ورودی محدود -> خروجی محدود",
        "خروجی سیستم با ورودی تغییر می‌کند"
    ]
}

print("\n📌 جدول جمع‌بندی مباحث و خواص سیستم‌ها:\n")
print(tabulate(data, headers='keys', tablefmt='grid'))
