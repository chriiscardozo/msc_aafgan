import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

########## Função degrau
def step():
    z = np.arange(-5, 5, .001)
    step_fn = np.vectorize(lambda z: 1.0 if z >= 0.0 else 0.0)
    step = step_fn(z)
    return z, step, "Função degrau"

def step_d():
    z = np.arange(-5, 5, .02)
    return z, [0]*len(z), "Derivada da função degrau"


########## Função logística
def logistic():
    z = np.arange(-10, 10, .1)
    sigma_fn = np.vectorize(lambda z: 1/(1+np.exp(-z)))
    sigma = sigma_fn(z)
    return z, sigma, "Função logística"

def logistic_d():
    z = np.arange(-10, 10, .1)
    sigma_fn = np.vectorize(lambda z: np.exp(-z)/((np.exp(-z) + 1)**2))
    sigma = sigma_fn(z)
    return z, sigma, "Derivada da função logística"


########## Função ReLU
def relu():
    z = np.arange(-2, 2, .01)
    zero = np.zeros(len(z))
    y = np.max([zero, z], axis=0)
    return z, y, "Função ReLU"

def relu_d():
    z = np.arange(-2, 2, .01)
    y = [1 if x > 0 else 0 for x in z]
    return z, y, "Derivada da função ReLU"


########## Função LeakyReLU
def lrelu(alpha=.2):
    z = np.arange(-2, 2, .01)
    y = [x if x > 0 else alpha*x for x in z]
    return z, y, "Função LeakyReLU (α = 0.2)"

def lrelu_d(alpha=.2):
    z = np.arange(-2, 2, .01)
    y = [1 if x > 0 else alpha for x in z]
    return z, y, "Derivada da função LeakyReLU (α = 0.2)"


########## Função tangente hiperbólica
def tanh():
    z = np.arange(-5, 5, .1)
    t = np.tanh(z)
    return z, t, "Função tangente hiperbólica"

def tanh_d():
    z = np.arange(-5, 5, .1)
    # sech(z) = 1 / cosh(z)
    t = 1/np.cosh(z)
    return z, t, "Derivada da função tangente hiperbólica"

########## Função tangente hiperbólica
def mish():
    z = np.arange(-5, 5, .1)
    t = z * np.tanh(np.log(1 + np.exp(z)))
    return z, t, "Função Mish"

def mish_d():
    z = np.arange(-5, 5, .1)
    t = ((2*np.exp(z)*z*(1+np.exp(z)))/((1+np.exp(z))**2+1)) - \
            (2*np.exp(z)*z*((1+np.exp(z))**2 - 1)*(1+np.exp(z)))/((1+np.exp(z))**2 + 1)**2 + \
                ((1+np.exp(z))**2 - 1)/((1+np.exp(z))**2 + 1)
    return z, t, "Derivada da função Mish"

########## Função APL
def apl_func(x,a,b):
    return max(0, x) + sum([a[i]*max(0, -x + b[i]) for i in range(len(a))])

def apl(a=[.2], b=[0]):
    z = np.arange(-5, 5, .01)
    t = [apl_func(x,a,b) for x in z]
    return z, t, "Função APL (a = " + a + "; b = " + b + ")"

def apl_d_a02_b0():
    z = np.arange(-5, 5, .01)
    t = [1 if x > 0 else -0.2 for x in z]
    return z, t, "Derivada da função APL (a = 0.2; b = 0)"

########## Função SHReLU
def shrelu(tau=0.2):
    z = np.arange(-5, 5, .01)
    t = [0.5*(x + np.sqrt(x**2 + tau**2)) for x in z]
    return z, t, "Função SHReLU (τ = %.2f)" % (tau)

def shrelu_d(tau=0.2):
    z = np.arange(-5, 5, .01)
    t = [0.5*((x/(np.sqrt(x**2 + tau**2))) + 1) for x in z]
    return z, t, "Derivada da função SHReLU (τ = %.2f)" % (tau)

########## Função BHSA
def h1_bhsa(l, tau, z):
    return np.sqrt( (l**2) * (z + (1/(2*l)))**2 + tau**2)

def h2_bhsa(l, tau, z):
    return np.sqrt( (l**2) * (z - (1/(2*l)))**2 + tau**2)

def bhsa(l=1, tau=0, z = None):
    # h1 = torch.sqrt( ((self.l**2) * (x + (1/(2*self.l)))**2) + self.t1**2 )
    # h2 = torch.sqrt( ((self.l**2) * (x - (1/(2*self.l)))**2) + self.t2**2 )
    if z is None: z = np.arange(-5, 5, .01)
    t = h1_bhsa(l, tau, z) - h2_bhsa(l, tau, z)
    return z, t, "Função BHSA (λ = %.2f; τ = %.2f)" % (l, tau)

def bhsa_d(l=1, tau=0):
    z = np.arange(-5, 5, .01)
    t = (l * (1 - 2 * l * z))/np.sqrt(4 * tau**2 + (1 - 2 * l * z)**2) + (l * (1 + 2 * l * z))/np.sqrt(4 * tau**2 + (1 + 2 * l * z)**2)
    return z, t, "Derivada da função BHSA (λ = %.2f; τ = %.2f)" % (l, tau)

########## Função BHAA
def bhaa(l=1, tau1=0, tau2=0):
    z = np.arange(-5, 5, .01)
    t = np.sqrt((l**2) * (z+(1/(2*l)))**2 + (tau1**2)) - np.sqrt((l**2) * (z-(1/(2*l)))**2 + (tau2**2))
    return z, t, "Função BHAA (λ = %.2f; τ1 = %.2f; τ2 = %.2f)" % (l, tau1, tau2)

def bhaa_d(l=1, tau1=0, tau2=0):
    z = np.arange(-5, 5, .01)
    t = (l * (1 - 2 * l * z))/np.sqrt((1 - 2 * l * z)**2 + 4 * tau2**2) + (l * (2 * l * z + 1))/np.sqrt((2 * l * z + 1)**2 + 4 * tau1**2)
    return z, t, "Derivada da função BHAA (λ = %.2f; τ1 = %.2f; τ2 = %.2f)" % (l, tau1, tau2)

def bhata(l=1,tau1=0,tau2=0):
    z, t, _ = bhaa(l, tau1, tau2)
    t[t > 1] = 1
    t[t < -1] = -1
    return z, t, "Função BHAA truncada (λ = %.2f; τ1 = %.2f; τ2 = %.2f)" % (l, tau1, tau2)

def func_bh_derivative_equal_to_zero(tau1, tau2, l):
    return [(tau1 - tau2)/(2*l*(tau1 + tau2)), (tau1 + tau2)/(2*l*(tau1 - tau2))]

def raw_func_bh_derivative(x, tau1, tau2, l):
    return (l * (1 - 2 * l * x))/np.sqrt(4 * tau2**2 + (1 - 2 * l * x)**2) + (l * (1 + 2 * l * x))/np.sqrt(4 * tau1**2 + (1 + 2 * l * x)**2)

def raw_func_bh(x, tau1, tau2, l):
    return (np.sqrt((l**2) * (x+(1/(2*l)))**2 + (tau1**2)) - np.sqrt((l**2) * (x-(1/(2*l)))**2 + (tau2**2)))

def bhana(l=1,tau1=0,tau2=0):
    z = np.arange(-5, 5, .001)
    result = raw_func_bh(z, tau1, tau2, l)
    func_points_edge_value = func_bh_derivative_equal_to_zero(tau1, tau2, l)
    func_points_edge_value = max(func_points_edge_value) if abs(tau1) > abs(tau2) else min(func_points_edge_value)
    func_edge_value = raw_func_bh(func_points_edge_value, tau1, tau2, l)

    if(abs(tau1) > abs(tau2)):
        result = (result - (-1))/(func_edge_value + 0.001 - (-1))
    elif(abs(tau2) > abs(tau1)):
        result = (result - func_edge_value)/(1 + 0.001 - func_edge_value)

    result = (result - 0.5) * 2

    return z, result, "Função BHANA (λ = %.2f; τ1 = %.2f; τ2 = %.2f)" % (l, tau1, tau2)

########## Função MiDA
def raw_softmax(x):
    return np.log(1 + np.exp(x))

def mida(l=1, tau=1):
    z = np.arange(-5, 5, .01)
    t = z * bhsa(l,tau,raw_softmax(z))[1]
    return z, t, "Função MiDA (λ = %.2f; τ = %.2f)" % (l, tau)

def mida_d(l=1, tau=1):
    z = np.arange(-5, 5, .01)
    delta = (l*np.exp(z)*z)/(2*(1+np.exp(z)))
    t = bhsa(l, tau, raw_softmax(z))[1] + delta * (((1 + 2*l*raw_softmax(z))/h1_bhsa(l, tau, raw_softmax(z))) + ((1-2*l*raw_softmax(z))/(h2_bhsa(l, tau, raw_softmax(z)))))
    return z, t, "Derivada da função MiDA (λ = %.2f; τ = %.2f)" % (l, tau)

#############################################################
def conf(ax, xlim=2, ylim=None, offsetx=0, offsety=0):
    if ylim is None: ylim = xlim
    ax.set_ylim([-ylim + offsety, ylim + offsety])
    ax.set_xlim([-xlim + offsetx, xlim + offsetx])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.legend()

#############################################################
def step_and_derivative():
    fig, (ax) = plt.subplots(1, 1, figsize=(6,6))
    data, data_d = [step(), step_d()]
    ax.plot(data[0], data[1], label=data[2])
    ax.plot(data_d[0], data_d[1], ls='dashed', lw=1, label=data_d[2])
    conf(ax,xlim=1,offsety=.5)
    # fig.suptitle('Funções de ativação sigmoidais tradicionais')
    plt.show()

def logistic_and_derivative():
    fig, (ax) = plt.subplots(1, 1, figsize=(6,6))
    data, data_d = [logistic(), logistic_d()]
    ax.plot(data[0], data[1], label=data[2])
    ax.plot(data_d[0], data_d[1], ls='dashed', lw=1, label=data_d[2])
    conf(ax, xlim=4,ylim=1,offsety=.5)
    plt.show()

def tanh_and_derivative():
    fig, (ax) = plt.subplots(1, 1, figsize=(6,6))
    data, data_d = [tanh(), tanh_d()]
    ax.plot(data[0], data[1], label=data[2])
    ax.plot(data_d[0], data_d[1], ls='dashed', lw=1, label=data_d[2])
    conf(ax,ylim=1.5,offsety=.3)
    plt.show()

def mish_and_derivative():
    fig, (ax) = plt.subplots(1, 1, figsize=(6,6))
    data, data_d = [mish(), mish_d()]
    ax.plot(data[0], data[1], label=data[2])
    ax.plot(data_d[0], data_d[1], ls='dashed', lw=1, label=data_d[2])
    conf(ax,xlim=4,offsety=1.5)
    plt.show()

def relu_and_derivative():
    fig, (ax) = plt.subplots(1, 1, figsize=(6,6))
    data, data_d = [relu(), relu_d()]
    ax.plot(data[0], data[1], label=data[2])
    ax.plot(data_d[0], data_d[1], ls='dashed', lw=1, label=data_d[2])
    conf(ax, offsety=1,xlim=1.5)
    plt.show()

def lrelu_and_derivative():
    fig, (ax) = plt.subplots(1, 1, figsize=(6,6))
    data, data_d = [lrelu(), lrelu_d()]
    ax.plot(data[0], data[1], label=data[2])
    ax.plot(data_d[0], data_d[1], ls='dashed', lw=1, label=data_d[2])
    conf(ax, offsety=1,xlim=1.5)
    plt.show()

def apl_and_derivative():
    fig, (ax) = plt.subplots(1, 1, figsize=(6,6))
    data, data_d = [apl(), apl_d_a02_b0()]
    ax.plot(data[0], data[1], label=data[2])
    ax.plot(data_d[0], data_d[1], ls='dashed', lw=1, label=data_d[2])
    conf(ax,offsety=1)
    plt.show()

def shrelu_and_derivative():
    fig, (ax) = plt.subplots(1, 1, figsize=(6,6))
    for tau in [0.05, 0.25, 1]:
        data, data_d = [shrelu(tau), shrelu_d(tau)]
        ax.plot(data[0], data[1], label=data[2])
        ax.plot(data_d[0], data_d[1], ls='dashed', lw=1, label=data_d[2])
    conf(ax, offsety=1,xlim=1.5)
    plt.show()

def bhsa_and_derivative():
    fig, (ax) = plt.subplots(1, 1, figsize=(6,6))
    for l, tau in [(1, 1), (.5, 1), (1, .5)]:
        data, data_d = [bhsa(l, tau), bhsa_d(l, tau)]
        ax.plot(data[0], data[1], label=data[2])
        ax.plot(data_d[0], data_d[1], ls='dashed', lw=1, label=data_d[2])
    conf(ax)
    plt.show()

def bhaa_and_derivative():
    fig, (ax) = plt.subplots(1, 1, figsize=(6,6))
    for l, tau1, tau2 in [(1, .5, .5), (1, .5, .1), (1, .1, .5)]:
        data, data_d = [bhaa(l, tau1, tau2), bhaa_d(l, tau1, tau2)]
        ax.plot(data[0], data[1], label=data[2])
        ax.plot(data_d[0], data_d[1], ls='dashed', lw=1, label=data_d[2])
    conf(ax)
    plt.show()

def mida_and_derivative():
    fig, (ax) = plt.subplots(1, 1, figsize=(6,6))
    for l, tau in [(1, 1), (.5, 1), (1, .5)]:
        data, data_d = [mida(l, tau), mida_d(l, tau)]
        ax.plot(data[0], data[1], label=data[2])
        ax.plot(data_d[0], data_d[1], ls='dashed', lw=1.25, label=data_d[2])
    conf(ax,xlim=3,offsety=1.5)
    plt.show()

def bhata_only():
    fig, (ax) = plt.subplots(1, 1, figsize=(6,6))
    for l, tau1, tau2 in [(1, .5, .1), (1, .1, .5)]:
        data = bhata(l, tau1, tau2)
        ax.plot(data[0], data[1], label=data[2])
    conf(ax)
    plt.show()

def bhana_only():
    fig, (ax) = plt.subplots(1, 1, figsize=(6,6))
    for l, tau1, tau2 in [(1, .75, .1), (1, .1, .75)]:
        data = bhana(l, tau1, tau2)
        ax.plot(data[0], data[1], label=data[2])
    conf(ax)
    plt.show()

# step_and_derivative()
# logistic_and_derivative()
# tanh_and_derivative()
# mish_and_derivative()
# relu_and_derivative()
# lrelu_and_derivative()
# apl_and_derivative()
# shrelu_and_derivative()
# bhsa_and_derivative()
# bhaa_and_derivative()
# mida_and_derivative()
# bhata_only()
bhana_only()
