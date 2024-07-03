from ipywidgets import interact, widgets
from IPython.display import Image, clear_output
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

from sklearn.linear_model import ElasticNet, LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import warnings

out = widgets.Output(layout=widgets.Layout(height='300px'))

def make_linear(n_samples=100, coef=[3, 0], noise=0.1, span=(0, 5), test_size=0, random_state=None):
    assert len(coef)==2
    return make_poly(n_samples, coef, noise, span, test_size, random_state=random_state)


def make_quadratic(n_samples=100, coef=[1, -3.5, 10], noise=0.1, span=(0, 5), test_size=0, random_state=None):
    assert len(coef)==3
    return make_poly(n_samples, coef, noise, span, test_size, random_state=random_state)
    

def make_poly(n_samples=100, coef=[-0.2, 2, -6, 4.5, 20], noise=0.01, span=(0, 5), test_size=0, random_state=None):
    assert test_size>=0   
    if random_state is not None:
        np.random.seed(random_state)    
    X = np.linspace(span[0], span[1], n_samples)
    y = noise * np.random.randn(n_samples)    
    for i, c in enumerate(coef, 1):
        y += c*X**(len(coef)-i)
    if test_size > 0:
        if random_state is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        X_train = X_train.reshape(-1,1)
        X_test = X_test.reshape(-1,1)
        return X_train, X_test, y_train, y_test
    else:
        return X, y
    

def make_nonlinear(n_samples=100, coef=[5, 0.5, 10, 5, 0.2], noise=0.5, span=(0, 5), test_size=0, random_state=None):
    assert len(coef)==5
    assert test_size>=0  
    if random_state is not None:
        np.random.seed(random_state)    
    X = np.linspace(span[0], span[1], n_samples)
    y = noise * np.random.randn(n_samples)
    y += coef[0]*(1-coef[1]*X)**2*np.exp(-X**2) - coef[2]*(X/coef[3] - X**3)*np.exp(-X**2) - coef[4]*np.exp(-X**2) 
    if test_size > 0:
        if random_state is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        X_train = X_train.reshape(-1,1)
        X_test = X_test.reshape(-1,1)
        return X_train, X_test, y_train, y_test
    else:
        return X, y
    

def make_sine(n_samples=100, coef=[1, 1], noise=0.5, span=(0, 5), test_size=0, random_state=None):
    assert len(coef)==2
    assert test_size>=0
    if random_state is not None:
        np.random.seed(random_state)
    X = np.linspace(span[0], span[1], n_samples)
    y = noise * np.random.randn(n_samples)
    y += coef[0]*np.sin(coef[1]*X)
    if test_size > 0:
        if random_state is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        X_train = X_train.reshape(-1,1)
        X_test = X_test.reshape(-1,1)
        return X_train, X_test, y_train, y_test
    else:
        return X, y    
    

def linreg(a=1, b=0):
    def _simul(a=a, b=b):
        with out:
            out.clear_output(wait=True)
            plt.figure(figsize=(6,6))
            plt.scatter(X, y, c="b")            
            plt.plot(space, a*space+b, "r--")
            plt.axis('equal')
            plt.xlabel("X"); plt.ylabel("y"); plt.xlim(0, 5); plt.ylim(0, 5)
        plt.show()
    space = np.linspace(0, 5, 10)
    X, y = make_linear(coef=[-0.5, 3], noise=0.2)    
    interact(_simul, a=(-3, 3, 0.1), b=(-3, 3, 0.1));
    

def linreg_loss(a=1, b=0):
    def _simul(a=a, b=b): 
        with out:
            out.clear_output(wait=True)
            plt.figure(figsize=(6,6))
            plt.scatter(X, y, c="b")
            plt.plot(space, a*space+b, "r--")
            pred = a*X+b
            cost = np.mean((pred-y)**2)
            plt.title(fr"$y={a}X+{b}$ | MSE: {cost:.2f}", fontsize=15)        
            for x, p, d in zip(X, pred, y):
                plt.plot([x, x], [p, d], 'g-')
            plt.axis('equal')
            plt.xlabel("X"); plt.ylabel("y"); plt.xlim(0, 5); plt.ylim(0, 5)
        plt.show()    
    space = np.linspace(0, 5, 10)
    X, y = make_linear(n_samples=10, coef=[0.5, 1], noise=0.5)    
    interact(_simul, a=(0, 2, 0.1), b=(0, 2, 0.1));     
    

def linreg_a(a=0.1, show_loss_line=False, show_loss=False, show_tuning=False):
    def _simul(a=a, show_loss_line=show_loss_line, show_loss=show_loss, show_tuning=show_tuning):
        with out:
            out.clear_output(wait=True)
            plt.figure(figsize=(15,6))
            plt.subplot(121)
            plt.scatter(X, y, c="b")    
            plt.plot(space, a*space, "r--")
            pred = a*X
            cost = np.mean((pred-y)**2)
            plt.title(fr"$y={a}X$ | MSE: {cost:.2f}", fontsize=15)
            plt.axis('equal')
            plt.xlabel("X"); plt.ylabel("y"); plt.xlim(0, 5); plt.ylim(0, 5)
            if show_loss_line:
                for x, p, d in zip(X, pred, y):
                    plt.plot([x, x], [p, d], 'g-')            
                    
            if show_loss:
                plt.subplot(122)
                plt.scatter(a, cost, s=100, c="green", zorder=2, label="Current MSE (Loss)")
                plt.title("Parameter Tuning", fontsize=15)
                plt.xlabel("a"); plt.ylabel("MSE"); plt.xlim(0, 3); plt.ylim(-5, 40)
                plt.legend()
            
            if show_loss & show_tuning:
                params = np.arange(0, 3, 0.1)
                cost_list = [np.mean((a*X-y)**2) for a in params]
                plt.scatter(params, cost_list, s=20, c='r', zorder=1, alpha=0.5)
        plt.show()
    X, y = make_linear(coef=[1, 0], noise=0.2)
    space = np.linspace(0, 6, 10)    
    interact(_simul, a=(0, 3, 0.1));
    

def linreg_ab(a=1, b=0, show3d=False, elev=30, azim=-65):
    def _simul(a=a, b=b, show3d=show3d):
        with out:
            out.clear_output(wait=True)
            if not show3d:
                fig, ax = plt.subplots(1, 2, figsize=(15, 6))
                ax[0].scatter(X, y, c="b")    
                ax[0].plot(space, a*space+b, "r--")
                cost = np.mean( ((a*X+b)-y)**2 )
                ax[0].set_title(fr"$y={a}X+{b}$ | MSE: {cost:.2f}", fontsize=15)
                ax[0].set(xlim=(0, 10), ylim=(0, 10))        
                
                ax[1].contour(A, B, Z, cmap='nipy_spectral', levels=np.logspace(-1, 2, 10))
                ax[1].scatter(a, b, c="green", s=75)
                ax[1].set(xlim=(0, 2), ylim=(0, 5), title="Loss Plane")
            else:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(A, B, Z, linewidth=0, cmap='nipy_spectral')
                ax.view_init(elev=elev, azim=azim)
                ax.contour(A, B, Z, zdir='z', offset=-5, cmap='nipy_spectral', levels=np.logspace(-1, 2, 10))    
                ax.set(xlabel="a", ylabel= "b", zlabel= "Cost (MSE)", zlim=-5)    
                ax.grid(True)
        plt.show()

    def _calc_cost(A, B, X, y):
        sz = len(X)
        nb, na = A.shape
        A = np.expand_dims(A, axis=2).repeat(sz, axis=2)
        B = np.expand_dims(B, axis=2).repeat(sz, axis=2)
        X = X.reshape(1,1,-1).repeat(nb, axis=0).repeat(na, axis=1)
        Y = y.reshape(1,1,-1).repeat(nb, axis=0).repeat(na, axis=1)
        return np.mean((A*X+B - Y)**2, axis=2)
    
    space = np.linspace(0, 10, 10)
    A = np.linspace(0, 2, 101)
    B = np.linspace(0, 5, 101)
    A, B = np.meshgrid(A, B)
    X, y = make_linear(coef=[0.75, 2], noise=0.3, span=(0, 10))
    Z = _calc_cost(A, B, X, y)
    interact(_simul, a=(0,2,0.25), b=(0,5,0.25));


def reg_ridge_lasso(degree=1, alpha=0, regularization="L1 / LASSO"):
    def _simul(degree=degree, alpha=alpha, regularization=regularization):
        try:
            alpha = float(alpha)
            if alpha==0:
                lr = Pipeline([
                    ("poly", PolynomialFeatures(degree)),
                    ("lr", LinearRegression())
                ])            
            elif regularization=="L1 / LASSO / Sparsity":
                lr = Pipeline([
                    ("poly", PolynomialFeatures(degree)),
                    ("lr", Lasso(alpha=alpha, max_iter=5000, tol=1e-8))
                ])
            elif regularization=="L2 / Ridge/ Simplicity":
                lr = Pipeline([
                    ("poly", PolynomialFeatures(degree)),
                    ("lr", Ridge(alpha=alpha, max_iter=5000, tol=1e-8))
                ])                

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lr.fit(X_train, y_train)

            with out:
                out.clear_output(wait=True)
                plt.figure(figsize=(8,6))
                plt.title(f"R2_train: {lr.score(X_train, y_train):.2f} | R2_test: {lr.score(X_test, y_test):.2f}", fontsize=15)
                plt.scatter(X_train, y_train, c="b", s=10)
                plt.scatter(X_test, y_test, c="r", marker="x")    
                plt.plot(space, lr.predict(space), "k--", linewidth=1);
                plt.xlim(0, 5)
                plt.ylim(-1.5, 1.5)
            plt.show()
        except ValueError:
            with out:
                out.clear_output(wait=True)
                plt.figure(figsize=(8,6))
                plt.title(f"R2_train: - | R2_test: -", fontsize=15)
                plt.scatter(X_train, y_train, c="b", s=10)
                plt.scatter(X_test, y_test, c="r", marker="x")    
                plt.xlim(0, 5)
                plt.ylim(-1.5, 1.5)  
            plt.show()

    X_train, X_test, y_train, y_test = make_sine(noise=0.3, test_size=0.2, span=(0, 5))
    space = np.linspace(0, 6, 100).reshape(-1, 1)
    interact(_simul, degree=(1,14,1), regularization=["L1 / LASSO / Sparsity", "L2 / Ridge/ Simplicity"], alpha="0");    
    

def reg_elastic(degree=1, l1_ratio=0, alpha=0):
    def _simul(degree=degree, l1_ratio=l1_ratio, alpha=alpha):
        try:
            alpha = float(alpha)
            if alpha==0:
                lr = Pipeline([
                    ("poly", PolynomialFeatures(degree)),
                    ("lr", LinearRegression())
                ])            
            else:
                lr = Pipeline([
                    ("poly", PolynomialFeatures(degree)),
                    ("lr", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol=1e-8))
                ])            

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lr.fit(X_train, y_train)

            with out:
                out.clear_output(wait=True)
                plt.figure(figsize=(8,6))
                plt.title(f"R2_train: {lr.score(X_train, y_train):.2f} | R2_test: {lr.score(X_test, y_test):.2f}", fontsize=15)
                plt.scatter(X_train, y_train, c="b", s=10)
                plt.scatter(X_test, y_test, c="r", marker="x")    
                plt.plot(space, lr.predict(space), "k--", linewidth=1);
                plt.xlim(0, 5)
                plt.ylim(-1.5, 1.5)
            plt.show()
        except ValueError:
            with out:
                out.clear_output(wait=True)
                plt.figure(figsize=(8,6))
                plt.title(f"R2_train: - | R2_test: -", fontsize=15)
                plt.scatter(X_train, y_train, c="b", s=10)
                plt.scatter(X_test, y_test, c="r", marker="x")    
                plt.xlim(0, 5)
                plt.ylim(-1.5, 1.5)
            plt.show()            
        
    X_train, X_test, y_train, y_test = make_sine(noise=0.3, test_size=0.2, span=(0, 5))
    space = np.linspace(0, 6, 100).reshape(-1, 1)
    interact(_simul, degree=(1,14,1), l1_ratio=(0,1,0.2), alpha="0");
    
def reg_coef(degree=1, noise=0.2, test_size=0.2):
    def _simul(degree=degree):
        model = Pipeline([
                ("poly", PolynomialFeatures(degree)),
                ("lr", LinearRegression())
            ])
         
        model.fit(X_train, y_train)

        with out:
            out.clear_output(wait=True)
            plot_coef(X_train, X_test, y_train, y_test, model)
        plt.show()

    X_train, X_test, y_train, y_test = make_sine(noise=noise, test_size=test_size)
    interact(_simul, degree=(1,36,1));

def plot_coef(X1, X2, y1, y2, model, span=(0, 5)):

    X_pred = np.linspace(span[0], span[1], 100).reshape(-1, 1)
    name = model.named_steps.poly.get_feature_names_out()
    coef = model.named_steps.lr.coef_
    
    plt.figure(figsize=(15,10))
    
    plt.subplot(221)
    plt.scatter(X1, y1, s=10, c='r')
    plt.title(f"R2_train: {model.score(X1, y1):.3f}")
    plt.plot(X_pred, model.predict(X_pred), 'k-')

    plt.subplot(222)
    plt.scatter(X2, y2, s=10, c='r')
    plt.plot(X_pred, model.predict(X_pred), 'k-')
    plt.title(f"R2_test: {model.score(X2, y2):.3f}");
    
    plt.subplot(212)    
    plt.bar(range(len(coef)), coef, color='b')
    plt.xticks(range(len(coef)), name)