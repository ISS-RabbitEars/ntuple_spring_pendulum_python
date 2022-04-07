import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation


def integrate(ic, ti, p):
	gc, m, k, req, xp, yp = p
	ic_list = ic

	sub = {g:gc, Xp:xp, Yp:yp}
	for i in range(m.size):
		sub[M[i]] = m[i]
		sub[K[i]] = k[i]
		sub[Req[i]] = req[i]
		sub[R[i]] = ic_list[4 * i]
		sub[Rdot[i]] = ic_list[4 * i + 1]
		sub[THETA[i]] = ic_list[4 * i + 2]
		sub[THETAdot[i]] = ic_list[4 * i + 3]


	diff_eq = []
	for i in range(m.size):
		diff_eq.append(ic_list[4 * i + 1])
		diff_eq.append(A[i].subs(sub))
		diff_eq.append(ic_list[4 * i + 3])
		diff_eq.append(ALPHA[i].subs(sub))

	print(ti)

	return diff_eq

N = 2

Xp, Yp, g, t = sp.symbols('Xp Yp g t')
M = sp.symbols('M0:%i'%N)
K = sp.symbols('K0:%i'%N)
Req = sp.symbols('Req0:%i'%N)
R = dynamicsymbols('R0:%i'%N)
THETA = dynamicsymbols('THETA0:%i'%N)

Rdot = np.asarray([i.diff(t, 1) for i in R])
Rddot = np.asarray([i.diff(t, 2) for i in R])
THETAdot = np.asarray([i.diff(t, 1) for i in THETA])
THETAddot = np.asarray([i.diff(t, 2) for i in THETA])

X = []
Y = []
X.append(Xp + R[0] * sp.cos(THETA[0]))
Y.append(Yp + R[0] * sp.sin(THETA[0]))
for i in range(1, N):
	X.append(X[i-1] + R[i] * sp.cos(THETA[i]))
	Y.append(Y[i-1] + R[i] * sp.sin(THETA[i]))
X = np.asarray(X)
Y = np.asarray(Y)

Xdot = np.asarray([i.diff(t, 1) for i in X])
Ydot = np.asarray([i.diff(t, 1) for i in Y])

dR = []
dR.append(sp.sqrt((X[0] - Xp)**2 + (Y[0] - Yp)**2))
for i in range(1, N):
	dR.append(sp.sqrt((X[i] - X[i-1])**2 + (Y[i] - Y[i-1])**2))
dR = np.asarray(dR)

T = sp.simplify(sp.Rational(1, 2) * sum(M * (Xdot**2 + Ydot**2)))
V1 = sp.simplify(sp.Rational(1, 2) * sum(K * (dR - Req)**2))
V2 = sp.simplify(g * sum(M * Y))
V = V1 + V2

L = T - V

dLdR = np.asarray([L.diff(i, 1) for i in R])
dLdRdot = np.asarray([L.diff(i, 1) for i in Rdot])
ddtdLdRdot =np.asarray([i.diff(t, 1) for i in dLdRdot])
dLR = ddtdLdRdot - dLdR

dLdTHETA = np.asarray([L.diff(i, 1) for i in THETA])
dLdTHETAdot = np.asarray([L.diff(i, 1) for i in THETAdot])
ddtdLdTHETAdot = np.asarray([i.diff(t, 1) for i in dLdTHETAdot])
dLTHETA = ddtdLdTHETAdot - dLdTHETA

dL = np.append(dLR,dLTHETA).tolist()
ddot = np.append(Rddot,THETAddot).tolist()

sol = sp.solve(dL,ddot)

A = [sp.simplify(sol[i]) for i in Rddot]
ALPHA = [sp.simplify(sol[i]) for i in THETAddot]

#------------------------------------------------- 

gc = 9.8
ma,mb = [1, 1]
ka,kb = [25, 25]
reqa,reqb = [2, 2]
roa,rob = [2, 2] 
voa,vob = [0, 0]
thetaoa,thetaob = [10, 90] 
omegaoa,omegaob = [0, 0]
xp,yp = [1, 1] 
tf = 240 

m = np.linspace(ma, mb, N)
k = np.linspace(ka, kb, N)
req = np.linspace(reqa, reqb, N)
ro = np.linspace(roa, rob, N)
vo = np.linspace(voa, vob, N)
theta = np.linspace(thetaoa, thetaob, N) * np.pi/180
omega = np.linspace(omegaoa, omegaob, N) * np.pi/180

p = gc, m, k, req, xp, yp

ic = []
for i in range(N):
	ic.append(ro[i])
	ic.append(vo[i])
	ic.append(theta[i])
	ic.append(omega[i])

nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

rth = odeint(integrate, ic, ta, args = (p,))


x = np.zeros((N,nframes))
y = np.zeros((N,nframes))
for i in range(nframes):
	sub = {Xp:xp, Yp:yp}
	for j in range(N):
		sub[R[j]] = rth[i,4 * j]
		sub[THETA[j]] = rth[i,4 *j + 2]
		x[j][i] = X[j].subs(sub)
		y[j][i] = Y[j].subs(sub)

ke = np.zeros(nframes)
pe = np.zeros(nframes)
for i in range(nframes):
	ke_sub = {Xp:xp, Yp:yp}
	pe_sub = {g:gc, Xp:xp, Yp:yp}
	for j in range(N):
		ke_sub[M[j]] = m[j]
		ke_sub[R[j]] = rth[i,4 * j]
		ke_sub[Rdot[j]] = rth[i,4 * j + 1]
		ke_sub[THETA[j]] = rth[i,4 * j + 2]
		ke_sub[THETAdot[j]] = rth[i,4 * j + 3]
		pe_sub[M[j]] = m[j]
		pe_sub[K[j]] = k[j]
		pe_sub[Req[j]] = req[j]
		pe_sub[R[j]] = rth[i,4 * j]
		pe_sub[Rdot[j]] = rth[i,4 * j + 1]
		pe_sub[THETA[j]] = rth[i,4 * j + 2]
		pe_sub[THETAdot[j]] = rth[i,4 * j + 3]
	ke[i] = T.subs(ke_sub)
	pe[i] = V.subs(pe_sub)
E = ke + pe

#---------------------------------------------------

xmax = x.max() if x.max() > xp else xp
xmin = x.min() if x.min() < xp else xp
ymax = y.max() if y.max() > yp else yp
ymin = y.min() if y.min() < yp else yp

mrf = 1 / 30
drs = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
mr = mrf * drs
mra = mr * m/max(m)

xmax += 2 * max(mra)
xmin -= 2 * max(mra)
ymax += 2 * max(mra)
ymin -= 2 * max(mra)

dr = np.zeros((N,nframes))
theta = np.zeros((N,nframes))
dr[0] = np.asarray([np.sqrt((xp - x[0])**2 + (yp - y[0])**2)],dtype=float)
theta[0] = np.asarray([np.arccos((yp-y[0])/dr[0])],dtype=float)
for i in range(1,N):
	dr[i] = np.asarray([np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)],dtype=float)
	theta[i] = np.asarray([np.arccos((y[i-1] - y[i])/dr[i])],dtype=float)
drmax = np.asarray([max(dr[i]) for i in range(N)])
nl = np.asarray(np.ceil(drmax/(2 * mra)),dtype=int)
l = np.zeros((N,nframes))
h = np.zeros((N,nframes))
l[0] = np.asarray([(dr[0][i] - mra[0])/nl[0] for i in range(nframes)],dtype=float)
h[0] = np.asarray([np.sqrt(mra[0]**2 - (0.5 * l[0][i])**2) for i in range(nframes)],dtype=float)
for i in range(1,N):
	l[i] = np.asarray([(dr[i] - (mra[i] + mra[i-1]))/nl[i]],dtype=float) 
	h[i] = np.asarray([np.sqrt(((mra[i] + mra[i-1])/2)**2 - (0.5 * l[i])**2)])
flipa = np.zeros((N,nframes))
flipb = np.zeros((N,nframes))
flipc = np.zeros((N,nframes))
xlo = np.zeros((N,nframes))
ylo = np.zeros((N,nframes))
flipa[0] = np.asarray([-1 if x[0][j]>xp and y[0][j]<yp else 1 for j in range(nframes)])
flipb[0] = np.asarray([-1 if x[0][j]<xp and y[0][j]>yp else 1 for j in range(nframes)])
flipc[0] = np.asarray([-1 if x[0][j]<xp else 1 for j in range(nframes)])
xlo[0] = np.asarray([x[0] + np.sign((yp - y[0]) * flipa[0] * flipb[0]) * mra[0] * np.sin(theta[0])])
ylo[0] = np.asarray([y[0] + mra[0] * np.cos(theta[0])])
for i in range(1,N):
	flipa[i] = np.asarray([-1 if x[i][j]>x[i-1][j] and y[i][j]<y[i-1][j] else 1 for j in range(nframes)])	
	flipb[i] = np.asarray([-1 if x[i][j]<x[i-1][j] and y[i][j]>y[i-1][j] else 1 for j in range(nframes)])
	flipc[i] = np.asarray([-1 if x[i][j]<x[i-1][j] else 1 for j in range(nframes)])
	xlo[i] = np.asarray([x[i] + np.sign((y[i-1] - y[i]) * flipa[i] * flipb[i]) * mra[i] * np.sin(theta[i])])
	ylo[i] = np.asarray([y[i] + mra[i] * np.cos(theta[i])])
xl = np.zeros((N,max(nl),nframes))
yl = np.zeros((N,max(nl),nframes))
for j in range(nl[0]):
	xl[0][j] = xlo[0] + np.sign((yp - y[0])*flipa[0]*flipb[0]) * (0.5 + j) * l[0] * np.sin(theta[0]) - np.sign((yp - y[0])*flipa[0]*flipb[0]) * flipc[0] * (-1)**j * h[0] * np.sin(np.pi/2 - theta[0])
	yl[0][j] = ylo[0] + (0.5 + j) * l[0] * np.cos(theta[0]) + flipc[0] * (-1)**j * h[0] * np.cos(np.pi/2 - theta[0])
for i in range(1,N):
	for j in range(nl[i]):
		xl[i][j] = xlo[i] + np.sign((y[i-1] - y[i])*flipa[i]*flipb[i]) * (0.5 + j) * l[i] * np.sin(theta[i]) - np.sign((y[i-1] - y[i])*flipa[i]*flipb[i]) * flipc[i] * (-1)**j * h[i] * np.sin(np.pi/2 - theta[i])
		yl[i][j] = ylo[i] + (0.5 + j) * l[i] * np.cos(theta[i]) + flipc[i] * (-1)**j * h[i] * np.cos(np.pi/2 - theta[i])
xlf = np.zeros((N-1,nframes))
ylf = np.zeros((N-1,nframes))
for i in range(N-1):
	xlf[i] = x[i] - mra[i] * np.sign((y[i]-y[i+1])*flipa[i+1]*flipb[i+1]) * np.sin(theta[i+1])
	ylf[i] = y[i] - mra[i] * np.cos(theta[i+1])

fig, a=plt.subplots()

def run(frame):
	plt.clf()
	plt.subplot(211)
	for i in range(N):
		circle=plt.Circle((x[i][frame],y[i][frame]),radius=mra[i],fc='xkcd:red')
		plt.gca().add_patch(circle)
	circle=plt.Circle((xp,yp),radius=max(mra)/2,fc='xkcd:cerulean')
	plt.gca().add_patch(circle)
	plt.plot([xl[0][nl[0]-1][frame],xp],[yl[0][nl[0]-1][frame],yp],'xkcd:cerulean')
	for i in range(N):
		plt.plot([xlo[i][frame],xl[i][0][frame]],[ylo[i][frame],yl[i][0][frame]],'xkcd:cerulean')
		for j in range(nl[i]-1):
			plt.plot([xl[i][j][frame],xl[i][j+1][frame]],[yl[i][j][frame],yl[i][j+1][frame]],'xkcd:cerulean')
	for i in range(1,N):
		plt.plot([xl[i][nl[i]-1][frame],xlf[i-1][frame]],[yl[i][nl[i]-1][frame],ylf[i-1][frame]],'xkcd:cerulean')
	plt.title("N-Tuple Spring Pendulum (N=%i)"%N)
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([float(xmin),float(xmax)])
	plt.ylim([float(ymin),float(ymax)])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=1.0)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=1.0)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.5)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('ntuple_spring_pendulum_v2.mp4', writer=writervideo)
plt.show()











