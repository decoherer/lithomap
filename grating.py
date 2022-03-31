
import numpy as np
from numpy import pi,sqrt,sin,cos,ceil
# from wave import Wave

def isvalid(g0,g1=None):
    bs = g0 if g1 is None else grating2bars(g0,g1)
    return all([bi<bj for bi,bj in zip(bs[:-1],bs[1:])])
def grating2bars(starts,ends):
    return [b for p in zip(starts,ends) for b in p]
def bars2grating(bs):
    return bs[0::2],bs[1::2]
def grating2xy(starts,ends,delta=0): # convert to plottable format, also used for piecewise linear ft
    # assume starts[n]<ends[n] and ends[n]<starts[n+1] # starts,ends,delta = [20,40,60],[25,45,65],1 # x = [q+r for p in zip(starts,ends) for q in p for r in [0,0]]         # [20,20,25,25,40,40,45,45,60,60,65,65]
    x = [q+r for p in zip(starts,ends) for q in p for r in [-delta,delta]]  # [19,21,24,26,39,41,44,46,59,61,64,66]
    y = [q for p in zip(starts,ends) for q in [0,1,1,0]]                    # [ 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
    return x,y
def grating2wave(starts,ends,delta=0):
    x,y = grating2xy(starts,ends,delta)
    return Wave(y,x)
def gratingperiod2wave(starts,ends,delta=0):
    y = np.array(starts[1:]) - np.array(starts[:-1])
    return Wave(y,starts[1:])
def ftgrating(starts,ends,p0=None,dp=None,normalize=True,res=2001):
    p0 = (np.array(starts[1:])/2-np.array(starts[:-1])/2+np.array(ends[1:])/2-np.array(ends[:-1])/2).mean() if p0 is None else p0
    dp = p0/10 if dp is None else dp
    x,y = grating2xy(starts,ends)
    w = Wave(y,x)
    wp = np.linspace(p0-dp/2,p0+dp/2,res)
    wf = Wave(abs(w.ft(1/wp)),wp)
    if normalize:
        return wf/wf[wf.pmax()] # print(wf[wf.pmax()])
    return wf
def grating(period,dc,padx,padcount=1,gapx=0,phasex=0,x0=0,apodize=None): # don't use phasex for electrode, not tested yet
    # assert 0==phasex, 'phasex implementation not yet tested'
    #if dev: period = periodmag*period
    barcount = int(np.ceil(1.*padcount*padx/period))
    barstarts = [x0+phasex+i*period for i in range(barcount)] # startx of each bar
    assert 0<dc and dc<=1
    if gapx:
        def isingap(x): return x%padx + period*dc > padx-gapx # remove if bar end is past gap start
        barstarts = [x for x in barstarts if not isingap(x)]
    barends = [x+period*dc for x in barstarts] # endx of each bar
    if apodize in [0,1]: # backwards compatibility with Dec20A and earlier
        def tri(x): return 1-2*abs(x-0.5)
        def trap(x): return np.minimum(1,2*tri(x))
        barstarts,barends = dropsmallbars(*apodizebars(barstarts,barends,afunc=trap),tolerance=0)
        return barstarts,barends
    apodize = {0:None,1:'trapezoidal'}[apodize] if apodize in [0,1] else apodize # keep backwards compatible with Dec20A mask
    if apodize is not None:
        assert 0.5==dc, 'apodization assumes 50% duty cycle'
        assert apodize in 'triangle,trapezoidal,asintriangle,asingauss23'.split(',')
        # # if apodize is an apodizing function, use it as is, else use a trapezoidal apodization function
        # if apodize in [1,True]:
        #     def tri(x): return 1-2*abs(x-0.5)
        #     def trap(x): return np.minimum(1,2*tri(x))
        #     apodize = trap
        barstarts,barends = dropsmallbars(*apodizebars(barstarts,barends,apodize=apodize),tolerance=0)
    return barstarts,barends
def mergetouchingbars(barstarts,barends,tolerance=1.0): # <1.0um won't resolve lithographically and be a resistive gap
    a,b = [barstarts[0]],[]
    for i in range(len(barstarts)-1):
        if barstarts[i+1]-barends[i]>tolerance: # if barstarts[i+1]!=barends[i]:
            a,b = a+[barstarts[i+1]],b+[barends[i]]
    b += [barends[-1]]
    return a,b
def dropsmallbars(barstarts,barends,tolerance=1.0):
    a,b = [],[]
    for i in range(len(barstarts)):
        if barends[i]-barstarts[i]>tolerance:
            a,b = a+[barstarts[i]],b+[barends[i]]
    return a,b
def overpolebars(barstarts,barends,op=1.0): # positive op → make bars bigger, negative op → make bars smaller
    if callable(op): # op is overpole function, op(barsize) = amount of overpole
        dxs = [op(b1-b0) for b0,b1 in zip(barstarts,barends)] 
        return [a-dx/2 for a,dx in zip(barstarts,dxs)],[b+dx/2 for b,dx in zip(barends,dxs)]
    else: # op = amount of overpole
        return [a-op/2 for a in barstarts],[b+op/2 for b in barends]  # can be negative bar length, so call dropsmallbars()
def invertbarsgaps(starts,ends):
    return ends[:-1],starts[1:]
def apodizebars(starts,ends,afunc=None,apodize=None): # typical: afunc(0)=2, afunc(0.5)=1, afunc(1)=2 (where 0,1,2 = 0%,50%,100% duty cycle)
    # functions that return 0 to 1 to 0 for x from 0 to 1
    def tri(x): return 1-2*abs(x-0.5) # 0 to 1 to 0
    def trap(x): return np.minimum(1,2*tri(x))
    def gauss(x,σ): return np.exp( -0.5 * (x-0.5)**2 / σ**2 )
    # functions that return duty cycle 1 to 0.5 to 1
    def triangledutycycle(x): return 1-0.5*tri(x)
    def trapezoidaldutycycle(x): return 1-0.5*trap(x)
    def asintriangle(x):
        dc = np.arcsin(tri(x))/np.pi # duty cycle from 0 to 0.5 to 0
        return 1-dc                  # duty cycle from 1 to 0.5 to 1
    def asingauss(x,σ): return 1-np.arcsin(gauss(x,σ))/np.pi
    if afunc is None and apodize is not None:
        afunc = {
            'triangle':     lambda x: 2 * triangledutycycle(x),
            'trapezoidal':  lambda x: 2 * trapezoidaldutycycle(x),
            'asintriangle': lambda x: 2 * asintriangle(x),
            'asingauss23':  lambda x: 2 * asingauss(x,0.23),
        }[apodize]
    a,b = np.array(starts),np.array(ends)
    c = a/2+b/2                 # c = centers
    w0 = b-a                    # old bar widths
    fp = (c-c[0])/(c[-1]-c[0])  # fp = fractionalposition
    w1 = w0*afunc(fp)           # new bar widths
    return c-w1/2,c+w1/2        # new starts,ends
def kchirpgrating(p0,p1,dc,padx,padcount=1,gapx=0,phase=0,x0=0): # https://www.gaussianwaves.com/2014/07/chirp-signal-frequency-sweeping-fft-and-power-spectral-density/
    assert 0==phase, 'phase not implemented'
    k0,k1 = 2*pi/p0,2*pi/p1
    barcount = int(padcount*padx*(k1+k0)/4/pi)
    L = 4*pi*barcount/(k1+k0)
    def k(x): return 0.5*(k1-k0)*x/L + k0
    def m(x): return x*k(x)/2/pi
    def xx(m): return ( sqrt(m*4*pi*(k1-k0)/L + k0*k0) - k0 )*L/(k1-k0)
    ms = np.arange(barcount+1)  # period number
    xs = xx(ms)                 # start/end location of each period
    periods = np.diff(xs)       # length of each period
    barstarts = xs[:-1]
    if gapx:
        def isingap(x,period): return x%padx + period*dc > padx-gapx # remove if bar end is past gap start
        barstarts,periods = zip(*[(x,p) for x,p in zip(barstarts,periods) if not isingap(x,p)])
    barends = [x+dc*p for x,p in zip(barstarts,periods)]
    return barstarts,barends
def entwinedgrating(period1,period2,padcount,gx):
    return interleavedgrating(period1,period2,padcount,gx)
def interleavedgrating(period1,period2,padcount,gx,overpole=0): # todo: modify for non-50% duty-cycle, mergebars if gap is less than minbarsize
    res,minbarsize = 0.1,0.6+overpole
    xx = np.arange(0,1.*padcount*gx,res)
    yy = np.sign( np.sin(2*np.pi*xx/period1) + np.sin(2*np.pi*xx/period2) )
    barstarts,barends = [],[]
    if 1==yy[0]: barstarts += [xx[0]]
    for p in range(1,len(xx)):  #print yy[p-1],yy[p],(yy[p-1]<1 and yy[p]==1),(-1<yy[p-1] and yy[p]==-1),xx[p]
        if yy[p-1]<1 and yy[p]==1:
            barstarts += [xx[p]]
        if -1<yy[p-1] and yy[p]==-1 and not 1==p:
            barends += [xx[p]]
    if 1==yy[-1]: barends += [xx[-1]]
    if minbarsize:
        bars = [(x,y) for x,y in zip(barstarts,barends) if y-x>minbarsize]
        barstarts,barends = zip(*bars)
    return barstarts,[b-overpole for b in barends]
def alternatinggrating2(g0,g1,L,repeats=1):
    def keepit(bar,j):
        x0,x1 = bar
        if 0==j:
            return x0%(L/repeats)<0.5*(L/repeats) and x1%(L/repeats)<0.5*(L/repeats) and 0<=x0 and x1<=L
        else:
            return x0%(L/repeats)>0.5*(L/repeats) and x1%(L/repeats)>0.5*(L/repeats) and 0<=x0 and x1<=L
    b0 = [b for b in zip(*g0) if keepit(b,0)]
    b1 = [b for b in zip(*g1) if keepit(b,1)]
    bars = sorted(b0+b1)
    starts,ends = zip(*bars)
    return starts,ends
def alternatinggrating(p0,p1,dc0,dc1,padcount,gx,repeats=1):
    def bars(k,ends=False): # k = section number
        p,dc = (p0,dc0) if 0==k%2 else (p1,dc1) # p0 for even sections, p1 for odd
        x0,x1 = k*padcount*gx/2./repeats,(k+1)*padcount*gx/2./repeats # startx,endx of section
        n0,n1 = np.ceil(x0/p),np.floor(x1/p)-1 # number of first,last bar (assuming bar 0 is at x=0)
        xs = p*np.arange(n0,n1+1,1) + ends*p*dc # there will always be a small unpoled part at the start and/or end of each section: ___■■__■■__■■__■■____
        # assert xs[0]<x0+p+ends*p*dc, f'{xs[0]} {x0} {p} {x0/p} {ceil(x0/p)}'
        # assert xs[-1]>=x1-2*p, f'{xs[-1]} {x1} {p} {x1/p} {floor(x1/p)} {k} {ends}'
        assert all([x0<=x<=x1 for x in xs]), f'{xs[-1]} {x1} {p} {x1/p} {floor(x1/p)} {k} {ends}'
        # for x in xs: assert x0<=x<=x1, f'{x0},{x},{x1} n0*p:{n0*p} n0:{n0} n1:{n1} section number:{k} ends:{ends}'
        return xs
    b0s = [b for n in range(2*repeats) for b in bars(n,ends=False)]
    b1s = [b for n in range(2*repeats) for b in bars(n,ends=True )]
    return b0s,b1s
def piphasegrating(g0,g1): # g0,g1 = starts,ends
    n = len(g1)//2
    a0,a1 = g0[:n],g1[:n]
    b0,b1 = g0[n:],g1[n:]
    b0,b1 = invertbarsgaps(b0,b1)
    return a0+b0,a1+b1
def reverse(L,g0,g1):
    return [L-g for g in g1[::-1]],[L-g for g in g0[::-1]]
def phasegrating(g0,g1,n,x0=0,x1=None):
        ### recommended method for one less missing bar at end
        # gg = grating(100,0.5,300,padcount=1,gapx=0,phasex=0,x0=0)
        # hh = phasegrating(*reverse(300,*gg),2)
    x1 = x1 if x1 is not None else g1[-1]
    xs = [x0+(x1-x0)*i/n for i in range(1,n)]
    bs = sorted(grating2bars(g0,g1) + xs)
    bs = bs[:len(bs)//2*2]
    gg = bars2grating(bs)
    if isvalid(bs):
        return gg
    return mergetouchingbars(*dropsmallbars(*gg,tolerance=0),tolerance=0)
def flipgratingbars(signs,period): # note signed segments are length period/2
    def lacc(*args,**kwargs):
        from itertools import accumulate
        return list(accumulate(*args,**kwargs))
    zs = lacc([period/2. for _ in signs],initial=0)
    signs = [+1] + signs + [+1]
    sign0s,sign1s = signs[:-1],signs[1:] # signs before,after position z
    assert len(zs)==len(signs)-1==len(sign0s)==len(sign1s)
    barstarts = [z for z,sign0,sign1 in zip(zs,sign0s,sign1s) if sign0==+1 and sign1==-1]
    barends = [z for z,sign0,sign1 in zip(zs,sign0s,sign1s) if sign0==-1 and sign1==+1]
    assert barstarts==[z for z,sign0,sign1 in zip(zs,sign0s,sign1s) if sign0>sign1]
    return mergetouchingbars(barstarts,barends)
def spectralcomb(g0,g1,r,L=None): # r=offratio, removes all bars in middle: ■■____■■
    L = L if L is not None else g1[-1]
    x0,x1 = (1-r)*L/2,(1+r)*L/2
    ps = [(a,b) for a,b in zip(g0,g1) if a<x0 or x1<b]
    ps = [(a,min(b,x0) if a<x0 else b) for a,b in ps]
    ps = [(max(a,x1) if x1<b else a,b) for a,b in ps]
    return list(zip(*ps))
def Λ2λft(w):
    from sellmeier import polingperiod
    λs = np.linspace(1300,1700,201)
    Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
    return Wave(w.y,[Wave(Λs,λs).xaty(Λ) for Λ in w.x.y])
def electrodegratingft(barstarts,barends):
    a,b = np.array(barstarts),np.array(barends)
    w = 1-grating2wave(a,b)
    # w.plot()
    Λ = 0.5*np.mean(a[1:]-a[:-1]) + 0.5*np.mean(b[1:]-b[:-1])
    L = b[-1]-a[0]
    dΛ = 10*Λ*Λ/L
    vx = np.linspace(Λ-dΛ,Λ+dΛ,501)
    def waveft(w):
        return Wave(abs(w.ft(1/vx,norm=True)),vx)
    w0 = grating2wave(*grating(Λ,0.5,L,padcount=1,gapx=0))
    us = [waveft(u) for u in (w0,w)]
    Wave.plots(*[u/us[0](Λ) for u in us],x='Λ (µm)',y='relative nonlinear response')


