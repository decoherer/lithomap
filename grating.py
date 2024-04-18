import numpy as np
from numpy import pi,sqrt,sin,cos,ceil
import sys
# sys.path.insert(1, '/py/work/sell')
from wavedata import Wave

def isvalid(g0,g1=None):
    bs = g0 if g1 is None else grating2bars(g0,g1)
    return all([bi<bj for bi,bj in zip(bs[:-1],bs[1:])])
def grating2bars(starts,ends):
    return [b for p in zip(starts,ends) for b in p]
def bars2grating(bs):
    return bs[0::2],bs[1::2]
def bars2file(file,bars):
    with open(file if file.endswith('.dat') else file+'.dat','w') as f:
        for b in bars:
            f.write(f'{b:.3f}\n')
def grating2xy(starts,ends,delta=0): # convert to plottable format, also used for piecewise linear ft
    # assume starts[n]<ends[n] and ends[n]<starts[n+1] # starts,ends,delta = [20,40,60],[25,45,65],1 # x = [q+r for p in zip(starts,ends) for q in p for r in [0,0]]         # [20,20,25,25,40,40,45,45,60,60,65,65]
    x = [q+r for p in zip(starts,ends) for q in p for r in [-delta,delta]]  # [19,21,24,26,39,41,44,46,59,61,64,66]
    y = [q for p in zip(starts,ends) for q in [0,1,1,0]]                    # [ 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
    return x,y
# def fixedresgrating2bars(xx,yy): # xx,yy = x position, y grating sign (±1) as np arrays
#     barstarts,barends = [],[]
#     if 1==yy[0]: barstarts += [xx[0]]
#     for p in range(1,len(xx)):  #print yy[p-1],yy[p],(yy[p-1]<1 and yy[p]==1),(-1<yy[p-1] and yy[p]==-1),xx[p]
#         if yy[p-1]<1 and yy[p]==1:
#             barstarts += [xx[p]]
#         if -1<yy[p-1] and yy[p]==-1 and not 1==p:
#             barends += [xx[p]]
#     if 1==yy[-1]: barends += [xx[-1]]
#     return barstarts,barends
def fixedresgrating2bars(xx,yy): # xx,yy = x position, y grating sign (±1) as np arrays
    y0,y1 = min(yy),max(yy)
    barstarts,barends = [],[]
    if y1==yy[0]: barstarts += [xx[0]]
    for p in range(1,len(xx)):
        if yy[p-1]<y1 and yy[p]==y1:
            barstarts += [xx[p]]
        if y0<yy[p-1] and yy[p]==y0 and not 1==p:
            barends += [xx[p]]
    if y1==yy[-1]: barends += [xx[-1]]
    return barstarts,barends
def signedbars2grating(yy,xx=None,Λ=None): # yy,xx = y grating sign (±1) as np arrays, x position of bar start
    xx = xx if xx is not None else np.arange(len(yy)+1)*Λ/2
    y0,y1 = min(yy),max(yy)
    assert len(xx)==len(yy)+1, 'xx defines positions between bars, yy defines poling of bar'
    assert all([y==y0 or y==y1 for y in yy]), 'expected poled or unpoled, no intermediate values'
    barstarts,barends = [],[]
    ys = [y0]+list(yy)+[y0] # xx[i] is straddled by ys[i],ys[i+1]
    barstarts = [x for i,x in enumerate(xx) if ys[i]<ys[i+1]]
    barends   = [x for i,x in enumerate(xx) if ys[i]>ys[i+1]]
    assert all(b0<b1 for b0,b1 in zip(barstarts,barends))           # check barend follows barstart
    assert all(b0<b1 for b0,b1 in zip(barends[:-1],barstarts[1:]))  # check barstart follows previous barend
    return barstarts,barends
def grating2wave(starts,ends,delta=0):
    x,y = grating2xy(starts,ends,delta)
    return Wave(y,x)
def gratingperiod2wave(starts,ends,delta=0):
    y = np.array(starts[1:]) - np.array(starts[:-1])
    return Wave(y,starts[1:])
def ftgrating(starts,ends,p0=None,dp=None,normalize=True,amplitude=True,res=2001):
    p0 = (np.array(starts[1:])/2-np.array(starts[:-1])/2+np.array(ends[1:])/2-np.array(ends[:-1])/2).mean() if p0 is None else p0
    dp = p0/10 if dp is None else dp
    x,y = grating2xy(starts,ends)
    w = Wave(y,x)
    wp = np.linspace(p0-dp/2,p0+dp/2,res)
    wf = Wave(abs(w.ft(1/wp)),wp)
    wf = wf if amplitude else wf**2
    return wf/wf[wf.pmax()] if normalize else wf
def grating(period,dc,padx,padcount=1,gapx=0,phasex=0,x0=0,apodize=None,omitpads=()): # don't use phasex for electrode, not tested yet
    # assert 0==phasex, 'phasex implementation not yet tested'
    #if dev: period = periodmag*period
    barcount = int(np.ceil(1.*padcount*padx/period))
    barstarts = [x0+phasex+i*period for i in range(barcount)] # startx of each bar
    assert 0<dc and dc<=1
    if gapx:
        def isingap(x): return x%padx + period*dc > padx-gapx # remove if bar end is past gap start
        barstarts = [x for x in barstarts if not isingap(x)]
    def padnum(x): return x//padx
    barstarts = [x for x in barstarts if padnum(x) not in omitpads]
    barends = [x+period*dc for x in barstarts] # endx of each bar
    if apodize in [0,1]: # backwards compatibility with Dec20A and earlier
        def tri(x): return 1-2*abs(x-0.5)
        def trap(x): return np.minimum(1,2*tri(x))
        barstarts,barends = dropsmallbars(*apodizebars(barstarts,barends,afunc=trap),tolerance=0)
        return barstarts,barends
    apodize = {0:None,1:'trapezoidal'}[apodize] if apodize in [0,1] else apodize # keep backwards compatible with Dec20A mask
    if apodize is not None:
        assert 0.5==dc, 'apodization assumes 50% duty cycle'
        assert apodize in 'triangle,trapezoidal,asintriangle,asingauss23'.split(','), apodize
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
def shrinkbars(barstarts,barends,dx):
    return expandbars(barstarts,barends,-dx)
def expandbars(barstarts,barends,op): # positive op → make bars bigger, negative op → make bars smaller
    if callable(op): # op is overpole function, op(barsize) = amount of overpole
        dxs = [op(b1-b0) for b0,b1 in zip(barstarts,barends)] 
        return [a-dx/2 for a,dx in zip(barstarts,dxs)],[b+dx/2 for b,dx in zip(barends,dxs)]
    else: # op = amount of overpole
        return [a-op/2 for a in barstarts],[b+op/2 for b in barends]  # can be negative bar length, so call dropsmallbars()
def invertbarsgaps(starts,ends):
    return ends[:-1],starts[1:]
def breakupgaps(barstarts,barends,maxgap,barsize):
    if not maxgap:
        return barstarts,barends
    assert barsize<maxgap
    gapstarts,gapends = barends[:-1],barstarts[1:]
    xx0,xx1 = [],[] # xx0,xx1 will be the new gapstarts,gapends
    for x0,x1 in zip(gapstarts,gapends):
        if maxgap < x1-x0:
            n = 1 + int((x1-x0+barsize)/(maxgap+barsize))
            gap = (x1-x0+barsize)/n - barsize
            xx0 += [x0+i*(gap+barsize) for i in range(n)]
            xx1 += [x0+i*(gap+barsize)+gap for i in range(n)]
            assert np.allclose(x1,x0+(n-1)*(gap+barsize)+gap)
        else:
            xx0,xx1 = xx0+[x0],xx1+[x1]
    return barstarts[:1]+xx1,xx0+barends[-1:]
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
def apodizedinterleavedgrating(period1,period2,padcount,gx,apodize='triangle',res=0.1):
    xx = np.arange(0,1.*padcount*gx,res)
    def polingamplitude(a,b): # return the preferred poling amplitude of the two choices
        return np.where(np.abs(a)>np.abs(b),a,b)
    a1 = np.sin(2*np.pi*xx/period1)
    a2 = np.sin(2*np.pi*xx/period2)
    assert 'triangle'==apodize
    yy = (polingamplitude(a1,a2) < np.abs(1-2*xx/(padcount*gx)))
    barstarts,barends = fixedresgrating2bars(xx,yy)
    return barstarts,barends
def interleavedgrating(period1,period2,padcount,gx,res=0.1):
    # res,minbarsize = 0.1,max(0.6,overpole)
    xx = np.arange(0,1.*padcount*gx,res)
    yy = np.sign( np.sin(2*np.pi*xx/period1) + np.sin(2*np.pi*xx/period2) )
    barstarts,barends = fixedresgrating2bars(xx,yy)
    # if minbarsize:
    #     bars = [(x,y) for x,y in zip(barstarts,barends) if y-x>minbarsize]
    #     barstarts,barends = zip(*bars)
    # return barstarts,[b-overpole for b in barends]
    return barstarts,barends
def halfperiodgapgrating(period,n,padcount,gx): # n = number of periods between half-period gaps
    m = int(padcount*gx/(period*0.5*(2*n+1)))
    barstarts = [0.5*(2*i+1) + j*0.5*(2*n+1) for j in range(m) for i in range(n)]
    barstarts = period*np.array(barstarts)
    barends = barstarts + 0.5*period
    return barstarts,barends
def phaseflipgrating(period,n,padcount,gx): # n = number of periods between half-period gaps
    m = 2*int(padcount*gx/(2*n*period))
    def sectionstarts(n,j):
        return [2*i for i in range(1,n)] if j%2 else [2*i+1 for i in range(n)]
    barstarts = [(x+j*2*n)*0.5*period for j in range(m) for x in sectionstarts(n,j)]
    barends =   [(x+j*2*n)*0.5*period for j in range(m) for x in sectionstarts(n,j+1)]
    return barstarts,barends
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
def spectralcomb(g0,g1,offfraction,L=None): # removes all bars in middle: ■■____■■
    L = L if L is not None else g1[-1]
    x0,x1 = (1-offfraction)*L/2,(1+offfraction)*L/2
    ps = [(a,b) for a,b in zip(g0,g1) if a<x0 or x1<b]
    ps = [(a,min(b,x0) if a<x0 else b) for a,b in ps]
    ps = [(max(a,x1) if x1<b else a,b) for a,b in ps]
    return list(zip(*ps))
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


if __name__ == '__main__':
    def frequencybandwidth(Δλ,λ): # returns df in GHz for dlambda,lambda in nm, or df in Hz for dlambda,lambda in m
        return Δλ*299792458/λ**2 # in GHz
    def simplephasegrating():
        from grating import grating,invertbarsgaps,ftgrating
        g0,g1 = grating(100,0.5,padx=20001,padcount=1,gapx=0,phasex=0,x0=0)
        n = len(g1)//2
        a0,a1 = g0[:n],g1[:n]
        b0,b1 = g0[n:],g1[n:]
        b0,b1 = invertbarsgaps(b0,b1)
        # b0,b1 = list(np.array(b0)+50),list(np.array(b1)+50)
        f = ftgrating(g0,g1,p0=100,dp=6,normalize=0,res=2001)**2
        ff = ftgrating(a0+b0,a1+b1,p0=100,dp=6,normalize=0,res=2001)**2
        norm = f.max()
        Wave.plots(f/norm,ff/norm)
    # def Λ2λft(w):
    #     from sellmeier import polingperiod
    #     λs = np.linspace(1300,1700,201)
    #     Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
    #     return Wave(w.y,[Wave(Λs,λs).xaty(Λ) for Λ in w.x])
    def phasegratingfreq():
        from grating import grating,phasegrating,reverse,ftgrating,spectralcomb
        from sellmeier import polingperiod
        # print(spectralcomb([0,2,4,6],[1,3,5,7],0.1,L=8))
        # Λ,L = 18.5,20000
        λ0,L = 1550,50000
        Λ = polingperiod(λ0,sell='mglnridgewg',Type='zzz')
        print('Λ',Λ)
        gg0 = reverse(L,*grating(Λ,0.5,padx=L))
        gs = [phasegrating(*gg0,n) for n in [1,2,6]]
        gs += [spectralcomb(*phasegrating(*gg0,2),2/3)]
        fs = [ftgrating(*g,p0=Λ,dp=0.08,normalize=0,res=1001)**2 for g in gs]
        fs = [f/fs[0].max() for f in fs]
        # Wave.plots(*fs,seed=3)
        def Λ2λ(w):
            from sellmeier import polingperiod
            λs = np.linspace(1300,1700,201)
            Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
            return Wave(w.y,[Wave(Λs,λs).xaty(Λ) for Λ in w.x])
        def Λ2f(w):
            from sellmeier import polingperiod
            λs = np.linspace(λ0-200,λ0+200,401)
            c = 299792458 # nm/ns
            dfs = c/λs - c/λ0 # GHz
            Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
            return Wave(w.y,[Wave(Λs,dfs).xaty(Λ) for Λ in w.x])
        ns = ['1 phase section','2 phase sections','6 phase sections','"double slit"']
        # Wave.plots(*[Λ2λ(f) for f in fs],seed=3)
        Wave.plots(*[Λ2λ(f).rename(n) for f,n in zip(fs,ns)],seed=3,x='λ (nm)',grid=1,save=f'{L/1000:g}mm MgLN phase grating spectrum vs wavelength')
        Wave.plots(*[Λ2f(f).rename(n) for f,n in zip(fs,ns)],seed=3,x='Δf (GHz)',grid=1,save=f'{L/1000:g}mm MgLN phase grating spectrum vs frequency')
    def phasegratingvsL():
        from grating import grating,phasegrating,reverse,ftgrating,spectralcomb
        from sellmeier import polingperiod
        λ0,L = 1550,50000
        Λ = polingperiod(λ0,sell='mglnridgewg',Type='zzz')
        print('Λ',Λ)
        Ls = np.linspace(5,100,100//5)
        gs = [phasegrating(*reverse(L*1000,*grating(Λ,0.5,padx=L*1000)),2) for L in Ls]
        fs = [ftgrating(*g,p0=Λ,dp=0.08,normalize=0,res=1001)**2 for g in gs]
        fs = [f/fs[0].max() for f in fs]
        def Λ2f(w):
            from sellmeier import polingperiod
            λs = np.linspace(λ0-200,λ0+200,401)
            c = 299792458 # nm/ns
            dfs = c/λs - c/λ0 # GHz
            Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
            return Wave(w.y,[Wave(Λs,dfs).xaty(Λ) for Λ in w.x])
        fs = [Λ2f(f).rename(f'{L:g}mm') for f,L in zip(fs,Ls)]
        # Wave.plots(*fs,seed=3,x='Δf (GHz)',grid=1,save=f'MgLN phase grating spectrum vs frequency, L')
        def df(w):
            n = len(w)//2
            return abs( w[n:].xmax() - w[:n].xmax() )
        Wave([df(f) for f in fs],Ls).plot(m=1,x='L (mm)',y='Δf (GHz)',grid=1,save='dfvsL')
        Wave([df(f) for f in fs],Ls).plot(m=1,x='L (mm)',y='Δf (GHz)',grid=1,save='dfvsL, log',log=1)
    def phasegrating2(Ls=[25000,50000],λ0=1550):
        from grating import grating,phasegrating,reverse,ftgrating
        from sellmeier import polingperiod
        def Λ2λ(w):
            λs = np.linspace(1300,1700,201)
            Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
            return Wave(w.y,[Wave(Λs,λs).xaty(Λ) for Λ in w.x])
        def Λ2f(w,λ0):
            λs = np.linspace(λ0-200,λ0+200,401)
            c = 299792458 # nm/ns
            dfs = c/λs - c/λ0 # GHz
            Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
            return Wave(w.y,[Wave(Λs,dfs).xaty(Λ) for Λ in w.x])
        Λ = polingperiod(λ0,sell='mglnridgewg',Type='zzz')
        print('Λ',Λ)
        gg0 = reverse(Ls[0],*grating(Λ,0.5,padx=Ls[0]))
        gg1 = reverse(Ls[1],*grating(Λ,0.5,padx=Ls[1]))
        gg2 = phasegrating(*gg1,2)
        fs = [ftgrating(*g,p0=Λ,dp=0.08,normalize=0,res=1001)**2 for g in [gg0,gg1,gg2]]
        fs = [f/fs[0].max() for f in fs]
        # Wave.plots(*fs,seed=3)
        ns = [f'{Ls[0]/10000:g}cm',f'{Ls[1]/10000:g}cm',f'{Ls[1]/10000:g}cm with π phase shift']
        # Wave.plots(*[Λ2λ(f) for f in fs],seed=3)
        Wave.plots(*[Λ2λ(f).rename(n) for f,n in zip(fs,ns)],seed=3,x='λ (nm)',y='relative SHG efficiency',grid=1,save=f'{Ls[0]/1000:g}mm,{Ls[1]/1000:g}mm MgLN phase grating spectrum vs wavelength')
        Wave.plots(*[Λ2f(f,λ0).rename(n) for f,n in zip(fs,ns)],seed=3,x='Δf (GHz)',y='relative SHG efficiency',grid=1,save=f'{Ls[0]/1000:g}mm,{Ls[1]/1000:g}mm MgLN phase grating spectrum vs frequency')
    def phasegrating3(df=50,λ0=1550,L=50000):
        from grating import grating,phasegrating,reverse,ftgrating,alternatinggrating
        from sellmeier import polingperiod
        def Λ2λ(w):
            λs = np.linspace(1300,1700,201)
            Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
            return Wave(w.y,[Wave(Λs,λs).xaty(Λ) for Λ in w.x])
        def Λ2f(w,λ0):
            λs = np.linspace(λ0-200,λ0+200,401)
            c = 299792458 # nm/ns
            dfs = c/λs - c/λ0 # GHz
            Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
            return Wave(w.y,[Wave(Λs,dfs).xaty(Λ) for Λ in w.x])
        Δf = frequencybandwidth(1,λ0); print('Δf',Δf,'GHz')
        Δλ = df/Δf; print('Δλ',Δλ,'nm')
        Λ0 = polingperiod(λ0-Δλ/2,sell='mglnridgewg',Type='zzz')
        Λ1 = polingperiod(λ0+Δλ/2,sell='mglnridgewg',Type='zzz')
        Λ = Λ0/2+Λ1/2
        print('Λ0',Λ0,'Λ1',Λ1)
        gg0 = phasegrating( *reverse(L,*grating(Λ,0.5,padx=L)) ,2)
        reps = [1,2,50]
        gg1 = alternatinggrating(Λ0,Λ1,0.5,0.5,1,gx=L,repeats=reps[0])
        gg2 = alternatinggrating(Λ0,Λ1,0.5,0.5,1,gx=L,repeats=reps[1])
        gg3 = alternatinggrating(Λ0,Λ1,0.5,0.5,1,gx=L,repeats=reps[2])
        fs = [ftgrating(*g,p0=Λ,dp=0.2 if df>200 else 0.05,normalize=0,res=4001)**2 for g in [gg0,gg1,gg2,gg3]]
        fs = [f/fs[0].max() for f in fs]
        ns = [''] + [f'{L/r/2000:g}mm' for r in reps]
        Wave.plots(*[Λ2λ(f).rename(n) for f,n in zip(fs,ns)],seed=3,x='λ (nm)',y='relative SHG efficiency',legendtext='segment length',
            grid=1,save=f'MgLN alternating grating spectrum vs wavelength, df={df}GHz')
        Wave.plots(*[Λ2f(f,λ0).rename(n) for f,n in zip(fs,ns)],seed=3,x='Δf (GHz)',y='relative SHG efficiency',legendtext='segment length',
            grid=1,save=f'MgLN alternating grating spectrum vs frequency, df={df}GHz')
    def ag2test(reps=1):
        p0,p1 = 80,120
        g = alternatinggrating(p0,p1,0.5,0.5,1,2500,repeats=reps)
        gg = alternatinggrating(p0,p1,0.5,0.5,1,2500,repeats=reps)
        g2 = agwithphase(p0,p1,0,0,2500,reps)
        Wave.plots(grating2wave(*g),2+grating2wave(*gg),4+grating2wave(*g2),m=1)
    def agwithphase(p0,p1,f0,f1,L,repeats=1):
        from grating import grating,alternatinggrating2
        g0 = grating(p0,0.5,padx=L,phasex=f0*p0)
        g1 = grating(p1,0.5,padx=L,phasex=f1*p1)
        return alternatinggrating2(g0,g1,L,repeats)
    def phasegrating4(df=25,λ0=1550,L=50000,reps=1):
        from grating import grating,phasegrating,reverse,ftgrating,alternatinggrating,grating2wave,alternatinggrating2,gratingperiod2wave
        from sellmeier import polingperiod
        Δf = frequencybandwidth(1,λ0); print('Δf',Δf,'GHz')
        Δλ = df/Δf; print('Δλ',Δλ,'nm')
        Λ0 = polingperiod(λ0-Δλ/2,sell='mglnridgewg',Type='zzz')
        Λ1 = polingperiod(λ0+Δλ/2,sell='mglnridgewg',Type='zzz')
        Λ = Λ0/2+Λ1/2
        print('Λ0',Λ0,'Λ1',Λ1)
        gg0 = phasegrating( *reverse(L,*grating(Λ,0.5,padx=L)) ,2)
        # reps = [1,1,1] # reps = [50,50,50]
        # from random import random
        # phases = [random(),random(),random()]
        # gg1 = agwithphase(Λ0,Λ1,0,phases[0],L,reps[0])
        # gg2 = agwithphase(Λ0,Λ1,0,phases[1],L,reps[1])
        # gg3 = agwithphase(Λ0,Λ1,0,phases[2],L,reps[2])
        # gg1 = alternatinggrating(Λ0,Λ1,0.5,0.5,1,gx=L,repeats=reps[0])
        # gg2 = alternatinggrating(Λ0,Λ1,0.5,0.5,1,gx=L,repeats=reps[1])
        # gg3 = alternatinggrating(Λ0,Λ1,0.5,0.5,1,gx=L,repeats=reps[2])
        # fs = [ftgrating(*g,p0=Λ,dp=0.2 if df>200 else 0.05,normalize=0,res=4001)**2 for g in [gg0,gg1,gg2,gg3]]
        # ns = [''] + [f'{L/r/2000:g}mm' for r in reps]
        def phases(n): return [i/n for i in range(n)]
        ggs = [gg0] + [agwithphase(Λ0,Λ1,0,f,L,reps) for f in phases(10)]
        ns = ['reference'] + [f'{i}' for i,g in enumerate(ggs)]
        fs = [ftgrating(*g,p0=Λ,dp=0.2 if df>200 else 0.05,normalize=0,res=4001)**2 for g in ggs]
        fs = [-i/2+f/fs[0].max() for i,f in enumerate(fs)]
        def Λ2λ(w):
            λs = np.linspace(1300,1700,201)
            Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
            return Wave(w.y,[Wave(Λs,λs).xaty(Λ) for Λ in w.x])
        def Λ2f(w,λ0):
            λs = np.linspace(λ0-200,λ0+200,401)
            c = 299792458 # nm/ns
            dfs = c/λs - c/λ0 # GHz
            Λs = polingperiod(λs,sell='mglnridgewg',Type='zzz')
            return Wave(w.y,[Wave(Λs,dfs).xaty(Λ) for Λ in w.x])
        Wave.plots(*[Λ2λ(f).rename(n) for f,n in zip(fs,ns)],c='k012345678901234567890123456789',seed=3,x='λ (nm)',
            y='relative SHG efficiency', fontsize=8,
            grid=1,save=f'MgLN alternating grating 2, spectrum vs wavelength, df={df}GHz, segments={2*reps}')
        Wave.plots(*[Λ2f(f,λ0).rename(n) for f,n in zip(fs,ns)],c='k012345678901234567890123456789',seed=3,x='Δf (GHz)',
            y='relative SHG efficiency', fontsize=8,
            grid=1,save=f'MgLN alternating grating 2, spectrum vs frequency, df={df}GHz, segments={2*reps}')
    def phasegratingtest0():
        from grating import grating,invertbarsgaps,ftgrating,reverse,phasegrating
        Λ,L = 30,20000
        gg0 = reverse(L,*grating(Λ,0.5,padx=L))
        gg1 = phasegrating(*gg0,12)
        f0,f1 = [ftgrating(*gg,p0=Λ,dp=1,normalize=0,res=2001)**2 for gg in [gg0,gg1]]
        Wave.plots(f0/f0.max(),f1/f0.max())
    def phasegratingtest():
        from grating import grating,invertbarsgaps,ftgrating,reverse,phasegrating
        Λ,L = 30,20000
        gg0 = reverse(L,*grating(Λ,0.5,padx=L))
        gg1 = phasegrating(*gg0,12)
        f0,f1 = [ftgrating(*gg,p0=Λ,dp=1,normalize=0,res=2001)**2 for gg in [gg0,gg1]]
        Wave.plots(f0/f0.max(),f1/f0.max())
    def consecutivegrating(Λ1,Λ2,L1=2500,L2=2500,ΔL=0,apodize=None): # consecutive poling, grating lengths in mm
        from grating import grating,invertbarsgaps,ftgrating
        print(f" {Λ1:g}Λ1 {Λ2:g}Λ2")
        a0,a1 = grating(Λ1,0.5,padx=L1,padcount=1,gapx=0,phasex=0,x0=-L1/2-ΔL/2,apodize=apodize)
        b0,b1 = grating(Λ2,0.5,padx=L2,padcount=1,gapx=0,phasex=0,x0=-L2/2+ΔL/2,apodize=apodize)
        fa = ftgrating(a0,a1,p0=Λ1,dp=3*abs(Λ1-Λ2),normalize=0,res=1001)**2
        fb = ftgrating(b0,b1,p0=Λ2,dp=3*abs(Λ1-Λ2),normalize=0,res=1001)**2
        ff = ftgrating(a0+b0,a1+b1,p0=0.5*(Λ1+Λ2),dp=6*abs(Λ1-Λ2),normalize=0,res=1001)**2
        # ffb = ftgrating(a0+b0,a1+b1,p0=Λ2,dp=4*abs(Λ1-Λ2),normalize=0,res=1001)**2
        norm = max(fa.max(),fb.max())
        Wave.plots(fa/norm,fb/norm,ff/norm,c='012',l='3300',scale=(2,1),x='Λ (µm)',y='relative QPM response',legendtext=f"{ΔL/1000:g}mm separation",
            save=f"consecutivegrating {L1}L1 {L2}L2 {ΔL}ΔL" + (" apodized" if apodize else ""))
    def phasegratingtests():
        import sellmeier
        phasegratingfreq()
        phasegratingvsL()
        phasegrating2([20000,40000])
        phasegrating2([25000,50000])
        phasegrating2([20000,50000])
        phasegrating3(500)
        phasegrating3(50)
        phasegrating3(25)
        ag2test(3)
        phasegrating4(500)
        phasegrating4(200)
        phasegrating4(50)
        phasegrating4(25)
        phasegrating4(12.5)
        phasegrating4(50,reps=50)
        phasegrating4(25,reps=50)
        phasegrating4(12.5,reps=50)
    def consecutivegratings():
        # consecutivegrating(Λ1=97,Λ2=113,L1=5000,L2=5000,ΔL=0)
        consecutivegrating(Λ1=2500/29.5,Λ2=2500/27.5,L1=2500,L2=2500,ΔL=0)
        consecutivegrating(Λ1=2500/29.5,Λ2=2500/27.5,L1=2500,L2=2500,ΔL=2500)
        consecutivegrating(Λ1=2500/29.5,Λ2=2500/27.5,L1=2500,L2=2500,ΔL=5000)
        consecutivegrating(Λ1=2500/29.5,Λ2=2500/27.5,L1=2500,L2=2500,ΔL=7500)
        consecutivegrating(Λ1=2500/29.5,Λ2=2500/27.5,L1=2500,L2=2500,ΔL=10000)
        # consecutivegrating(Λ1=2500/29.5,Λ2=2500/27.5,L1=5000,L2=5000,ΔL=0)
        # consecutivegrating(Λ1=2500/29.5,Λ2=2500/27.5,L1=2500,L2=2500,ΔL=5000,apodize='triangle')
    def chirptest():
        f0 = ftgrating(*kchirpgrating(p0=14.45,p1=14.55,dc=0.5,padx=51000,padcount=1),p0=14.5,dp=0.2,normalize=1,res=1001)**2
        f0.plot(x='Λ (µm)',fewerticks=1)
    def apodizebandwidth(Λ=100,L=10000):
        apodizes = [None,*'trapezoidal,triangle,asingauss23,asintriangle'.split(',')]
        names = 'unapodized,trapezoidal,triangle,asin-gauss,asin-triangle'.split(',')
        gs = [grating(Λ,dc=0.5,padx=L,padcount=1,gapx=0,phasex=0,x0=0,apodize=apodize) for apodize in apodizes]
        fs = [ftgrating(*g,p0=Λ,dp=10,normalize=0,amplitude=0,res=2001).rename(s) for s,g in zip(names,gs)]
        fwhms = [f.fwhm() for f in fs]
        names = [s+f" {fwhm:.2f}µm FWHM" for s,fwhm in zip(names,fwhms)]
        fs = [f.rename(s) for s,f in zip(names,fs)]
        Wave.plots(*[f/fs[0].max() for f in fs],c='k0123',l='30000',log=1,xlim='f',ylim=(1e-4,1),grid=1,x='Λ (µm)',y='relative intensity',fontsize=8,abbrev=1,
            save=f"apodized grating bandwidth, L={L/1000:g}mm")
    # phasegratingtests() # ~1hr run time
    # simplephasegrating()
    # phasegratingtest0()
    # phasegratingtest()
    consecutivegratings()
    # chirptest()
    # apodizebandwidth(106.383)
