import numpy as np
from numpy import pi,sqrt,sin,cos,ceil,abs
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
def grating2xy(starts,ends,delta=0): # convert to plotable format, also used for piecewise linear ft
    # assume starts[n]<ends[n] and ends[n]<starts[n+1]                      # starts,ends = [20,40,60],[25,45,65]
    x = [q+r for p in zip(starts,ends) for q in p for r in [-delta,delta]]  # [20,20,25,25,40,40,45,45,60,60,65,65] or if Δ=1 [19,21,24,26,39,41,44,46,59,61,64,66]
    y = [q for p in zip(starts,ends) for q in [0,1,1,0]]                    # [ 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
    return x,y
def gratingfromxys(xys):
    barstarts = [0.5*(x0+x1) for (x0,y0),(x1,y1) in zip(xys[:-1],xys[1:]) if y0<y1]
    barends   = [0.5*(x0+x1) for (x0,y0),(x1,y1) in zip(xys[:-1],xys[1:]) if y0>y1]
    return barstarts,barends
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
    # w = Wave(y,x)
    w = 2*Wave(y,x)-1
    wp = np.linspace(p0-dp/2,p0+dp/2,res)
    wf = Wave(abs(w.ft(1/wp)),wp)
    # wf = wf if amplitude else wf**2
    wf = wf if amplitude else abs(wf)**2
    return wf/wf[wf.pmax()] if normalize else wf
def grating(period,dc,padx,padcount=1,gapx=0,phasex=0,x0=0,apodize=None,omitpads=()):
    barcount = int(np.ceil(1.*padcount*padx/period))
    barstarts = [x0+phasex+i*period for i in range(barcount)] # startx of each bar
    assert 0<dc and dc<=1
    if gapx:
        def isingap(x): return x%padx + period*dc > padx-gapx # remove if bar end is past gap start
        barstarts = [x for x in barstarts if not isingap(x)]
    def padnum(x): return x//padx
    barstarts = [x for x in barstarts if padnum(x) not in omitpads]
    barends = [x+period*dc for x in barstarts] # endx of each bar
    apodize = {0:None,1:'trapezoidal'}[apodize] if apodize in [0,1] else apodize # keep backwards compatible with Dec20A mask
    if apodize is not None:
        assert 0.5==dc, 'apodization assumes 50% duty cycle'
        assert apodize in 'triangle,trapezoidal,asintriangle,asingauss23'.split(','), apodize
        barstarts,barends = dropsmallbars(*apodizebars(barstarts,barends,apodize=apodize),tolerance=0)
    return barstarts,barends
def apodizebars(starts,ends,afunc=None,apodize=None,σ=0.23): # typical: afunc(0)=2, afunc(0.5)=1, afunc(1)=2 (where 0,1,2 = 0%,50%,100% duty cycle)
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
            'asingauss':  lambda x: 2 * asingauss(x,σ),
        }[apodize]
    a,b = np.array(starts),np.array(ends)
    c = a/2+b/2                 # c = centers
    w0 = b-a                    # old bar widths
    fp = (c-c[0])/(c[-1]-c[0])  # fp = fractionalposition
    w1 = w0*afunc(fp)           # new bar widths
    return c-w1/2,c+w1/2        # new starts,ends
def mergetouchingbars(barstarts,barends,tolerance=1.0): # <1.0um won't resolve lithographically and maintain a resistive gap
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
def breakupbars(barstarts,barends,maxbar,gapsize):
    if not maxbar:
        return barstarts,barends
    assert gapsize<=maxbar
    xx0,xx1 = [],[] # xx0,xx1 will be the new barstarts,barends
    for x0,x1 in zip(barstarts,barends):
        if maxbar < x1-x0:
            n = 1 + int((x1-x0+gapsize)/(maxbar+gapsize))
            gap = (x1-x0+gapsize)/n - gapsize
            xx0 += [x0+i*(gap+gapsize) for i in range(n)]
            xx1 += [x0+i*(gap+gapsize)+gap for i in range(n)]
            assert np.allclose(x1,x0+(n-1)*(gap+gapsize)+gap)
        else:
            xx0,xx1 = xx0+[x0],xx1+[x1]
    return xx0,xx1
def breakupgaps(barstarts,barends,maxgap,barsize):
    gapstarts,gapends = barends[:-1],barstarts[1:]
    xx0,xx1 = breakupbars(gapstarts,gapends,maxbar=maxgap,gapsize=barsize)
    return barstarts[:1]+xx1,xx0+barends[-1:]
# def breakupgaps(barstarts,barends,maxgap,barsize):
#     if not maxgap:
#         return barstarts,barends
#     assert barsize<maxgap
#     gapstarts,gapends = barends[:-1],barstarts[1:]
#     xx0,xx1 = [],[] # xx0,xx1 will be the new gapstarts,gapends
#     for x0,x1 in zip(gapstarts,gapends):
#         if maxgap < x1-x0:
#             n = 1 + int((x1-x0+barsize)/(maxgap+barsize))
#             gap = (x1-x0+barsize)/n - barsize
#             xx0 += [x0+i*(gap+barsize) for i in range(n)]
#             xx1 += [x0+i*(gap+barsize)+gap for i in range(n)]
#             assert np.allclose(x1,x0+(n-1)*(gap+barsize)+gap)
#         else:
#             xx0,xx1 = xx0+[x0],xx1+[x1]
#     return barstarts[:1]+xx1,xx0+barends[-1:]
def kchirpgrating(p0,p1,dc,padx,padcount=1,gapx=0,phase=0,x0=0): # https://www.gaussianwaves.com/2014/07/chirp-signal-frequency-sweeping-fft-and-power-spectral-density/
    assert 0==phase, 'phase not implemented'
    k0,k1 = 2*pi/p0,2*pi/p1
    barcount = int(padcount*padx*(k1+k0)/4/pi)
    L = 4*pi*barcount/(k1+k0)
    def k(x): return 0.5*(k1-k0)*x/L + k0 # note that ϕ(x) = ∫xk(x)dx so the instantaneous k at x=L is ϕ'(L) = k₁ not ½k₁ as naively expected
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
    def splitnumbers(s: str):
        # e.g. 'abc2.5xxx5e8$' → ['abc', 2.5, 'xxx', 5., 'e', 8., '$']
        import re
        pattern = r'\d+(?:\.\d+)?'                      # integer or decimal
        tokens = re.split(f'({pattern})', s)            # keep the numbers in the list
        return [float(t) if re.fullmatch(pattern, t) else t for t in tokens if t]
    L,Δ = padcount*gx,1
    if apodize[-1] in '0123456789':
        apodize,z = splitnumbers(apodize) # z = length in mm over which it happens, full length by default
        Δ = 1000*z/L
        assert 0<Δ<1, Δ
    assert apodize in ['tri','gauss23','lin','revlin']
    def amptri(x,Δ=0.2): return np.clip( 1-abs(2*(x-0.5)/Δ), 0, 1) # 0 to 1 to 0, Δ = non-zero fraction
    def ampgauss23(x,Δ=0.2,σ=0.23): return np.exp( -0.5 * (x-0.5)**2 / Δ**2  / σ**2 ) # 0 to 1 to 0
    def amplin(x,Δ=0.2): return 0.5 + np.clip((x-0.5)/Δ, -0.5, 0.5) # 0 at start, 1 at end, Δ = linear fraction
    def amprevlin(x,Δ=0.2): return 0.5 + np.clip((0.5-x)/Δ, -0.5, 0.5)
    def f1(x):
        f = {'tri':amptri,'gauss23':ampgauss23,'lin':amprevlin,'revlin':amplin}[apodize]
        return f(x,Δ)
    def f2(x):
        f = {'tri':amptri,'gauss23':ampgauss23,'lin':amplin,'revlin':amprevlin}[apodize]
        return f(x,Δ)
    xx = np.arange(0,L,res)
    a1 = -np.sin(2*np.pi*xx/period1)
    a2 = -np.sin(2*np.pi*xx/period2)
    t1 = .6* 2/pi * Wave(np.cumsum(2/pi * f1(xx/L)),xx,'target')
    t2 = .6* 2/pi * Wave(np.cumsum(2/pi * f2(xx/L)),xx)
    i,yy,area1,area2 = 0,0*xx,0*xx,0*xx
    while i<len(xx)-1:
        i += 1
        # toolow = (area1[i-1] + abs(a1[i]) < t1[i] - 1)
        # yy[i] = (+1 if (toolow and 0<a1[i]) else -1)
        # area1[i] = area1[i-1] + a1[i]*yy[i]
        pole1 = ((area1[i-1] + abs(a1[i]) < t1[i] - 1) and 0<a1[i])
        pole2 = ((area2[i-1] + abs(a2[i]) < t2[i] - 1) and 0<a2[i])
        pole = (pole1 or pole2)
        pole = pole1 if abs(a1[i])>abs(a2[i]) else pole2
        # pole = pole1 if f1(xx[i]/L) * abs(np.arcsin(a1[i])) > f2(xx[i]/L) * abs(np.arcsin(a2[i])) else pole2
        polingsign = +1 if pole else -1
        yy[i] = 0.5*(polingsign+1)
        area1[i] = area1[i-1] + a1[i]*polingsign
        area2[i] = area2[i-1] + a2[i]*polingsign
    barstarts,barends = fixedresgrating2bars(xx,yy)
    # u = Wave(yy,xx,'yy')
    # dc = u.smooth(1000,savgol=0).rename('mean dutycycle')
    # w1 = Wave(np.cumsum(a1*(2*yy-1))*res/L*pi/2,xx,'∫A$_1$').sp(l='0',lw=2)
    # w2 = Wave(np.cumsum(a2*(2*yy-1))*res/L*pi/2,xx,'∫A$_2$').sp(l='0',lw=2)
    # # wa = Wave(area1*res/L*pi/2,xx,'area1').sp(l='3',lw=1)
    # wt1 = (t1*res/L*pi/2).sp(c='k',l='2')
    # wt2 = (t2*res/L*pi/2).sp(c='k',l='2')
    # # Wave.plot(dc,w1,w2,wt1,wt2,grid=1,seed=1)
    # v1 = w1.smooth(10000).diff().rename('A$_1$')*L/res
    # v2 = w2.smooth(10000).diff().rename('A$_2$')*L/res
    # Wave.plot(dc,v1,v2,w1,w2,wt1,wt2,grid=1,seed=1,lw=1,x='x (µm)',y='relative amplitude')
    # def amplitude(w):
    #     from wavedata import wrange
    #     ux = wrange(0,L,0.1)
    #     wu = Wave([w(x,extrapolate='constant') for x in ux],ux)
    #     return [( pi/2 * (1-2*wu) * np.sin(2*pi*ux/Λ) ).smooth(10000) for Λ in (period1,period2)]
    # w = grating2wave(barstarts,barends)
    # Wave.plot(*amplitude(w),seed=1,lw=1)
    # exit()
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
def onoffs2grating(d,pattern,offvalue=0,δ=0,L=None,wave=False):
    pattern = [(1 if p in '1*#^x+' else offvalue) for p in pattern] if isinstance(pattern, str) else list(pattern)
    L = L if L is not None else d*len(pattern)
    N = int(L//(d*len(pattern)))
    ps = [0] + N*pattern + [0]
    xys = [(n*d+f*δ,p) for n,(p0,p1) in enumerate(zip(ps[:-1],ps[1:])) for p,f in ((p0,-1),(p1,+1)) if p0!=p1]
    a,b = bool(0<xys[0][0]), bool(xys[-1][0]<L)
    xys = a*[(0,offvalue)] + xys + b*[(L,offvalue)]
    return Wave.fromxys(xys) if wave else gratingfromxys(xys)
def deltasigmamatch(target):
    p = np.arange(len(target)) % 2 # parity mask [0 1 0 1 0 ...]
    return np.round((target - p) / 2) * 2 + p # snap to the nearest integer with the required parity
def fixedbarapodized(Λ,L,σ,wave=False): # Λ in µm, L in mm
    from scipy.special import erf
    Λ = abs(Λ)
    def g(n): # half gauss, n = halfbar index
        return np.exp(-0.5*n**2/σn**2)
    def intg(n): # integral of half gauss
        return σn * np.sqrt(np.pi/2) * erf( n / (np.sqrt(2)*σn) )
    N,σn = int(1e3*L/Λ-1),σ*2*1e3*L/Λ # N = number of halfbars in last half of grating, not counting the middle halfbar
    ns = [0]+[0.5+i for i in range(N)]
    t = g(Wave(ns,ns)).trapezoid().rename('t')
    h = Wave([intg(n) for n in ns],ns,'h')
    mh = Wave(deltasigmamatch(h.y[1:]-0.5)+0.5, h.x[1:])
    mt = Wave(deltasigmamatch(h.y[1:]-0.5)+0.5, h.x[1:])
    # Wave.plot(t,h,mt,mh,l='2323',grid=1,m='o',ms=3,lw=1.5,seed=1)
    d = mt.diff() # ±1 for each halfbar, +1=in-phase, -1=out-of-phase
    assert all(di in (-1,+1) for di in d.y)
    flip = [int(f) for f in ~(d.y==+1)^(np.arange(len(d))%2)] # e.g. flip=[0 1 0 1...] if d is all +1 or [1 0 1 0...] if d is all -1
    ff = flip[::-1] + [1] + flip # flips is for second half, ff is for full length including middle halfbar
    ff = [ff[0]^fi for fi in ff] # invert if needed to ensure beginning and end are unpoled
    assert ff==ff[::-1], 'flipping pattern must be symmetric'
    return onoffs2grating(Λ/2,ff,wave=wave)
    
if __name__ == '__main__':
    def simplephasegrating():
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
    simplephasegrating()
