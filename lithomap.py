#!/usr/bin/env python
#  -*- coding: utf-8 -*-
# from __future__ import print_function,division,unicode_literals
import numpy as np
from numpy import pi,sqrt,sin,cos,tan,arcsin,arccos,arctan,arctan2,log,exp,floor,ceil
import os,pickle
from decimal import Decimal
from geometry import chipidtext,rotatecurve,curvesboundingbox,rectsboundingbox,cropcurves,plotcurves,plotcurvelist,issimplepolygon
from geometry import upsamplecurve,eliminateholes,textcurves,scaletext,float2string,sbend,polyarea,reorient,cpwelectrode,cpwelectrode2
from geometry import checkerboardmetric,splittapersbend,curveboundingbox,addpointtocurve,signedpolyarea,combinepolys
from geometry import grating,averageperiod,oldfindpadboundarys,findpadboundarys,makepadtext,turncurvesupsidedown
from geometry import DotDict,PPfloat,OrderedDictFromCSV,comparedxf,advrlogo,dottedsegment
from font import Font
from savedxf import savemask,savedxfwithlayers

dev = True          # development version in which elements can be modified for the purpose of creating mask maps and chip maps
periodmag = 50      # period magnification for dev
scalemag =  1      # chip aspect ratio for dev
savepng = False     # enabling png file creation is slow

class Note:
    def __init__(self,x,y,dy=25,draw=[],**kwargs):
        self.x,self.y,self.dy = x,y,dy
        self.note = {k:kwargs[k] for k in kwargs}
        self.draw = draw # list of segments where segment = [(x0,y0),(x1,y1)]
    def __eq__(self,other):
        if isinstance(other,Note):
            return tuple(self)==tuple(other)
        return False
    def __lt__(self, other):
        return tuple(self) < tuple(other)
    def __iter__(self):# defined so that tuple(self) can be called, so attributes can be sorted, so Notes can be compared (i.e n1==n2)
        for a in ['x','y','dy','draw']:
            yield getattr(self,a)
        yield sorted(tuple(self.note.items()))
    def text(self):
        return ' '.join([f'{v}' for k,v in self.note.items()])
    def scale(self,xscale=1,yscale=None):
        yscale = yscale if yscale is not None else xscale
        return Note(xscale*self.x,yscale*self.y,yscale*self.dy,draw=[[(xscale*x,yscale*y) for x,y in seg] for seg in self.draw],**self.note)
    def translate(self,x0=0,y0=0):
        return Note(x0+self.x,y0+self.y,self.dy,draw=[[(x0+x,y0+y) for x,y in seg] for seg in self.draw],**self.note)
    def rotate(self,angle,x0=0,y0=0):
        def rotatepoint(x,y):
            return x0 + (x-x0)*cos(angle) - (y-y0)*sin(angle), y0 + (x-x0)*sin(angle) + (y-y0)*cos(angle)
        return Note(*rotatepoint(self.x,self.y),self.dy,draw=[[rotatepoint(x,y) for x,y in seg] for seg in self.draw],**self.note)
    def __str__(self):
        return f"({self.x},{self.y}) {self.text()}"
    def __repr__(self):
        return repr(tuple(self))
class Element:
    # def __init__(self,name='elem',layer=None,parent=None,x=None,y=None,guidespacing=None):
    def __init__(self,name='elem',layer=None,parent=None,x=None,y=None):
        self.info,self.rects,self.polys,self.elems,self.notes = DotDict(),[],[],[],[]
        self.name,self.parent = name,None
        self.x,self.y = 0,0
        if parent:
            self.x,self.y = parent.x,parent.y
            self.layer = layer if layer is not None else parent.layer
            # parent.addelem(self,guidespacing=guidespacing)
            parent.addelem(self)
        else:
            self.info.maskname = name
            assert layer is not None, 'root element must have layer'
            self.layer = layer
        self.x,self.y = x if x is not None else self.x, y if y is not None else self.y
        if not 'MASK'==self.layer:
            self.info.masklayer = layer
    def __eq__(self,other):
        if isinstance(other,Element):
            # for a,b in zip(tuple(self),tuple(other)): print(a,b,a==b)
            if not tuple(self)==tuple(other):
                print(self.parents(),other.parents(),'not equal')
                print(tuple(self))
                print(tuple(other))
                return False
            return True
            # return tuple(self)==tuple(other)
        return False
    def __lt__(self, other):
        return tuple(self) < tuple(other)
    def __iter__(self): # defined so that tuple(self) can be called, so that Element attributes can be sorted, so that Elements can be compared (i.e e1==e2)
        # yield str(self.name)
        yield str(self.layer)
        yield str(self.info)
        for a in ['rects','polys','elems','notes']:
            yield sorted(tuple(getattr(self,a)))
    def __str__(self):
        s = str(self.name)+' '+str(self.info)+'\n'
        def indent(s): return ''.join(['\t'+line for line in s.splitlines(True)])
        if self.elems:
            s += ''.join([indent('►'+str(e)+'\n') for e in self.elems if not e.name.startswith('pad')])
            #s += ''.join([indent('>'+str(e)+'\n') for e in self.elems if not e.name.startswith('pad')]) # in Python 2 str is bytes not unicode
        return s.rstrip('\n')
    def __repr__(self):
        s = repr(self.name)+' '+repr(self.info)+'\n'
        def indent(s): return ''.join(['\t'+line for line in s.splitlines(True)])
        if self.elems:
            s += ''.join([indent('►'+repr(e)+'\n') for e in self.elems if not e.name.startswith('pad')])
        return s.rstrip('\n')
    def tree(self):
        return tuple(self)
    def haselem(self,elemname):
        return any(elemname in e.name for e in self.elems)
    def nextname(self,elemname,n=1):
        name = elemname if 1==n else elemname+str(n)
        return name if not self.haselem(name) else self.nextname(elemname,n+1)
    def parents(self):
        return ('' if self.parent is None else self.parent.parents()+'→') + self.name
    def lastelem(self,cls=None):
        return self.elem[-1] if cls is None else [e for e in self.elems if isinstance(e,cls)][-1]
    def elemcount(self,cls=None):
        return len(self.elems) if cls is None else len([e for e in self.elems if isinstance(e,cls)])
    def vertexcount(self,includerects=True):
        return sum(len(pp) for pp in self.subpolys()) + 4*len(self.subrects())*includerects
    def addelem(self,e,guidespacing=None):
        e.parent = self
        name,e.name = e.name,self.nextname(e.name) # change name to a unique name # print('e.name',e.name)
        self.elems += [e]
        # if name.endswith('chip') and not isinstance(e,Chip):
        #     print("warning, Element('chip') is deprecated")
        #     n = self.info.chipcount = e.info.chipnumber = getattr(self.info,'chipcount',0) + 1
        #     nx,ny = 0,n-1
        #     if hasattr(self.info,'rows'):
        #         nx,ny = (n-1)//self.info.rows,(n-1)%self.info.rows
        #         self.info.columns = max(getattr(self.info,'columns',0),nx+1)
        #     e.info.chipid = str(ny+1)+'ABCDEFGHIJ'[nx]
        # print(e,isinstance(e,Group),type(e))
        if name.endswith('group') and not isinstance(e,Group):
            assert 0, "discontinued, use Group(parent=e) instead of Element('group',parent=self)"
            # e.info.groupnumber = self.info.groupcount = getattr(self.info,'groupcount',0) + 1
            # if self.info.groupcount>1: self.y += self.info.groupspacing
            # if 2==e.info.groupnumber: # show group spacing note between g1.last and g2.1
            #     self.addnote(self.x+1000,self.y-self.info.groupspacing/2,separation=self.info.groupspacing)
            # e.x,e.y = self.x,self.y
        if name.endswith('guide') and not isinstance(e,Guide):
            assert 0, "discontinued, use Guide(parent=e) instead of Element('guide',parent=self)"
        #     self.info.guidecount = getattr(self.info,'guidecount',0) + 1
        #     guidespacing = guidespacing if guidespacing is not None else self.finddefaultinfo('guidespacing') # self.parent.info.guidespacing
        #     if self.info.guidecount>1:
        #         self.y += guidespacing
        #         self.parent.y = self.y
        #         # both group and chip are updated with y position of current waveguide
        #         # chip.y should probably be updated when adding wide guides (spltter,mz) but isn't yet
        #     if 2==self.info.guidecount and 1==self.info.groupnumber: # show guide spacing note between g1.1 and g1.2
        #         self.addnote(self.x+1000,self.y-guidespacing/2,separation=guidespacing)
        #     e.info.guidenumber = str(self.info.groupnumber) + '.' + str(self.info.guidecount)
        #     e.x,e.y = self.x,self.y
        # if hasattr(self,'width'):
        #     e.width = self.width  # for keeping track of current waveguide width
        return self
    def subrects(self,layer=None):
        rects = self.rects if (layer is None or layer==self.layer) else []
        subrects = [r for e in self.elems for r in e.subrects(layer=layer)]
        return rects + subrects
        # return self.rects + [r for e in self.elems for r in e.subrects()]
    def subpolys(self,layer=None):
        polys = self.polys if (layer is None or layer==self.layer) else []
        subpolys = [c for e in self.elems for c in e.subpolys(layer=layer)]
        return polys + subpolys
        # return self.polys + [c for e in self.elems for c in e.subpolys(layer=layer)]
    def subelems(self):
        return self.elems + [ee for e in self.elems for ee in e.subelems()]
    def elem2curves(self):
        def recttopoly(x,y,dx,dy):
            return [(x,y),(x+dx,y),(x+dx,y+dy),(x,y+dy),(x,y)]
        return self.subpolys() + [recttopoly(*r) for r in self.subrects()]
    def subnotes(self):
        return self.notes + [n for e in self.subelems() for n in e.notes]
    def subpolystospreadsheet(self):
        for p in self.subpolys():
            for a,b in p:
                print(str(a)+','+str(b))
            print('NaN,NaN')
    def summary(self):
        print('rects:',len(self.rects),'polys:',len(self.polys),'elems:',len(self.elems))
        print('subrects:',len(self.subrects()),'subpolys:',len(self.subpolys()),'subelems:',len(self.subelems()))
    def findsubelem(self,name,parentlabel='>'): # only finds first one
        if self.name==name:
            # print(parentlabel+'>'+self.name)
            return self
        for e in self.elems:
            if e.findsubelem(name,parentlabel=parentlabel+'>'+self.name):
                return e
        return None
    def subelemlayers(self):
        s = [self.layer]
        for e in self.subelems():
            s = s+[e.layer] if e.layer not in s else s # maintains ordered set
        return s
    def addcircle(self,x0,y0,r,numsides=100):
        def p(theta): return (x0+r*cos(theta),y0+r*sin(theta))
        self.addpoly([p(theta) for theta in np.linspace(0,2*pi,numsides,endpoint=False)]+[p(0)]) # self.addrect(x0-r,y0-r,2*r,2*r)
        return self
    def addpolyrect(self,x0,y0,dx,dy):
        self.addpoly([(x0,y0),(x0+dx,y0),(x0+dx,y0+dy),(x0,y0+dy),(x0,y0)])
        return self
    def addtri(self,x0,y0,dx,dy):
        self.addpoly([(x0,y0),(x0+dx,y0),(x0,y0+dy),(x0,y0)])
        return self
    def addcenteredrect(self,x0,y0,dx,dy,swapxy=False):
        return self.addrect(y0-dy/2,x0-dx/2,dy,dx) if swapxy else self.addrect(x0-dx/2,y0-dy/2,dx,dy)
    def adddashedline(self,x0,y0,dx,dy,period,dash,xaxis=True,maxradius=None):
        dz = dx if xaxis else dy
        n = int(dz//period//2)
        zs = [dz*(i+0.5)/(2*n) for i in range(-n,n)]
        def isvalid(x0,y0,dx,dy):
            if maxradius is None:
                return True
            def dist(x,y):
                return np.sqrt(x**2+y**2)
            return dist(x0,y0)+dist(dx/2,dy/2)<maxradius
        def adddash(x0,y0,dx,dy):
            if isvalid(x0,y0,dx,dy):
                self.addcenteredrect(x0,y0,dx,dy)
        for z in zs:
            adddash(x0+z,y0,dash,dy) if xaxis else adddash(x0,y0+z,dx,dash)
        return self
    def addrect(self,x0,y0,dx,dy):
        assert dx!=0 and dy!=0, 'zero size rectangle:'+str([x0,y0,dx,dy,self.parents()])
        assert dx>0 and dy>0, 'negative size rectangle not allowed:'+str([x0,y0,dx,dy,self.parents()])
        self.rects += [(x0,y0,dx,dy)]
        # self.addpolyrect(x0,y0,dx,dy)
        return self
    def addpoly(self,ps,autoclose=False,unclosed=False):
        if autoclose and ps[-1]!=ps[0]: ps += [ps[0]]
        if not unclosed:
            assert list(ps[-1])==list(ps[0]), 'unclosed poly not specified:'+str([ps,self])
        self.polys += [ps]
        return self
    def addxypoly(self,xs,ys,autoclose=False,unclosed=False):
        self.addpoly([(x,y) for x,y in zip(list(xs),list(ys))],autoclose=autoclose,unclosed=unclosed)
        return self
    def addnote(self,x=None,y=None,dy=25,draw=[],**kwargs):
        e = Element(self.name+'-note',layer='NOTES',parent=self)
        e.notes += [Note(x if x is not None else self.x, y if y is not None else self.y, dy, draw=draw, **kwargs)]
        return self
    def addverticalnote(self,width,x=None,y=None,dy=25,period=5,dc=0.2,**kwargs):
        assert kwargs, "key required e.g. electrode=''"
        e = Element(self.name+'-note',layer='NOTES',parent=self)
        line = [(x,y-width/2),(x,y+width/2)]
        e.addnote(x,y,dy=dy,draw=dottedsegment(*line,period,dc),**kwargs)
        return self
    def addverticalnotes(self,name,polys,xx,dd,mag,ys=[],dy=25,period=5,dc=0.2,**kwargs): # xx,dd = measurement x, text Δx
        from geometry import lineslicecurves
        widths,centers = lineslicecurves(xx,polys,ys=ys)
        for n,(w,q) in enumerate(zip(widths,centers)):
            x,y = q+mag*np.array((dd+20*(1-n%2),0))
            self.addverticalnote(w,x,y,dy=dy,period=period,dc=dc,**{name:f"{w:.1f}"},**kwargs)
    def addhorizontalnote(self,width,x=None,y=None,dy=25,period=5,dc=0.2,**kwargs):
        assert kwargs, "key required e.g. electrode=''"
        e = Element(self.name+'-note',layer='NOTES',parent=self)
        line = [(x-width/2,y),(x+width/2,y)]
        e.addnote(x,y,dy=dy,draw=dottedsegment(*line,period,dc),**kwargs)
        return self
    def addhorizontalnotes(self,name,polys,yy,dd,ys=[],dy=25,period=5,dc=0.2,**kwargs): # yy,dd = measurement y, text Δy
        from geometry import lineslicecurves
        widths,centers = lineslicecurves(yy,polys,vertical=False,ys=ys,debug=0)
        for n,(w,q) in enumerate(zip(widths,centers)):
            x,y = np.array(q) + (0,0-20*(1-n%2))
            self.addhorizontalnote(w,x,y+dd,dy=dy,period=period,dc=dc,**{name:f"{w:.1f}"},**kwargs)

    def addpolys(self,cc,autoclose=False,unclosed=False):
        for c in cc:
            self.addpoly(c,autoclose,unclosed)
        return self
    def newsubelement(self,*args,**kwargs):
        return Element(*args,parent=self,**kwargs)
    def selfrectcount(self,layer):
        return len(self.rects) if (not layer or layer==self.layer) else 0
    def rectcount(self,layer=''):
        return self.selfrectcount(layer) + sum(e.selfrectcount(layer) for e in self.subelems())
    def selfrectarea(self,layer,overpole,maxarea=None):
        if maxarea is not None and (not layer or layer==self.layer):
            for x,y,dx,dy in self.rects:
                assert dx*dy<maxarea, f'rect exceeds max area: name:{self} layer:{layer} x{x:g} y{y:g} dx{dx:g} dy{dy:g} dx*dy{dx*dy:g} maxarea{maxarea:g}µm²'
        return sum((dx+overpole)*(dy+overpole) for x,y,dx,dy in self.rects) if (not layer or layer==self.layer) else 0
    def rectarea(self,layer='',overpole=0,maxarea=None):
        return self.selfrectarea(layer,overpole,maxarea=maxarea) + sum(e.selfrectarea(layer,overpole,maxarea=maxarea) for e in self.subelems())
    def selfarea(self,layer):
        return sum(dx*dy for x,y,dx,dy in self.rects) + sum(polyarea(c) for c in self.polys) if (not layer or layer==self.layer) else 0
    def area(self,layer=''):
        return self.selfarea(layer) + sum(e.selfarea(layer) for e in self.subelems())
    def selfpolyarea(self,layer):
        return sum(polyarea(c) for c in self.polys) if (not layer or layer==self.layer) else 0
    def polyarea(self,layer=''):
        return self.selfpolyarea(layer) + sum(e.selfpolyarea(layer) for e in self.subelems())
    def boundingbox(self): # returns (x0,y1,dx,dy)
        x0,y0,x1,y1 = (2**99,2**99,-2**99,-2**99)
        for e in self.subelems():
            for c in e.polys():
                for x,y in c:
                    x0,y0 = min(x0,x),min(y0,y)
                    x1,y1 = max(x1,x),max(y1,y)
            for x,y,dx,dy in e.rects():
                x0,y0 = min(x0,x),min(y0,y)
                x1,y1 = max(x1,x+dx),max(y1,y+dy)
                assert 0<dx and 0<dy
        return (x0,y0,x1-x0,y1-y0)
    def finddefaultinfo(self,a,fail=True): #search parents for one that has a value for info.a
        if hasattr(self.info,a):
            return getattr(self.info,a)
        else:
            if fail:
                assert self.parent, f'cannot find defaultinfo:{a} in parents:{self.parents()}'
            return self.parent.finddefaultinfo(a,fail=fail) if self.parent else None
    def findinfo(self,a): # search sublems for info
        def info(e):
            return getattr(e.info,a) if hasattr(e.info,a) else None
        return [info(e) for e in self.subelems() if info(e) is not None]
    def scale(self,xscale=1,yscale=None):
        yscale = yscale if yscale is not None else xscale
        def scaled(scale,x):
            # return Decimal(f'{scale:g}')*x if isinstance(x,Decimal) else scale*x
            return scale*float(x) if isinstance(x,Decimal) else scale*x # changes Decimal to float
        self.polys = [[(scaled(xscale,x),scaled(yscale,y)) for x,y in c] for c in self.polys]
        self.rects = [(scaled(xscale,x),scaled(yscale,y),scaled(xscale,dx),scaled(yscale,dy)) for x,y,dx,dy in self.rects]
        self.notes = [note.scale(xscale,yscale) for note in self.notes]
        self.elems = [e.scale(xscale,yscale) for e in self.elems]
        return self
    def translate(self,x0=0,y0=0):
        self.polys = [[(x0+x,y0+y) for x,y in c] for c in self.polys]
        self.rects = [(x0+x,y0+y,dx,dy) for x,y,dx,dy in self.rects]
        self.notes = [note.translate(x0,y0) for note in self.notes]
        self.elems = [e.translate(x0,y0) for e in self.elems]
        return self
    def translatex(self,x0):
        return self.translate(x0,0)
    def translatey(self,y0):
        return self.translate(0,y0)
    def showorientations(self):
        print(''.join(['+' if 0<=signedpolyarea(c) else '-' for c in self.subpolys()]))
        return self
    def orient(self,sign=+1):
        self.polys = [(c if 0<=sign*signedpolyarea(c) else c[::-1]) for c in self.polys]
        self.elems = [e.orient(sign) for e in self.elems]
        return self
    def convertrectstopolys(self): # convert all rects to polys in element and all sub elements
        def recttopoly(x,y,dx,dy):
            return [(x,y),(x+dx,y),(x+dx,y+dy),(x,y+dy),(x,y)]
        self.polys += [recttopoly(*r) for r in self.rects]
        self.rects = []
        self.elems = [e.convertrectstopolys() for e in self.elems]
        return self
    def rotate(self,angle,x0=0,y0=0):
        def recttopoly(x,y,dx,dy):
            return [(x,y),(x+dx,y),(x+dx,y+dy),(x,y+dy),(x,y)]
        def rotatepoint(x,y):
            return x0 + (x-x0)*cos(angle) - (y-y0)*sin(angle), y0 + (x-x0)*sin(angle) + (y-y0)*cos(angle)
        self.polys = [rotatecurve(c,angle,x0,y0) for c in self.polys] + [rotatecurve(recttopoly(*r),angle,x0,y0) for r in self.rects]
        self.rects = []
        self.notes = [note.rotate(angle,x0,y0) for note in self.notes]
        self.elems = [e.rotate(angle,x0,y0) for e in self.elems]
        return self
    def roundoff(self,res=0.0001,decimal=False,warn=True): # python Decimal class is used to print precisely rounded numbers to .dxf (e.g. 0.805 instead of 0.805000001)
        if decimal and warn:
            print(f'roundoff(res={res:g}): warning, all floats will be printed as Decimal')
        def rounded(x):
            return PPfloat(x,res) if decimal else res*round(x/res)
        self.polys = [[(rounded(x),rounded(y)) for x,y in c] for c in self.polys]
        self.rects = [(rounded(x),rounded(y),rounded(dx),rounded(dy)) for x,y,dx,dy in self.rects]
        self.elems = [e.roundoff(res,decimal=decimal,warn=0) for e in self.elems]
        return self
    def turnupsidedown(self,frame=None): # rectangle defined by frame will occupy exact same space as before but things inside will appear rotated 180° 
        if not frame: frame = self.boundingbox()
        x0,y0,dx0,dy0 = frame
        self.polys = turncurvesupsidedown(self.polys,frame)
        self.rects = turnrectsupsidedown(self.rects,frame)
        #self.polys = [[(2*x0+dx0-x,2*y0+dy0-y) for x,y in c] for c in self.polys]
        #self.rects = [(2*x0+dx0-x-dx,2*y0+dy0-y-dy,dx,dy) for x,y,dx,dy in self.rects]
        self.elems = [e.turnupsidedown(frame) for e in self.elems]
        return self
    def addoversizemetrics(self,width,xmetric,ymetric,i0=11,dx=10,nx=8,ny=6,w0=None,dw=0.5,textscale=1,relative=True,layer=None):
        # width = target waveguide width, (xmetric,ymetric) = position of center of the linear array of metrics
        # width on litho mask: width
        # widths after etch: range from w0-dw to w0+dw
        w0 = w0 if w0 is not None else width
        w0s = w0 if hasattr(w0,'__len__') else np.linspace(w0-dw,w0+dw,i0)
        for i,w in enumerate(w0s):
            xi = xmetric + (i-len(w0s)/2)*(dx+15*textscale)*nx
            m = Element('metric',parent=self,layer=layer)
            m.addoversizemetric(width=width,oversize=w,x=xi,y=ymetric,dx=dx,nx=nx,ny=ny)
            m.addtext(f"{w-width:+.1f}" if relative else f"{w:.1f}",xi+dx*nx+10,ymetric,scale=0.2*textscale)
    def addoversizemetric(self,width,oversize,x,y,dx=10,nx=8,ny=6):
        # if the target waveguide width on the mask is 3 but due to litho exposure it ends up 0.2 wider
        #  than intended (i.e. waveguide width on wafer is now 3.2), then the metric with value
        #  width = 3, oversize = 3.2 will now be the one that looks like a perfect checkerboard.
        dy,ds = width,2*width-oversize
        for i in range(nx):
            for j in range(ny):
                xi,yj = x+i*dx,y-j*dy-dy-0.5*(ds-dy)
                if 0==(i+j)%2:
                    self.addrect(xi,yj,dx,ds)
                if 0==i+j:
                    self.addnote(xi,yj+ds/2,hot=f"{ds:g}",dy=10)
                    self.addnote(xi-dx/2,yj-dy+ds/2,gap=f"{2*dy-ds:g}",dy=10)
        return self
    def addmetric(self,x,y,dx,dy,nx,ny):
        for i in range(nx):
            for j in range(ny):
                if 0==(i+j)%2:
                    self.addrect(x+i*dx,y-j*dy-dy,dx,dy)
        return self
    def addsubmountmetric(self,x,y,nx=8,ny=4):
        x0 = x
        for a in [2,1,0.7,0.6,0.5,0.4,0.3]:
        # for a in [2,1,0.9,0.8,0.7,0.6]:
        # for a in [2,1]:
            self.addmetric(x0,y,a,12,nx,ny) # 7s
            self.addtext('%.1f' % a, x0+a*nx/2.-6, y-12*ny-4, 6+6, 12,scaleifdev=False) # 13s
            x0 += a*nx+12
        #self.addtext('2.0 1.0 0.9 0.8 0.7 0.6', x, y-12*ny-4, x0-x, 12) # why so slow?
        return self
    def addtextrelative(self,s,x=0,y=0,**kwargs):
        self.addtext(s,self.x+x,self.y+y,**kwargs)
        return self
    def addcenteredtext(self,s,x=0,y=0,scale=1,vertical=False,upsidedown=False,scaleifdev=True,font=None,justify='center'):
        font = self.finddefaultinfo('font') if font is None else font
        self.info.text = getattr(self.info,'text','') + ' ' + s
        f = Font(font,size=128,screencoordinates=True)
        cc = f.textcurves(s,verticallayout=vertical)
        cc = scaletext(cc,x,y,fitx=0,fity=0,margin=0,scale=scale,center=True,dev=(dev and scaleifdev),scalemag=sqrt(scalemag))
        if upsidedown: cc = turncurvesupsidedown(cc)
        if justify not in 'l left'.split():
            x,y,dx,dy = curveboundingbox(cc)
            assert justify in 'c center centered r right'.split()
            p0 = (-dx,0) if justify in 'r right'.split() else (-dx/2,0)
            cc = [addpointtocurve(p0,c) for c in cc]
        e = Element('text',parent=self)
        e.addpolys(cc)
        return self
    def addtext(self,s,x=0,y=0,fitx=0,fity=0,margin=0,scale=1,center=True,vertical=False,upsidedown=False,skew=0,scaleifdev=True,font=None):
        if font is None:
            try:
                font = self.finddefaultinfo('font')
            except:
                font = ''
        if s.startswith('↕'): scale = 5
        if s.startswith('↓'): scale,vertical,upsidedown = 10,True,True
        self.info.text = getattr(self.info,'text','') + ' ' + s
        if font:
            if skew>0 and font: assert 0, 'skew not implmented for custom fonts'
            f = Font(font,size=128,screencoordinates=True)
            cc = f.textcurves(s,verticallayout=vertical)
        else:
            cc = textcurves(s,vertical,skew,screencoordinates=True)
        cc = scaletext(cc,x,y,fitx,fity,margin,scale,center,dev=(dev and scaleifdev),scalemag=sqrt(scalemag))
        if upsidedown: cc = turncurvesupsidedown(cc)
        e = Element('text',parent=self)
        e.addpolys(cc)
        return self
    def addinsettext(self,polylist,s,x0=0,y0=0,fitx=0,fity=0,margin=0,scale=1,fitcenter=True,vertical=False,scaleifdev=True,font=None,justify='left'):
        assert polylist[0][0], 'polylist must be a list of polygons, where each polygon is a list of (x,y) points'
        f = Font(font if font is not None else self.finddefaultinfo('font'),size=128,screencoordinates=True)
        cc = f.textcurves(s,verticallayout=vertical)
        cc = [upsamplecurve(c) for c in cc]
        cc = scaletext(cc,x0,y0,fitx,fity,margin,scale,fitcenter,dev=(dev and scaleifdev),scalemag=sqrt(scalemag))
        if justify not in 'l left'.split():
            _,_,dx,_ = curveboundingbox(cc)
            assert justify in 'c center centered r right'.split()
            p0 = (-dx,0) if justify in 'r right'.split() else (-dx/2,0)
            cc = [addpointtocurve(p0,c) for c in cc]
        self.addpolys(eliminateholes(polylist+cc))
        return self
    def insetelement(self,e):
        assert 'TEMP'==e.layer
        polys = e.convertrectstopolys().subpolys()
        notes = e.subnotes()
        self.insetshapes(polys)
        self.notes += notes
        return self
    def insetshapes(self,cc,sort=True): # doesn't check whether shape fits entirely inside
        assert not self.rects and not self.elems
        oldpolys = self.polys
        self.polys = []
        newpoly = combinepolys(cc,sort=sort)

        self.addpolys(eliminateholes(oldpolys+[newpoly]))
        return self

    def boundingbox(self): # curvesboundingbox
        pbb = curvesboundingbox(self.subpolys())
        rbb = rectsboundingbox(self.subrects())
        return rectsboundingbox([pbb,rbb])
    def polingarea(self,overpole,maxarea=None):
        # This will give you the poling area in microns squared so divide by 1e6 for area in mm². If you run it once for 0um overpole then again for 1um overpole, we can input both numbers into the masks.xlsx spreadsheet to generate the target poling charge for any amount of target overpoling. Also I generally do a rough sum up of the poling area manually as a double check, e.g. guide length x guide poled width x 0.5 x number of guides per chip x number of chips.
        # Running polingarea(overpole=1), where overpole=1 is 1um overpole, will print these values:
        #   polingarea, the area of all features on the mask with 0um overpoling
        #   rectpolingareawithoutoverpole, the area of all rectangles on the mask with 0um overpoling
        #   rectpolingareawithoverpole, the area of all rectangles on the mask with 1um overpoling (area of all rectangles with +1um height and +1 um width of each rectangle)
        # Generally polingarea will be equal to rectpolingareawithoutoverpole in which case you can ignore polingarea. However if there are large non-rectangle poling features on the mask, typically fiducials, then they will be non-equal. It is generally a small correction so I have not had to calculate overpoling area of the non-rectangles, so a good approximation in this case is:
        #   0um overpole area = polingarea
        #   1um overpole area = polingarea + rectpolingareawithoverpole - rectpolingareawithoutoverpole
        self.info.polingarea = self.area(layer='POLING')
        self.info.rectpolingareawithoutoverpole = self.rectarea(layer='POLING',maxarea=maxarea)
        self.info.overpole = overpole
        self.info.rectpolingareawithoverpole = self.rectarea(layer='POLING',maxarea=maxarea,overpole=self.info.overpole)
        print('self.info.polingarea',self.info.polingarea)
        print('self.info.rectpolingareawithoutoverpole',self.info.rectpolingareawithoutoverpole)
        print('self.info.overpole',self.info.overpole)
        print('self.info.rectpolingareawithoverpole',self.info.rectpolingareawithoverpole)
    def checkpolygonvalidity(self,verbose=True):
        if verbose: print('checking polygon validity...')
        def check(c):
            if not issimplepolygon(c):
                plotcurvelist([c],m=1,lw=0.2,ms=0.3,fewerticks=1)
                assert 0, 'invalid polygon found'
                return False
            return True
        bs = [check(c) for c in self.subpolys()]
        assert all(bs), 'invalid polygon found'
        if verbose: print('all polygons valid')
    def plot(self,screencoordinates=True,scale=1,notes=True,aspect=1):
        # plotcurves(cc,screencoordinates=True)
        import matplotlib
        import matplotlib.pyplot as plt
        plt.rcParams['font.size'] = 4
        plt.rcParams['keymap.quit'] = ['ctrl+w','cmd+w','q','escape']
        def recttopoly(x,y,dx,dy):
            return [(x,y),(x+dx,y),(x+dx,y+dy),(x,y+dy),(x,y)]
        def curvestocurve(pss):
            return [p for ps in pss for p in list(ps)+[(float('nan'),float('nan'))] ]
        pss = self.subpolys() + [recttopoly(*r) for r in self.subrects()]
        xs,ys = np.array(list(zip(*curvestocurve(pss))))
        plt.plot(xs,ys,'darkred',linewidth=1); plt.ylabel('y (µm)'); plt.xlabel('x (µm)')
        if screencoordinates: plt.gca().invert_yaxis()
        plt.gca().set_aspect(aspect)
        fig = plt.gcf()
        sx,sy = scale if hasattr(scale,'__len__') else (scale,scale)
        fig.set_size_inches((fig.get_size_inches()[0]*sx, fig.get_size_inches()[1]*sy))
        if notes:
            for note in self.subnotes():
                plt.text(note.x,note.y,note.text())
        plt.show()
        return self
    def savemask(self,filename,txt=False,layers=[],layernames=[],svg=True,png=True,gds=True,pdf=True,verbose=True,folder=None,pickle=True):
        return savemask(self,filename,txt=txt,layers=layers,layernames=layernames,svg=svg,png=png,gds=gds,pdf=pdf,verbose=verbose,folder=folder,dev=dev,scalemag=scalemag,pickle=pickle)
    def savedxfwithlayers(self,filename='',singlelayertosave='',svg=False,svgdebug=False,verbose=False,nomodify=True):
        return savedxfwithlayers(self,filename=filename,singlelayertosave=singlelayertosave,svg=svg,svgdebug=svgdebug,verbose=verbose,nomodify=nomodify,scalemag=scalemag)
    def savegds(self,filename,yinvert=True,layer=None):
        from phidl import Device
        D = Device('MASK')
        def recttopoly(x,y,dx,dy):
            return [(x,y),(x+dx,y),(x+dx,y+dy),(x,y+dy),(x,y)]
        pss = self.subpolys(layer=layer) + [recttopoly(*r) for r in self.subrects(layer=layer)]
        pss = [[(x,-y) for x,y in ps] for ps in pss] if yinvert else pss
        for ps in pss:
            D.add_polygon(ps)
        filename = f"{filename.replace('.gds','')}.gds"
        D.write_gds( filename, unit=1e-06, precision=1e-09, auto_rename=True, max_cellname_length=28, cellname='toplevel')
    def addchipdicepath(self,x0,y0,dx,dy=10,layer=None):
        # Element('facetdicepath',layer='ELECTRODE',parent=self).addpoly(zip(np.array([-1,+1,+1,-1,-1]),np.array([-1,-1,+1,+1,-1])))
        Element('facetdicepath',layer='ELECTRODE',parent=self).addrect(x0-dx/2,y0-dy/2,dx,dy)
    def addfacetdicepath(self,x0,y0,dy,dx=10,angle=5.5/180*pi,layer=None):
        cx,cy = np.array([x0-dx/2, x0+dx/2, x0+dy*np.tan(angle)+dx/2, x0+dy*np.tan(angle)-dx/2, x0-dx/2]), dy*np.array([0, 0,  1,  1, 0])
        Element('facetdicepath',layer=layer,parent=self).addpoly(list(zip(cx,cy)))
        return self
    def addelectrodedicing(self,dx,dy,xtop,xbot,angle,layer):
        self.addfacetdicepath(+dx/2+xtop,0,-dy/2,angle=5.5/180*pi,layer='ELECTRODE')
        self.addfacetdicepath(-dx/2+xtop,0,-dy/2,angle=5.5/180*pi,layer='ELECTRODE')
        self.addfacetdicepath(+dx/2+xbot,0,+dy/2,angle=5.5/180*pi,layer='ELECTRODE')
        self.addfacetdicepath(-dx/2+xbot,0,+dy/2,angle=5.5/180*pi,layer='ELECTRODE')
        self.addchipdicepath(x0=0,y0=-dy/2,dx=76000,layer='ELECTRODE')
        self.addchipdicepath(x0=0,y0=00000,dx=76000,layer='ELECTRODE')
        self.addchipdicepath(x0=0,y0=+dy/2,dx=76000,layer='ELECTRODE')
        return self
    def adddiceguides(self,x=None,y=None,chipx=None,chipy=None,s=25,strip=False,repx=1000,legacy=False):
        e = Element('dice',parent=self)
        x,y = getattr(self,'x',0) if x is None else x, getattr(self,'y',0) if y is None else y
        chipx = chipx if chipx is not None else self.finddefaultinfo('chiplength')
        chipy = chipy if chipy is not None else self.finddefaultinfo('chipwidth')
        def repeatx(r,periodx,endx,tri=False):
            x0,y0,dx,dy = r
            for xx in range(0,endx,periodx):
                # self.addrect(x0+xx+x,y0+y,dx,dy) # won't show up in chip map! why not?
                e.addtri(x0+xx+x,y0+y,dx,dy) if tri and not legacy else e.addpolyrect(x0+xx+x,y0+y,dx,dy)
        repeatx((0,0,s,s),repx,chipx)
        repeatx((s,0,s,s),repx,chipx,tri=1)
        repeatx((0,chipy,s,-s),repx,chipx)
        repeatx((s,chipy,s,-s),repx,chipx,tri=1)
        repeatx((s,s,-s,2*s if legacy else s),5*repx,chipx,tri=1)
        repeatx((0,chipy-s,s,-s),2*repx,chipx,tri=1)
        if strip:
            ee = Element('dicestrip',layer='STRIPPING',parent=self)
            sy = 150
            ee.addrect(0,-sy/2,chipx,sy).addrect(0,chipy-sy/2,chipx,sy)
        return self
    def adddicelines(self,x=None,y=None,chipx=None,chipy=None,s=80,repx=1000,layer=None):
        e = Element('dice',parent=self,layer=layer)
        x,y = getattr(self,'x',0) if x is None else x, getattr(self,'y',0) if y is None else y
        chipx = chipx if chipx is not None else self.finddefaultinfo('chiplength')
        chipy = chipy if chipy is not None else self.finddefaultinfo('chipwidth')
        x0 = x+0.5*chipx
        n = x0//repx
        for f in (-1,+1):
            for xx in repx*np.arange(0.5,n,4):
                e.addtri(x0+f*xx,y,-f*s,s).addtri(x0+f*xx,y,-f*s,-s)
                e.addtri(x0+f*xx,y+chipy,-f*s,+s).addtri(x0+f*xx,y+chipy,-f*s,-s)
            for xx in repx*np.arange(1.5,n,4):
                e.addpolyrect(x0+f*xx-f*repx/2,y,f*repx,s).addpolyrect(x0+f*xx-f*repx/2,y,f*repx,-s)
                e.addpolyrect(x0+f*xx-f*repx/2,y+chipy,f*repx,+s).addpolyrect(x0+f*xx-f*repx/2,y+chipy,f*repx,-s)
            for xx in repx*np.arange(2.5,n,4):
                e.addpolyrect(x0+f*xx-f*repx/2,y,f*repx,s).addpolyrect(x0+f*xx-f*repx/2,y,f*repx,-s)
                e.addpolyrect(x0+f*xx-f*repx/2,y+chipy,f*repx,+s).addpolyrect(x0+f*xx-f*repx/2,y+chipy,f*repx,-s)
            for xx in repx*np.arange(3.5,n,4):
                e.addtri(x0+f*xx,y,f*s,s).addtri(x0+f*xx,y,f*s,-s)
                e.addtri(x0+f*xx,y+chipy,f*s,+s).addtri(x0+f*xx,y+chipy,f*s,-s)
        return self
    def addcrossfiducial(self,w,x=0,y=0):
        s = w/2.
        self.addrect(-s+0+x,-s+0+y,w,w)
        self.addrect(-s-2*w+x,-s+0+y,2*w,w)
        self.addrect(-s+w+x,-s+0+y,2*w,w)
        self.addrect(-s+0+x,-s-2*w+y,w,2*w)
        self.addrect(-s+0+x,-s+w+y,w,2*w)
        return self
    def addvernierset(self,layer,x0,y0,rotation,partial=None,invert=False):
        tx,ty = 250,250
        for t,swap in [(ty,0),(-ty,1)]:
            Element('vernierfiducial',parent=self).addvernier(inner=0,swapxy=swap).translate(+tx,t).rotate(rotation).translate(x0,y0)
            Element('vernierfiducial',parent=self).addvernier(inner=1,swapxy=swap).translate(-tx,t).rotate(rotation).translate(x0,y0)
            Element('vernierfiducial',layer=layer,parent=self).addvernier(inner=1,swapxy=swap,partial=partial,invert=invert).translate(+tx,t).rotate(rotation).translate(x0,y0)
            Element('vernierfiducial',layer=layer,parent=self).addvernier(inner=0,swapxy=swap,partial=partial,invert=invert).translate(-tx,t).rotate(rotation).translate(x0,y0)
        return self
    def addvernier(self,inner=False,s0=8.0,s1=8.2,w0=30,w1=40,num=11,swapxy=0,partial=None,invert=False):
        # for partial, only show half of the fiducial (partial=1 for 1st half, partial=2 for 2nd half)
        def ys():
            if inner:
                return [(n+0.5)*s1 for n in range(-num,num)] if invert else [n*s1 for n in range(-num,num+1)]
            else:
                return [n*s0 for n in range(-num,num+1)] if invert else [(n+0.5)*s0 for n in range(-num,num)]
        def dxs():
            if inner:
                return [(w1 if n in [-1,0] else w0) for n in range(-num,num)] if invert else [(w0 if n%5 else w1) for n in range(-num,num+1)]
            else:
                return [(w0 if n%5 else w1) for n in range(-num,num+1)] if invert else [(w1 if n in [-1,0] else w0) for n in range(-num,num)]
        for y,dx in zip(ys(),dxs()):
            if partial is None or 1==partial:
                self.addcenteredrect(+dx/4 if inner else +w0*3/4,y,dx/2 if inner else dx,s0/2,swapxy)
            if partial is None or 2==partial:
                self.addcenteredrect(-dx/4 if inner else -w0*3/4,y,dx/2 if inner else dx,s0/2,swapxy)
        return self
    def addcrossrepeats(self,w,s,vnum=7,hnum=5,partial=None): # s = spacing
        evens,odds = (1,1) if partial is None else ((0,1) if 1==partial else (1,0))
        self.addcrossfiducial(w)
        for n in range(1,vnum):
            if (0==n%2 and evens) or (1==n%2 and odds):
                self.addcrossfiducial(w,0,-s*n).addcrossfiducial(w,0,+s*n)
        for n in range(1,hnum):
            if (0==n%2 and evens) or (1==n%2 and odds):
                self.addcrossfiducial(w,-s*n,0).addcrossfiducial(w,+s*n,0)
        return self
    def addplatingcontacts(self,layer='ELECTRODE'):
        waferdiameter,r = 76200,3500
        x,y = 2*[waferdiameter*sqrt(2)/4]
        e = Element('platingcontact',layer=layer,parent=self)
        e.addcircle(-x,-y,r).addcircle(+x,-y,r).addcircle(-x,+y,r).addcircle(+x,+y,r)
        return self
    def addplatingring(self,x0=0,y0=0,dr0=-3500,dr1=+500,waferdiameter=76200,numsides=100,layer='ELECTRODE'):
        r0,r1 = waferdiameter/2+dr0,waferdiameter/2+dr1
        e = Element('platingring',layer=layer,parent=self)
        def p(r,theta): return (x0+r*sin(theta),y0+r*cos(theta)) # start at (0,r+y0)
        p0s = [p(r0,theta) for theta in np.linspace(0,+2*pi,numsides,endpoint=False)]
        p1s = [p(r1,theta) for theta in np.linspace(0,-2*pi,numsides,endpoint=False)]
        ps = p0s + p0s[:1] + p1s + p1s[:1] + p0s[:1]
        # from waves import Vs; Vs(ps).wave().plot(m=1)
        e.addpoly(ps)
        return self
    def addfiducialwindow(self,x=0,y=0,layer='POLING',margin=10):
        e = Element('crossfiducial',layer=layer,parent=self)
        if layer=='STRIPPING':
            e.addcenteredrect(0,0,3800+margin,2000+margin)
        else:
            e.addcenteredrect(0,+800,3000,400).addcenteredrect(+1700,0,400,2000)
            e.addcenteredrect(0,-800,3000,400).addcenteredrect(-1700,0,400,2000)
        e.translate(x,y)
        return self
    def addcentralfiducialwindows(self,r,layers=('ELECTRODE','BUFFER'),layernums=(1,2)):
        for layer in layers+('STRIPPING',):
            self.addfiducialwindow(x=+r-10000,y=0,layer=layer)
            self.addfiducialwindow(x=-r+10000,y=0,layer=layer)
        for layer,num in zip(layers,layernums):
            self.addstanfordfiducialchip(x=+r-10000,y=0,masklayeropen=False,layer=layer,partial=num)
            self.addstanfordfiducialchip(x=-r+10000,y=0,masklayeropen=False,layer=layer,partial=num)
            self.addcrossfiducialchip(x=+r-10000,y=0,layer=layer,partial=num,invertvernier=True,vnum=5,hnum=4,diag=True)
            self.addcrossfiducialchip(x=-r+10000,y=0,layer=layer,partial=num,invertvernier=True,vnum=5,hnum=4,diag=True)
            # assert dev
            # self.addcrossfiducialchipold(x=1+r-10000,y=1,layer=layer,partial=num,invertvernier=True,vnum=5,hnum=4)
            # self.addcrossfiducialchipold(x=1-r+10000,y=1,layer=layer,partial=num,invertvernier=True,vnum=5,hnum=4)
    def addcrossfiducialchip(self,x=0,y=0,vnum=7,hnum=5,layer='POLING',onlymasklayer=False,rotation=0,partial=None,invertvernier=False,diag=False):
        smallw,bigw,spacing = 10,20,127
        s,p,r,v,h,c = spacing,partial,rotation,vnum,hnum,'crossfiducial'
        Element(c,parent=self).addcrossrepeats(  bigw,s,v,h,None).rotate(r).translate(x+1000,y)
        Element(c,parent=self).addcrossrepeats(smallw,s,v,h,None).rotate(r).translate(x-1000,y)
        if diag:
            for xi,yi in [(+s*3,+s),(-s*3,-s),(+s*3,-s),(-s*3,+s)]:
                Element(c,parent=self).addcrossfiducial(smallw).rotate(r).translate(x+1000+xi,y+yi)
                Element(c,parent=self).addcrossfiducial(  bigw).rotate(r).translate(x-1000+xi,y+yi)
        self.addvernierset(layer,x-1000,y,r,partial=p,invert=invertvernier)
        self.addvernierset(layer,x+1000,y,r,partial=p,invert=invertvernier)
        if not onlymasklayer:
            Element(c,layer,self).addcrossrepeats(smallw,s,v,h,p).rotate(r).translate(x+1000,y)
            Element(c,layer,self).addcrossrepeats(  bigw,s,v,h,p).rotate(r).translate(x-1000,y)
        if diag:
            assert layer and not onlymasklayer
            j,(w0,w1) = {1:+1,2:-1}[p],{1:(smallw,bigw),2:(bigw,smallw)}[p]
            for x0 in (x-1000,x+1000):
                Element(c,layer,self).addcrossfiducial(w0).rotate(r).translate(x0+s,y+s)
                Element(c,layer,self).addcrossfiducial(w0).rotate(r).translate(x0-s,y-s)
                Element(c,layer,self).addcrossfiducial(w1).rotate(r).translate(x0-s,y+s)
                Element(c,layer,self).addcrossfiducial(w1).rotate(r).translate(x0+s,y-s)
            for xi,yi in [(+s*3,+s*j),(-s*3,-s*j)]:
                Element(c,layer,self).addcrossfiducial(  bigw).rotate(r).translate(x+1000+xi,y+yi)
                Element(c,layer,self).addcrossfiducial(smallw).rotate(r).translate(x-1000+xi,y+yi)
        return self
    def addcrossfiducialchipold(self,x=0,y=0,vnum=7,hnum=5,layer='POLING',onlymasklayer=False,rotation=0,partial=None,invertvernier=False):
        smallsize,bigsize,spacing = 10,20,127
        Element('crossfiducial',parent=self).addcrossrepeats(bigsize,spacing,vnum=vnum,hnum=hnum,partial=None).rotate(rotation).translate(x+1000,y)
        Element('crossfiducial',parent=self).addcrossrepeats(smallsize,spacing,vnum=vnum,hnum=hnum,partial=None).rotate(rotation).translate(x-1000,y)
        self.addvernierset(layer,x-1000,y,rotation,partial=partial,invert=invertvernier).addvernierset(layer,x+1000,y,rotation,partial=partial,invert=invertvernier)
        if not onlymasklayer:
            Element('crossfiducial',layer=layer,parent=self).addcrossrepeats(smallsize,spacing,vnum=vnum,hnum=hnum,partial=partial).rotate(rotation).translate(x+1000,y)
            Element('crossfiducial',layer=layer,parent=self).addcrossrepeats(bigsize,spacing,vnum=vnum,hnum=hnum,partial=partial).rotate(rotation).translate(x-1000,y)
        return self
    def addstanfordfiducialchip(self,x=0,y=0,layer='POLING',masklayeropen=True,onlymasklayer=False,rotation=0,partial=None):
        smallsize,bigsize = 8,18
        if masklayeropen:
            Element('stanfordfiducial',parent=self).addstanfordfiducial(w=bigsize).rotate(rotation).translate(x,y)
            if not onlymasklayer:
                Element('stanfordfiducial',layer=layer,parent=self).addstanfordfiducial(w=smallsize,partial=partial).rotate(rotation).translate(x,y)
        else:
            Element('stanfordfiducial',parent=self).addstanfordfiducial(w=smallsize).rotate(rotation).translate(x,y)
            if not onlymasklayer:
                Element('stanfordfiducial',layer=layer,parent=self).addstanfordfiducial(w=bigsize,partial=partial).rotate(rotation).translate(x,y)
        return self
    def addstanfordfiducial(self,w=8,partial=None):
        # dx = length/2; dy = 1536/2 - 10848/2 + 10848*(0<vnum); wgwidth = 4*2 + 10*(0==cmpstr(filelabel,"Aug17"))
        length=246*2; length2=length-9*2*2
        # center box
        self.addrect(-w/2.,-w/2.,w,w)
        dx = 2*(-49-15); dy = +4*2+length/2. # -4*2-length/2
        y0,y1 = -length/2.+dy,length
        if partial is not None:
            y0,y1 = -length/2.+dy if 1==partial else dy,length/2
        if w<18:
            # stripes
            for i in range(12):
                self.addrect(-w/2.+dx+i*18+2*34*(5<i),y0,w,y1)
        else:
            # left and right boxes (merged stripes)
            w0 = 18
            self.addrect(-w0/2.+dx,y0,w0*6,y1).addrect(-w0/2.+dx+6*18+2*34,y0,w0*6,y1)
        # flux capacitor
        for j in [0,2,4]:
            dz = +4*2+length/2.
            c = list(zip([-1,  1,  1, -1, -1],[-1, -1,  1,  1, -1]))
            if partial is not None:
                c = list(zip([-1,  1,  1, -1, -1],[-1, -1,  0,  0, -1])) if 1==partial else list(zip([-1,  1,  1, -1, -1],[0, 0,  1,  1, 0]))
            c = [(x*w/2., dz+y*length2/2.) for x,y in c]
            c = [( x*cos(pi*j/3.) - y*sin(pi*j/3.), x*sin(pi*j/3.) + y*cos(pi*j/3.)) for x,y in c]
            self.addpoly(c)
        return self
    def addhwfiducials(self,x,y,layer='METAL'):
        self.addboxesfiducial(x,y)
        self.addboxesfiducial(x,y,layer=layer)
        Element('fiducial',layer=layer,parent=self).addhframe(700,200,100).translate(x,y)
        Element('fiducial',layer=layer,parent=self).addvframe(700,400,100).translate(x,y)
        return self
    def addboxesfiducial(self,x,y,layer=None):
        e = Element('fiducial',layer=layer,parent=self).addcenteredrect(-150,0,10,10).addcenteredrect(0,0,20,20).addcenteredrect(+150,0,30,30).translate(x,y)
        return self
    def addcornerfiducials(self,fidx,fidy,layers=['ELECTRODE','BUFFER']):
        maskrotation = self.finddefaultinfo('maskrotation')
        print(f"fidx:{fidx:+g} fidy:{fidy:+g} fidθ:{arctan2(fidy,fidx):+g} maskθ:{maskrotation:g}")
        assert np.isclose(abs(arctan2(fidy,fidx)),maskrotation,atol=0.001)
        for layer,num in zip(layers,[1,2]):
            self.addstanfordfiducialchip(x=-fidx,y=fidy,layer=layer,masklayeropen=False,rotation=-maskrotation,partial=num)
            self.addstanfordfiducialchip(x=+fidx,y=fidy,layer=layer,masklayeropen=False,rotation=-maskrotation,partial=num)
            self.addcrossfiducialchip(x=-fidx,y=fidy,layer=layer,vnum=5,rotation=-maskrotation,partial=num,invertvernier=True)
            self.addcrossfiducialchip(x=+fidx,y=fidy,layer=layer,vnum=5,rotation=-maskrotation,partial=num,invertvernier=True)
        for layer in layers+['STRIPPING']:
            self.addfiducialwindow(x=-fidx,y=fidy,layer=layer)
            self.addfiducialwindow(x=+fidx,y=fidy,layer=layer)
    def addhframe(self,dx,dy,dw,layer=None,left=True,right=True):
        e = Element('frame',layer=layer,parent=self)
        if left:  e.addrect(-dx/2,-dy/2,dw,dy)
        if right: e.addrect(dx/2-dw,-dy/2,dw,dy)
        return self
    def addvframe(self,dx,dy,dw,layer=None):
        e = Element('frame',layer=layer,parent=self).addrect(-dx/2,-dy/2,dx,dw).addrect(-dx/2,dy/2-dw,dx,dw)
        return self
    def addmodefilterspliced(self,width,splicex1,splicex2,dx=None,inwidth=0,outwidth=0,modefilterx=None,taperx=None,swapinandout=False):
        # same as addmodefilter() but with a gap missing in the channel part from splicex1 to splicex2
        if swapinandout: inwidth,outwidth = outwidth,inwidth
        e = Guide(parent=self)
        if inwidth: e.info.inputmodefilter = inwidth
        if outwidth: e.info.outputmodefilter = outwidth
        if not inwidth and not outwidth: modefilterx,taperx = 0,0
        if modefilterx: e.info.modefilterlength = modefilterx
        modefilterx = modefilterx if modefilterx is not None else e.finddefaultinfo('modefilterlength')
        if taperx: e.info.taperlength = taperx
        taperx = taperx if taperx is not None else e.finddefaultinfo('taperlength')
        dx = dx if dx is not None else self.finddefaultinfo('chiplength') # self.parent.parent.info.chipx-self.x # whereever the start, end is at guide end
        e.info.guidewidth = width
        e.addonmodefilter(width,splicex1,inwidth,0,modefilterx,taperx)
        e.x = splicex2
        e.addonmodefilter(width,dx-splicex2,0,outwidth,modefilterx,taperx)
        return self
    def addmodefilter(self,width,dx=None,inwidth=0,outwidth=0,modefilterx=None,taperx=None,swapinandout=False,x0=None,y0=None): # default will be placed at self.x,self.y
        if swapinandout: inwidth,outwidth = outwidth,inwidth
        e = Guide(parent=self,x=x0,y=y0)
        if inwidth: e.info.inputmodefilter = inwidth
        if outwidth: e.info.outputmodefilter = outwidth
        if not inwidth and not outwidth: modefilterx,taperx = 0,0
        if modefilterx: e.info.modefilterlength = modefilterx
        modefilterx = modefilterx if modefilterx is not None else e.finddefaultinfo('modefilterlength')
        if taperx: e.info.taperlength = taperx
        taperx = taperx if taperx is not None else e.finddefaultinfo('taperlength')
        dx = dx if dx is not None else self.finddefaultinfo('chiplength') # self.parent.parent.info.chipx-self.x # whereever the start, end is at guide end
        e.info.guidewidth = width
        e.addonmodefilter(width,dx,inwidth,outwidth,modefilterx,taperx)
        return self
    def addmodefilterpair(self,width,split,dx=None,inwidth=0,outwidth=0,modefilterx=None,taperx=None): # default will be placed at self.x,self.y
        for i in (0,1):
            e = Guide(parent=self,guidespacing=(None if 0==i else split),showspacing=(0<i))
            if inwidth: e.info.inputmodefilter = inwidth
            if outwidth: e.info.outputmodefilter = outwidth
            if not inwidth and not outwidth: modefilterx,taperx = 0,0
            if modefilterx: e.info.modefilterlength = modefilterx
            modefilterx = modefilterx if modefilterx is not None else e.finddefaultinfo('modefilterlength')
            if taperx: e.info.taperlength = taperx
            taperx = taperx if taperx is not None else e.finddefaultinfo('taperlength')
            dx = dx if dx is not None else self.finddefaultinfo('chiplength')
            e.info.guidewidth = width
            e.addonmodefilter(width,dx,inwidth,outwidth,modefilterx,taperx)
        return self
    def addchannel(self,width,dx=None,x0=None,y0=None):
        self.addmodefilter(width,dx,inwidth=0,outwidth=0,modefilterx=0,taperx=0,x0=x0,y0=y0)
        return self
    def addnotescircle(self,r=35500,bb=None,flat=False): # 71mm working dia for 76mm wafer
        g = Element('circle',layer='NOTES',parent=self)
        from numpy import linspace
        ws = list(linspace(0,1,5001))
        c = [(r*cos(2*pi*w),r*sin(2*pi*w)) for w in ws]
        if bb is not None:
            cs = cropcurves([c],bb) # c = clipcurve(c,bb); g.addpoly(c+[c[0]])
            g.addpolys(cs,unclosed=True) # curves are not closed!
        else:
            g.addpoly(c+[c[0]])
        if flat:
            if 1==flat:
                xflat,yflat,dx = 10160,36700,500
                self.addnote(0,yflat+25,note='x-cut LN wafer flat')
            else:
                xflat,yflat,dx = 10000,flat,500
                if 36400==flat:
                    self.addnote(0,yflat+25,note=f'x-cut TFLN wafer flat at {yflat/1000}mm radius')
                    # self.addnote(0,yflat+500,dy=500,note=f'x-cut TFLN wafer flat at {yflat/1000}mm radius')
                else:
                    self.addnote(0,yflat+25,note=f'wafer flat at {yflat/1000}mm radius')
            # g.addpoly([(-xflat,yflat),(0,yflat),(xflat,yflat)],unclosed=True)
            for n in range(int(xflat/dx)): # make a dotted line
                g.addpoly([(-xflat+(2*n+0)*dx,yflat),(-xflat+(2*n+0.5)*dx,yflat),(-xflat+(2*n+1)*dx,yflat)],unclosed=True)
        return self
    def addnotessquare(self,r=57150): # default is absolute max for Photosciences 5" mask
        #if dev: return self
        g = Element('square',layer='NOTES',parent=self)
        g.addpoly([(-r,-r),(r,-r),(r,r),(-r,r),(-r,-r)]) # centered at (0,0)
        # g.addpoly([(0,0),(2*r,0),(2*r,2*r),(0,2*r),(0,0)]) # corner at (0,0)
        return self
    def addnotesframe(self,margin,size=None,p0=(0,0)):
        g = Element('square',layer='NOTES',parent=self)
        mx,my = margin
        if size is None:
            bx,by,bdx,bdy = self.boundingbox()
            g.addpoly([(bx-mx,by-my),(bx+bdx+mx,by-my),(bx+bdx+mx,by+bdy+my),(bx-mx,by+bdy+my),(bx-mx,by-my)])
        else:
            g.addrect(p0[0]-mx,p0[1]-my,size[0]+2*mx,size[1]+2*my)
        return self
    def addshgmodefilter(self,width,period,gx=None,gy=None,dx=None,inwidth=0,outwidth=0,modefilterx=None,taperx=None,x0=None,y0=None,extendpoling=False,dc=None):
        e = Guide(parent=self)
        if inwidth: e.info.inputmodefilter = inwidth
        if outwidth: e.info.outputmodefilter = outwidth
        # if modefilterx: e.info.modefilterlength = modefilterx
        modefilterx = modefilterx if modefilterx is not None else e.finddefaultinfo('modefilterlength')
        # if taperx: e.info.taperlength = taperx
        taperx = taperx if taperx is not None else e.finddefaultinfo('taperlength')
        dx = dx if dx is not None else e.finddefaultinfo('chiplength')
        e.info.guidewidth = width
        gx = gx if gx is not None else (dx-2*(modefilterx+taperx) if not extendpoling else dx-(modefilterx+taperx)*(inwidth!=0)-(modefilterx+taperx)*(outwidth!=0))
        assert gy is not None
        x0,y0 = x0 if x0 is not None else self.x, y0 if y0 is not None else self.y # x,y are saved before being modified by addonmodefilter below
        e.addonmodefilter(width,dx,inwidth,outwidth,modefilterx,taperx)
        if extendpoling:
            e.addlayergrating(period,gx=gx,gy=gy,x0=x0+(modefilterx+taperx)*(inwidth!=0),y0=y0,dc=dc)
        else:
            e.addlayergrating(period,gx=gx,gy=gy,x0=x0+modefilterx+taperx,y0=y0,dc=dc)
        return self
    def addmultishgmodefilter(self,width,periods,gxs,dx=None,inwidth=0,outwidth=0,modefilterx=None,taperx=None,x0=None,y0=None,dc=None):
        e = Guide(parent=self)
        if inwidth: e.info.inputmodefilter = inwidth
        if outwidth: e.info.outputmodefilter = outwidth
        modefilterx = modefilterx if modefilterx is not None else e.finddefaultinfo('modefilterlength')
        taperx = taperx if taperx is not None else e.finddefaultinfo('taperlength')
        dx = dx if dx is not None else e.finddefaultinfo('chiplength')
        e.info.guidewidth = width
        x0,y0 = x0 if x0 is not None else self.x, y0 if y0 is not None else self.y # x,y are saved before being modified by addonmodefilter below
        e.addonmodefilter(width,dx,inwidth,outwidth,modefilterx,taperx)
        x0 = x0+modefilterx+taperx
        gxs = [(gx if gx>0 else dx-2*(modefilterx+taperx)-sum(gxs)) for gx in gxs]
        for period,gx in zip(periods,gxs):
            if period:
                e.addlayergrating(period,gx=gx,x0=x0,y0=y0,dc=dc)
            x0 += gx
        return self
    def addinterleavedshgmf(self,width,period1,period2,overpole,gx=None,gy=None,dx=None,inwidth=0,outwidth=0,modefilterx=None,taperx=None,x0=None,y0=None,extendpoling=False):
        e = Guide(parent=self)
        if inwidth: e.info.inputmodefilter = inwidth
        if outwidth: e.info.outputmodefilter = outwidth
        # if modefilterx: e.info.modefilterlength = modefilterx
        modefilterx = modefilterx if modefilterx is not None else e.finddefaultinfo('modefilterlength')
        # if taperx: e.info.taperlength = taperx
        taperx = taperx if taperx is not None else e.finddefaultinfo('taperlength')
        dx = dx if dx is not None else e.finddefaultinfo('chiplength')
        e.info.guidewidth = width
        gx = gx if gx is not None else (dx-2*(modefilterx+taperx) if not extendpoling else dx-(modefilterx+taperx)*(inwidth!=0)-(modefilterx+taperx)*(outwidth!=0))
        x0,y0 = x0 if x0 is not None else self.x, y0 if y0 is not None else self.y # x,y are saved before being modified by addonmodefilter below
        e.addonmodefilter(width,dx,inwidth,outwidth,modefilterx,taperx)
        if extendpoling:
            e.addinterleavedlayergrating(period1,period2,gx,gy,x0+(modefilterx+taperx)*(inwidth!=0),y0,overpole=overpole)
        else:
            e.addinterleavedlayergrating(period1,period2,gx,gy,x0+modefilterx+taperx,y0,overpole=overpole)
        return self
    def addconsecutiveshgmf(self,width,periods,gxs=None,inwidth=0,outwidth=0,gap=100,modefilterx=None,taperx=None,apodize=None):
        e = Guide(parent=self)
        if inwidth: e.info.inputmodefilter = inwidth
        if outwidth: e.info.outputmodefilter = outwidth
        modefilterx = modefilterx if modefilterx is not None else e.finddefaultinfo('modefilterlength')
        taperx = taperx if taperx is not None else e.finddefaultinfo('taperlength')
        dx = e.finddefaultinfo('chiplength')
        e.info.guidewidth = width
        if gxs:
            gap = (dx - 2*(modefilterx+taperx) - sum(gxs)) / (len(periods)-1)
        else:
            gx = (dx - 2*(modefilterx+taperx) - (len(periods)-1)*gap) / len(periods)
            gxs = gxs if gxs is not None else [gx for i in range(len(periods))]
        x0,y0 = self.x,self.y # x,y are gotten before being modified by addonmodefilter below
        e.addonmodefilter(width,dx,inwidth,outwidth,modefilterx,taperx)
        gx0s = [ x0 + modefilterx+taperx + i*gap + sum(gxs[:i]) for i in range(len(periods)) ]
        for i,(pj,gj,xj) in enumerate(zip(periods,gxs,gx0s)):
            e.addlayergrating(pj,gx=gj,x0=xj,y0=y0,apodize=apodize[i] if hasattr(apodize,'__len__') else apodize )
        return self
    def adddualshgmf(self,width,period1,period2,gx1=None,gx2=None,inwidth=0,outwidth=0,gap=100):
        e = Guide(parent=self)
        if inwidth: e.info.inputmodefilter = inwidth
        if outwidth: e.info.outputmodefilter = outwidth
        modefilterx = e.finddefaultinfo('modefilterlength')
        taperx = e.finddefaultinfo('taperlength')
        dx = e.finddefaultinfo('chiplength')
        e.info.guidewidth = width
        gx1 = gx1 if gx1 is not None else (dx-2*(modefilterx+taperx))/2.
        gx2 = gx2 if gx2 is not None else dx-2*(modefilterx+taperx)-gx1-gap
        x0,y0 = self.x,self.y # x,y are gotten before being modified by addonmodefilter below
        e.addonmodefilter(width,dx,inwidth,outwidth,modefilterx,taperx)
        e.addlayergrating(period1,gx=gx1,x0=x0+modefilterx+taperx,y0=y0)
        e.addlayergrating(period2,gx=gx2,x0=x0+modefilterx+taperx+gx1+gap,y0=y0)
        return self
    def addringguide(self,width,radius,yoffset,xinner=0,xoffset=None,invert=False,mf=0,modefilterx=0,taperx=5000,dx=None):
        dx = dx if dx is not None else self.finddefaultinfo('chiplength')
        xoffset = xoffset if xoffset is not None else 2*radius
        k = -1 if invert else +1
        e = Guide(parent=self,width=width)
        e.info.inputmodefilter,e.info.outputmodefilter,e.info.guidewidth = mf,mf,width
        if mf==0:
            e.addonchannel(modefilterx+taperx,width=width)
        else:
            e.addonchannel(modefilterx,width=mf).addontaper(taperx,outwidth=width)
        e.addonchannel(xoffset)
        e.addnote(y=e.y+0.5*k*yoffset,separation=yoffset)
        e.addonhalfring(radius,yoffset=yoffset,open=True,invert=invert)
        if xinner:
            e.y = e.y+k*yoffset
            e.addonchannel(xinner,note=False)
            e.x,e.y = e.x-xinner,e.y+k*2*radius
            e.addonchannel(xinner,note=False)
            e.x,e.y = e.x-xinner,e.y-k*2*radius-k*yoffset
            e.addonchannel(xinner)
        e.addonhalfring(radius,yoffset=yoffset,open=False,invert=invert)

        e.addonchannel(dx-2*modefilterx-2*taperx-xoffset-xinner)
        if mf==0:
            e.addonchannel(modefilterx+taperx,width=width)
        else:
            e.addontaper(taperx,outwidth=mf).addonchannel(modefilterx,width=mf)
        return self
    def addspiralguide(self,width,n,gap,invert=False,mf=0,modefilterx=0,taperx=5000,res=200,dx=None):
        assert 1==n%2, 'number of half turns must be'
        import phidls
        dx = dx if dx is not None else self.finddefaultinfo('chiplength')
        k = -1 if invert else +1
        e = Guide(parent=self,width=width)
        e.info.inputmodefilter,e.info.outputmodefilter,e.info.guidewidth = mf,mf,width
        if mf==0:
            e.addonchannel(modefilterx+taperx,width=width)
        else:
            e.addonchannel(modefilterx,width=mf).addontaper(taperx,outwidth=width)
        e.addnote(x=e.x,y=e.y+k*0.5*gap,separation=gap)
        D,L = phidls.spiral(width,0.5*n,gap,innergap=None,res=res,xin=0.5*n*gap,xout=0.5*n*gap)
        polys = phidls.layerpolygons(D,layers=[0],closed=True)
        polys = [[(e.x+x,e.y-k*y) for x,y in poly] for poly in polys]
        # import shapely
        # from shapely import geometry
        # pp = geometry.Polygon([(0,0), (0,3), (3,3), (3,0), (2,0), 
        #           (2,2), (1,2), (1,1), (2,1), (2,0), (0,0)])
        # print(pp.is_valid)
        # print(pp)
        # print(list(pp.exterior.coords))
        # for i,poly in enumerate(polys):
        #     pp = geometry.Polygon(poly)
        #     print(pp.is_valid)
        #     p = pp.buffer(0) # new Polygon
        #     print(p.is_valid)
        #     e.addpoly(p.exterior.coords)
        def removebowties(poly): # convert to valid, non-self-intersecting polygon
            from shapely import geometry
            pp = geometry.Polygon(poly) # print(pp.is_valid)
            p = pp.buffer(0) # new Polygon # print(p.is_valid)
            assert p.is_valid
            return p.exterior.coords
        polys = [removebowties(poly) for poly in polys]
        e.addpolys(polys)
        x,y,wx,wy = e.boundingbox()
        # print(e.checkpolygonvalidity())
        # e.plot()
        # for poly in polys:
        #     print(curveboundingbox(poly))
        # print('L',L,'wx',wx,'Δx',L-(wx-modefilterx-taperx))
        # plotcurvelist(polys,m=1,lw=0.2,ms=0.3,fewerticks=1)
        e.addnote(x=e.x-n*gap,y=e.y+k*0.5*n*gap,separation=wy-width)
        e.x,e.y = (x+wx,y+0.5*width) if invert else (x+wx,y+wy-0.5*width)
        e.addnote(x=e.x,y=e.y-k*0.5*wy,text=f"N={n:g}")
        e.addnote(x=e.x,y=e.y-k*0.25*wy,text=f"ΔL={L-(wx-modefilterx-taperx):.1f}")
        e.addonchannel(dx-modefilterx-taperx-wx,width=width)
        if mf==0:
            e.addonchannel(modefilterx+taperx,width=width)
        else:
            e.addontaper(taperx,outwidth=mf).addonchannel(modefilterx,width=mf)
        # if not invert: self.y += (n+1)*gap; self.parent.y += (n+1)*gap
        return self
    def addubendguide(self,radius,ubendcount,width,mf,xinner=None,period=None,modefilterx=0,taperx=3000,dx=None):
        dx = dx if dx is not None else self.finddefaultinfo('chiplength')
        xinner = xinner if xinner is not None else 3*radius
        xouter = period-xinner if period is not None else 2*xinner
        assert 0<xouter
        assert 0==ubendcount%2, 'ubendcount must be even'
        e = Guide(parent=self,width=width)
        e.info.inputmodefilter,e.info.outputmodefilter,e.info.guidewidth = mf,mf,width
        if mf==0:
            e.addonchannel(modefilterx+taperx,width=width)
        else:
            e.addonchannel(modefilterx,width=mf).addontaper(taperx,outwidth=width)
        for i in range(ubendcount//2):
            e.addonchannel(xinner/2 if i%2 else xouter/2,width=width,note=(i==ubendcount//2-1))
            e.addondoubleubend(radius,invert=i%2,note=(i==ubendcount//2-1))
            # plotcurvelist(e.subpolys(),m=1,lw=0.2,ms=0.3,fewerticks=1,aspect=1); exit()
            e.addonchannel(xouter/2 if i%2 else xinner/2,width=width,note=(i==ubendcount//2-1))
        e.addonchannel(dx-2*modefilterx-2*taperx-(ubendcount//2)*(xinner/2+xouter/2),width=width)
        if mf==0:
            e.addonchannel(modefilterx+taperx,width=width)
        else:
            e.addontaper(taperx,outwidth=mf).addonchannel(modefilterx,width=mf)
        return self
    def addsbendguide(self,sbendlength,sbendcount,pitch,width,mf,invert=False,modefilterx=0,taperx=3000,res=None):
        assert 0==sbendcount%2, 'sbendcount must be even'
        dx = self.finddefaultinfo('chiplength')-self.x
        x0,y0 = self.x,self.y
        e = Guide(parent=self)
        e.info.inputmodefilter,e.info.outputmodefilter,e.info.guidewidth = mf,mf,width
        if mf==0:
            e.addonchannel(modefilterx+taperx,width=width)
        else:
            e.addonchannel(modefilterx,width=mf).addontaper(taperx,outwidth=width)
        for i in range(sbendcount//2):
            e.addonsbend(sbendlength,-pitch if invert else +pitch,note=(i==sbendcount//2-1),res=res)
            e.addonsbend(sbendlength,+pitch if invert else -pitch,note=False,res=res)
            # if sbendlength==125: plotcurvelist(e.subpolys(),m=1,lw=0.2,ms=0.3,fewerticks=1,aspect=1); exit()
        e.addonchannel(dx-2*modefilterx-2*taperx-sbendcount*sbendlength,width=width)
        if mf==0:
            e.addonchannel(modefilterx+taperx,width=width)
        else:
            e.addontaper(taperx,outwidth=mf).addonchannel(modefilterx,width=mf)
    def addwdm(self,width,couplergap,couplerx,couplerwidth,
                mfsbendin,mfsbendout,mfchannelin,mfchannelout,
                modefilterx=0,taperx=5000,mirror=False,vgroovepitch=None,metrics=True,wdmatoutput=False,
                sb0=None,sb1=None):
        sb0,sb1 = (sb0 if sb0 is not None else self.finddefaultinfo('sbendlength')),(sb1 if sb1 is not None else self.finddefaultinfo('sbendlength'))
        vgroovepitch = vgroovepitch if vgroovepitch is not None else self.finddefaultinfo('vgroovepitch')
        dx = self.finddefaultinfo('chiplength')-self.x
        yb = vgroovepitch-couplergap
        # print('modefilterx,taperx,sbendlength,vgroovepitch',modefilterx,taperx,sbendlength,vgroovepitch)
        # print('modefilters:',mfsbendin,'\_/->',mfsbendout,',',mfchannelin,'-->',mfchannelout,'gap:',couplergap,'L0:',couplerx,'couplerguidewidth',couplerwidth,'criticaldimension:',couplergap-couplerwidth)
        x0,y0 = self.x,self.y
        gx = dx-(2*modefilterx+2*taperx+sb0+couplerx+sb1)
        e = Guide(parent=self)
        e.info.inputmodefilter,e.info.outputmodefilter,e.info.guidewidth = mfsbendin,mfsbendout,width
        if mfsbendin==mfchannelin==0:
            e.addonchannel(modefilterx+taperx+gx*wdmatoutput,width=width)
        else:
            e.addonchannel(modefilterx,width=mfsbendin).addontaper(taperx,outwidth=width)
            if wdmatoutput: e.addonchannel(gx)
        e.addonsbend(sb0,0 if mirror else +yb).addonchannel(couplerx,note=False)
        e.addnote(e.x-couplerx/2,e.y,gap=float2string(couplergap)+' split')
        e.addonsbend(sb1,0 if mirror else -yb)#.addontaper(taperx,outwidth=width)
        # gy0, gx0, gx = e.y, e.x, dx-(e.x-x0)-taperx-modefilterx
        gy0,gx0 = e.y,e.x
        if mfsbendout:
            if not wdmatoutput: e.addonchannel(gx)
            e.addontaper(taperx,outwidth=mfsbendout).addonchannel(modefilterx)
        else:
            e.addonchannel(gx*(not wdmatoutput)+taperx+modefilterx)

        if metrics:
            # xmetric,ymetric = x0+modefilterx+taperx+sb0+couplerx/2,y0+(vgroovepitch if mirror else width*6)
            xmetric,ymetric = gx0-sb1-couplerx/2,gy0+yb/2
            self.addoversizemetrics(width,xmetric,ymetric,nx=8,ny=6)

        ee = Guide(parent=self,guidespacing=vgroovepitch)
        ee.info.inputmodefilter,ee.info.outputmodefilter,ee.info.guidewidth = mfchannelin,mfchannelout,width
        if mfsbendin==mfchannelin==0:
            ee.addonchannel(modefilterx+taperx+gx*wdmatoutput,width=width)
        else:
            ee.addonchannel(modefilterx,width=mfchannelin).addontaper(taperx,outwidth=width)
            if wdmatoutput: ee.addonchannel(gx)
        ee.addonsbend(sb0,-yb if mirror else 0).addonchannel(couplerx)
        ee.addonsbend(sb1,+yb if mirror else 0)#.addontaper(taperx,outwidth=width)
        if mfchannelout:
            if not wdmatoutput: ee.addonchannel(gx)
            ee.addontaper(taperx,outwidth=mfchannelout).addonchannel(modefilterx)
        else:
            ee.addonchannel(gx*(not wdmatoutput)+taperx+modefilterx)
        f = Element('wdm',parent=ee) # only for informational purpose
        f.info.wdmgap,f.info.wdmlength,f.info.wdmguidewidth = couplergap,couplerx,width
        f.info.guidenumbers = e.info.guidenumber + '&' + ee.info.guidenumber
        return self

    def addqfc(self,mfpump,mfin,mfout,width,couplergap,couplerx,couplerwidth,period,pumponsbend,gratingonsbend,
            dx=None,samemfouts=False,extendpoling=False,grating=1,invertwdm=False,auxperiod=0,verbose=True):
        dx = dx if dx is not None else self.finddefaultinfo('chiplength')-self.x
        modefilterx = self.finddefaultinfo('modefilterlength')
        taperx = self.finddefaultinfo('taperlength')
        sbendlength = self.finddefaultinfo('sbendlength')
        vgroovepitch = self.finddefaultinfo('vgroovepitch')

        mfsbendin,mfsbendout,mfchannelin,mfchannelout = mfin,mfout,mfpump,mfpump
        if samemfouts:
            mfsbendin,mfsbendout,mfchannelin,mfchannelout = mfin,mfout,mfpump,mfout
        if pumponsbend:
            mfsbendin,mfsbendout,mfchannelin,mfchannelout = mfchannelin,mfchannelout,mfsbendin,mfsbendout
        if not pumponsbend and not gratingonsbend: # special hack that should only affect over-and-back wdm (swap output mfs)
            mfsbendout,mfchannelout = mfchannelout,mfsbendout; print('over-and-back')
        #print 'mfsbendin,mfsbendout,mfchannelin,mfchannelout',mfsbendin,'\_/->',mfsbendout,mfchannelin,'-->',mfchannelout,'gratingonsbend',gratingonsbend,'gap',couplergap,'couplerx',couplerx,'period',period
        if verbose: print('modefilters:',mfsbendin,'\_/->',mfsbendout,',',mfchannelin,'-->',mfchannelout,'gap:',couplergap,'L0:',couplerx,'couplerguidewidth',couplerwidth,'period:',period,'gratingonsbend:',gratingonsbend,'pumponsbend:',pumponsbend,'criticaldimension:',couplergap-couplerwidth)

        e = Guide(parent=self)
        x0,y0 = e.x,e.y
        e.info.inputmodefilter,e.info.outputmodefilter,e.info.guidewidth = mfsbendin,mfsbendout,width
        e.addonchannel(modefilterx,width=mfsbendin).addontaper(taperx,outwidth=couplerwidth)
        e.addonsbend(sbendlength,vgroovepitch-couplergap).addonchannel(couplerx)
        e.addnote(e.x-couplerx/2,e.y,gap=float2string(couplergap)+' split')
        if invertwdm:
            e.addonsbend(sbendlength,0).addontaper(taperx,outwidth=width)
        else:
            e.addonsbend(sbendlength,-vgroovepitch+couplergap).addontaper(taperx,outwidth=width)
        gx0, gx = e.x, dx-(e.x-x0)-taperx-modefilterx
        if mfsbendout:
            e.addonchannel(gx).addontaper(taperx,outwidth=mfsbendout).addonchannel(modefilterx)
        else:
            e.addonchannel(gx+taperx+modefilterx)
        # if ((invertwdm and not gratingonsbend) or (not invertwdm and gratingonsbend)) and grating:
        if 1==grating or 2<grating:
            e.addlayergrating(period,gx=gx+(taperx+modefilterx)*extendpoling*(0==mfsbendout),x0=gx0,y0=e.y)

        ee = Guide(parent=self,guidespacing=vgroovepitch)
        ee.info.inputmodefilter,ee.info.outputmodefilter,ee.info.guidewidth = mfchannelin,mfchannelout,width
        ee.addonchannel(modefilterx,width=mfchannelin).addontaper(taperx,outwidth=couplerwidth)
        ee.addonsbend(sbendlength,0).addonchannel(couplerx)
        if invertwdm:
            ee.addonsbend(sbendlength,vgroovepitch-couplergap).addontaper(taperx,outwidth=width)
        else:
            ee.addonsbend(sbendlength,0).addontaper(taperx,outwidth=width)
        if mfchannelout:
            ee.addonchannel(gx).addontaper(taperx,outwidth=mfchannelout).addonchannel(modefilterx)
        else:
            ee.addonchannel(gx+taperx+modefilterx)
        # if not ((invertwdm and not gratingonsbend) or (not invertwdm and gratingonsbend)) and grating:
        if 2==grating or 2<grating:
            ee.addlayergrating(period,gx=gx+(taperx+modefilterx)*extendpoling*(0==mfchannelout),x0=gx0,y0=ee.y)

        if 0<auxperiod:
            e.addlayergrating(period,gx=gx+(taperx+modefilterx)*extendpoling*(0==mfsbendout),x0=gx0,y0=0.5*(e.y+ee.y))
            if 2==auxperiod:
                print('auxperiod waveguide not implemented')
                raise NotImplementedError

        f = Element('wdm',parent=ee) # only for informational purpose
        f.info.wdmgap,f.info.wdmlength,f.info.wdmguidewidth = couplergap,couplerx,couplerwidth
        f.info.guidenumbers = e.info.guidenumber + '&' + ee.info.guidenumber
        return self
    def addqfcsymmetric(self,mfir,mfshg,width,couplergap,couplerx,couplerwidth,period,pumponsbend,gratingonsbend,
            samemfouts=False,extendpoling=False,grating=1,invertwdm=False,auxperiod=0):
        dx = self.finddefaultinfo('chiplength')
        modefilterx = self.finddefaultinfo('modefilterlength')
        taperx = self.finddefaultinfo('taperlength')
        sbendlength = self.finddefaultinfo('sbendlength')
        vgroovepitch = self.finddefaultinfo('vgroovepitch')
        guidespacing = self.finddefaultinfo('guidespacing')
        assert mfshg<mfir, 'assuming 1560,780,1560 input and output mode filters'
        assert 127==vgroovepitch, vgroovepitch
        assert 127/4==guidespacing, guidespacing

        x0,y0 = self.x,self.y
        # sb2dx = 1800
        # sb2dy = 2*guidespacing
        gx = dx-2*modefilterx-4*taperx-2*sbendlength-2*couplerx
        sx = dx-2*modefilterx-2*taperx-2*sbendlength-1*couplerx
        # sx = gx+taperx+sbendlength+couplerx

        e0 = Guide(x=x0,y=y0,parent=self) # input ir
        e0.addonchannel(modefilterx,width=mfir).addontaper(taperx,outwidth=couplerwidth)
        e0.addonsbend(sbendlength,vgroovepitch-couplergap).addonchannel(couplerx,note=0)
        e0.addnote(e0.x-couplerx/2,e0.y,gap=float2string(couplergap)+' split')
        e0.addonsbend(sbendlength,couplergap-vgroovepitch).addonchannel(sx)
        e0.addontaper(taperx,outwidth=mfir).addonchannel(modefilterx,width=mfir)

        e1 = Guide(x=x0,y=y0+vgroovepitch,parent=self) # input shg
        e1.addonchannel(modefilterx,width=mfshg).addontaper(taperx,outwidth=couplerwidth)
        e1.addonchannel(sbendlength).addonchannel(couplerx).addontaper(taperx,outwidth=width)
        gx0, gy0 = e1.x, e1.y
        e1.addonchannel(gx)
        e1.addlayergrating(period,gx=gx,x0=gx0,y0=gy0)
        gx1, gy1 = e1.x, e1.y
        e1.addontaper(taperx,outwidth=width).addonchannel(couplerx).addonchannel(sbendlength)
        e1.addontaper(taperx,outwidth=couplerwidth).addonchannel(modefilterx,width=mfshg)

        e2 = Guide(x=x0,y=y0+2*vgroovepitch,parent=self) # output ir
        e2.addonchannel(modefilterx,width=mfir).addontaper(taperx,outwidth=couplerwidth)
        e2.addonchannel(sx)
        e2.addonsbend(sbendlength,couplergap-vgroovepitch).addonchannel(couplerx,note=0)
        e2.addnote(e2.x-couplerx/2,e2.y,gap=float2string(couplergap)+' split')
        e2.addonsbend(sbendlength,vgroovepitch-couplergap)
        e2.addontaper(taperx,outwidth=mfir).addonchannel(modefilterx,width=mfir)

        assert e0.x==e1.x==e2.x, f"e0.x{e0.x}, e1.x{e1.x}, e2.x{e2.x}"
        self.y = y0+2*vgroovepitch+guidespacing
        f = Element('wdm',parent=e1) # only for informational purpose
        f.info.wdmgap,f.info.wdmlength,f.info.wdmguidewidth = couplergap,couplerx,couplerwidth
        f.info.guidenumbers = e0.info.guidenumber + '&' + e1.info.guidenumber + '&' + e2.info.guidenumber
        f.info.gx = gx
        return self
    def addqfcsymmetric2(self,mfir,mfshg,width,couplergap,couplerx,couplerwidth,period,pumponsbend,gratingonsbend,
            samemfouts=False,extendpoling=False,grating=1,invertwdm=False,auxperiod=0):
        dx = self.finddefaultinfo('chiplength')
        modefilterx = self.finddefaultinfo('modefilterlength')
        taperx = self.finddefaultinfo('taperlength')
        sbendlength = self.finddefaultinfo('sbendlength')
        vgroovepitch = self.finddefaultinfo('vgroovepitch')
        guidespacing = self.finddefaultinfo('guidespacing')
        assert mfshg<mfir, 'assuming 1560,780,1560 input and output mode filters'
        assert 127==vgroovepitch, vgroovepitch
        assert 127/4==guidespacing, guidespacing

        x0,y0 = self.x,self.y
        sb2dx = 1800
        sb2dy = 2*guidespacing

        e1 = Guide(x=x0,y=y0+guidespacing,parent=self) # input shg
        e1.addonchannel(modefilterx,width=mfshg).addontaper(taperx,outwidth=couplerwidth)
        e1.addonsbend(sbendlength,vgroovepitch-couplergap).addonchannel(couplerx).addontaper(taperx,outwidth=width)
        e1.addnote(e1.x-couplerx/2,e1.y,gap=float2string(couplergap)+' split')
        gx0, gy0, gx = e1.x, e1.y, dx-2*modefilterx-4*taperx-2*sbendlength-2*couplerx
        e1.addlayergrating(period,gx=gx,x0=gx0,y0=gy0)
        e1.addonchannel(gx)
        e1.addnote(e1.x+couplerx/2,e1.y,gap=float2string(couplergap)+' split')
        e1.addontaper(taperx,outwidth=couplerwidth).addonchannel(couplerx).addonsbend(sbendlength,vgroovepitch-couplergap)
        e1.addontaper(taperx,outwidth=mfshg).addonchannel(modefilterx,width=mfshg)
        sb2x = 2*taperx + sbendlength + couplerx + gx
        sb2y = vgroovepitch + guidespacing - 2*couplergap

        e0 = Guide(x=x0,y=y0,parent=self) # aux ir (output ir)
        e0.addonchannel(modefilterx,width=mfir).addontaper(taperx,outwidth=couplerwidth)
        e0.addonsbend(sb2x-sb2dx,sb2y-sb2dy).addonsbend(sb2dx,sb2dy).addonchannel(couplerx).addonchannel(sbendlength)
        e0.addontaper(taperx,outwidth=mfir).addonchannel(modefilterx,width=mfir)

        e2 = Guide(x=x0,y=y0+guidespacing+vgroovepitch,parent=self) # input ir
        e2.addonchannel(modefilterx,width=mfir).addontaper(taperx,outwidth=couplerwidth)
        e2.addonchannel(sbendlength).addonchannel(couplerx).addonsbend(sb2dx,sb2dy).addonsbend(sb2x-sb2dx,sb2y-sb2dy)
        e2.addontaper(taperx,outwidth=mfir).addonchannel(modefilterx,width=mfir)

        # self.y = e2.y
        self.y = y0+guidespacing+vgroovepitch + vgroovepitch

        f = Element('wdm',parent=e1) # only for informational purpose
        f.info.wdmgap,f.info.wdmlength,f.info.wdmguidewidth = couplergap,couplerx,couplerwidth
        f.info.guidenumbers = e0.info.guidenumber + '&' + e1.info.guidenumber + '&' + e2.info.guidenumber
        f.info.gx = gx
        return self
    def addqfc2wdm(self,mfpump,mfin,mfout,width,couplergap,couplerx,couplerwidth,period,dx=None,droppump=True): # wdm on both ends, can also be a MZ if droppump=False
        dx = dx if dx is not None else self.finddefaultinfo('chiplength')-self.x
        modefilterx = self.finddefaultinfo('modefilterlength')
        taperx = self.finddefaultinfo('taperlength')
        sbendlength = self.finddefaultinfo('sbendlength')
        vgroovepitch = self.finddefaultinfo('vgroovepitch')

        mfsbendin,mfsbendout,mfchannelin,mfchannelout = mfpump,mfpump,mfin,mfout # pumponsbend=True, gratingonsbend=False
        #print 'mfsbendin,mfsbendout,mfchannelin,mfchannelout',mfsbendin,'\_/->',mfsbendout,mfchannelin,'-->',mfchannelout,'gratingonsbend',gratingonsbend,'gap',couplergap,'couplerx',couplerx,'period',period
        print('modefilters:',mfsbendin,'\_/->',mfsbendout,',',mfchannelin,'-->',mfchannelout,'gap:',couplergap,'L0:',couplerx,'couplerguidewidth',couplerwidth,'period:',period,'criticaldimension:',couplergap-couplerwidth)

        e = Guide(parent=self)
        x0,y0 = e.x,e.y
        e.info.inputmodefilter,e.info.outputmodefilter,e.info.guidewidth = mfsbendin,mfsbendout,width
        e.addonchannel(modefilterx,width=mfsbendin).addontaper(taperx,outwidth=couplerwidth)
        e.addonsbend(sbendlength,vgroovepitch-couplergap).addonchannel(couplerx).addonsbend(sbendlength,-vgroovepitch+couplergap)
        e.addontaper(taperx,outwidth=width)
        gx0, gx = e.x, dx-2*(e.x-x0)
        if droppump:
            e.x += gx
        else:
            e.addonchannel(gx)
        e.addontaper(taperx,outwidth=couplerwidth)
        e.addonsbend(sbendlength,vgroovepitch-couplergap).addonchannel(couplerx).addonsbend(sbendlength,-vgroovepitch+couplergap)
        e.addontaper(taperx,outwidth=mfsbendout).addonchannel(modefilterx)

        ee = Guide(parent=self,guidespacing=vgroovepitch)
        ee.info.inputmodefilter,ee.info.outputmodefilter,ee.info.guidewidth = mfchannelin,mfchannelout,width
        ee.addonchannel(modefilterx,width=mfchannelin).addontaper(taperx,outwidth=couplerwidth)
        ee.addonsbend(sbendlength,0).addonchannel(couplerx).addonsbend(sbendlength,0)
        ee.addontaper(taperx,outwidth=width)
        ee.addonchannel(gx)
        ee.addontaper(taperx,outwidth=couplerwidth)
        ee.addonsbend(sbendlength,0).addonchannel(couplerx).addonsbend(sbendlength,0)
        ee.addontaper(taperx,outwidth=mfchannelout).addonchannel(modefilterx)

        ee.addlayergrating(period,gx=gx,x0=gx0,y0=ee.y)

        f = Element('wdm',parent=ee) # only for informational purpose
        f.info.wdmgap,f.info.wdmlength,f.info.wdmguidewidth = couplergap,couplerx,couplerwidth
        f.info.guidenumbers = e.info.guidenumber + '&' + ee.info.guidenumber
        return self
    def addqfc3(self,mfsbendin,mfsbendout,mfchannelin,mfchannelout,width,couplergap,couplerx,couplerwidth,period,gratingonsbend,dx=None,extendpoling=False,extendchannelpoling=True):
        dx = dx if dx is not None else self.finddefaultinfo('chiplength')-self.x
        modefilterx = self.finddefaultinfo('modefilterlength')
        taperx = self.finddefaultinfo('taperlength')
        sbendlength = self.finddefaultinfo('sbendlength')
        vgroovepitch = self.finddefaultinfo('vgroovepitch')
        precouplergap = 10+couplergap
        precouplerx = 3400
        #print('modefilters:',mfsbendin,'\_/->',mfsbendout,',',mfchannelin,'-->',mfchannelout,'gap:',couplergap,'L0:',couplerx,'couplerguidewidth',couplerwidth,'period:',period,'gratingonsbend:',gratingonsbend,'criticaldimension:',couplergap-couplerwidth)
        assert couplerx>50, 'coupler length in µm'

        e = Guide(parent=self)
        x0,y0 = e.x,e.y
        e.info.inputmodefilter,e.info.outputmodefilter,e.info.guidewidth = mfsbendin,mfsbendout,width
        e.addonchannel(modefilterx,width=mfsbendin).addontaper(taperx,outwidth=width)
        e.addonsbend(sbendlength,(vgroovepitch-precouplergap)).addontaper(taperx,outwidth=couplerwidth,note=0).addonsbend(precouplerx,(precouplergap-couplergap),note=0)
        e.addonchannel(couplerx,note=0)
        e.addnote(e.x-couplerx/2,e.y,gap=str(couplergap)+' split')
        e.addonsbend(precouplerx,-(precouplergap-couplergap)).addontaper(taperx,outwidth=width,note=0).addonsbend(sbendlength,-(vgroovepitch-precouplergap))
        gx0, gx = e.x, dx-(e.x-x0)-taperx-modefilterx
        if mfsbendout:
            e.addonchannel(gx).addontaper(taperx,outwidth=mfsbendout).addonchannel(modefilterx)
        else:
            e.addonchannel(gx+taperx+modefilterx)
        if gratingonsbend:
            e.addlayergrating(period,gx=gx+(taperx+modefilterx)*extendpoling*(0==mfsbendout),x0=gx0,y0=e.y)

        ee = Guide(parent=self,guidespacing=vgroovepitch)
        ee.info.inputmodefilter,ee.info.outputmodefilter,ee.info.guidewidth = mfchannelin,mfchannelout,width
        ee.addonchannel(modefilterx,width=mfchannelin).addontaper(taperx,outwidth=width)
        ee.addonsbend(sbendlength,0).addontaper(taperx,outwidth=couplerwidth).addonsbend(precouplerx,0)
        ee.addonchannel(couplerx)
        ee.addonsbend(precouplerx,0).addontaper(taperx,outwidth=width).addonsbend(sbendlength,0)
        if mfchannelout:
            ee.addonchannel(gx).addontaper(taperx,outwidth=mfchannelout).addonchannel(modefilterx)
        else:
            ee.addonchannel(gx+taperx+modefilterx)
        if not gratingonsbend:
            gx = gx + (taperx+modefilterx)*extendpoling*(0==mfchannelout) + (precouplerx+taperx+sbendlength)*extendchannelpoling
            gx0 = gx0 - (precouplerx+taperx+sbendlength)*extendchannelpoling
            ee.addlayergrating(period,gx=gx,x0=gx0,y0=ee.y)

        f = Element('wdm',parent=ee) # only for informational purpose
        f.info.wdmgap,f.info.prewdmgap,f.info.prewdmlength,f.info.wdmlength,f.info.wdmguidewidth = couplergap,precouplergap,precouplerx,couplerx,couplerwidth
        f.info.guidenumbers = e.info.guidenumber + '&' + ee.info.guidenumber
        print('  g'+f.info.guidenumbers+' criticaldimension:'+str(couplergap-couplerwidth))
        return self
    def addqfc4(self,mfsbendin,mfsbendout,mfchannelin,mfchannelout,width,couplergap,couplerx,couplerwidth,period,gratingonsbend,dx=None,extendpoling=False,extendchannelpoling=True):
        assert 0, 'ROCs appear wrong, verify before using'

        dx = dx if dx is not None else self.finddefaultinfo('chiplength')-self.x
        modefilterx = self.finddefaultinfo('modefilterlength')
        taperx = self.finddefaultinfo('taperlength')
        sbendlength = self.finddefaultinfo('sbendlength')
        vgroovepitch = self.finddefaultinfo('vgroovepitch')
        precouplergap = 10+couplergap
        precouplerx = 3400
        assert couplerx>50, 'coupler length in µm'
        e = Guide(parent=self)
        x0,y0 = e.x,e.y
        e.info.inputmodefilter,e.info.outputmodefilter,e.info.guidewidth = mfsbendin,mfsbendout,width
        e.addonchannel(modefilterx,width=mfsbendin).addontaper(taperx,outwidth=width)

        # e.addonsbend(sbendlength,(vgroovepitch-precouplergap)).addontaper(taperx,outwidth=couplerwidth,note=0).addonsbend(precouplerx,(precouplergap-couplergap),note=0)
        e.addonsplittapersbend(sbendlength+taperx+precouplerx/2,vgroovepitch-couplergap,taperx,precouplerx,10,inwidth=width,outwidth=couplerwidth,note=True)
        # print('e.info.splittaperdx',e.info.splittaperdx)

        e.addonchannel(couplerx,note=0)
        e.addnote(e.x-couplerx/2,e.y,gap=str(couplergap)+' split')

        # e.addonsbend(precouplerx,-(precouplergap-couplergap)).addontaper(taperx,outwidth=width,note=0).addonsbend(sbendlength,-(vgroovepitch-precouplergap))
        e.addonsplittapersbend(sbendlength+taperx+precouplerx/2,vgroovepitch-couplergap,taperx,precouplerx,10,inwidth=width,outwidth=couplerwidth,reverse=1,note=True)
        # print('e.info.splittaperdx',e.info.splittaperdx)

        gx0, gx = e.x, dx-(e.x-x0)-taperx-modefilterx
        if mfsbendout:
            e.addonchannel(gx).addontaper(taperx,outwidth=mfsbendout).addonchannel(modefilterx)
        else:
            e.addonchannel(gx+taperx+modefilterx)
        if gratingonsbend:
            e.addlayergrating(period,gx=gx+(taperx+modefilterx)*extendpoling*(0==mfsbendout)-e.info.splittaperdx,x0=gx0+e.info.splittaperdx,y0=e.y)

        ee = Guide(parent=self,guidespacing=vgroovepitch)
        ee.info.inputmodefilter,ee.info.outputmodefilter,ee.info.guidewidth = mfchannelin,mfchannelout,width
        ee.addonchannel(modefilterx,width=mfchannelin).addontaper(taperx,outwidth=width)
        ee.addonsbend(sbendlength,0).addontaper(taperx,outwidth=couplerwidth).addonsbend(precouplerx/2,0)
        ee.addonchannel(couplerx)
        ee.addonsbend(precouplerx/2,0).addontaper(taperx,outwidth=width).addonsbend(sbendlength,0)
        if mfchannelout:
            ee.addonchannel(gx).addontaper(taperx,outwidth=mfchannelout).addonchannel(modefilterx)
        else:
            ee.addonchannel(gx+taperx+modefilterx)
        if not gratingonsbend:
            ggx = gx + (taperx+modefilterx)*extendpoling*(0==mfchannelout) + (precouplerx/2+taperx+sbendlength)*extendchannelpoling - e.info.splittaperdx
            ggx0 = gx0 - (precouplerx/2+taperx+sbendlength)*extendchannelpoling + e.info.splittaperdx
            ee.addlayergrating(period,gx=ggx,x0=ggx0,y0=ee.y)

        f = Element('wdm',parent=ee) # only for informational purpose
        f.info.wdmgap,f.info.prewdmgap,f.info.prewdmlength,f.info.wdmlength,f.info.wdmguidewidth = couplergap,precouplergap,precouplerx/2,couplerx,couplerwidth
        f.info.guidenumbers = e.info.guidenumber + '&' + ee.info.guidenumber
        print('  g'+f.info.guidenumbers+' criticaldimension:'+str(couplergap-couplerwidth))
        return self
    def addqi(self,width,dx=None,inx=3000,innerpitch=100,outerpitch=200,taper=500,Ldc=10000,splitradius=1.0,verbose=True,sbendx=None,dconinput=False):
        sbendx = sbendx if sbendx is not None else {50:2400,100:3400,200:4800,400:6800,600:8100,800:9600}
        innersbend = sbendx[innerpitch]
        outersbend = sbendx[outerpitch]
        e = Guide(parent=self,x=self.x,y=self.y,width=width)
        dx = dx if dx is not None else self.finddefaultinfo('chiplength') # self.parent.parent.info.chipx-self.x # whereever the start, end is at guide end
        Lrf = dx-4*taper-2*innersbend-2*outersbend-2*inx-Ldc
        e.info.guidewidth = width
        e.addonchannel(inx)
        e.addontaper(taper,outwidth=2*width)
        # e.addondoublesbend(width=width)
        x1,y1 = e.x,e.y # print('x1',x1,'y1',y1)
        e.addondoublesbend(dx=outersbend,dy=outerpitch,splitradius=splitradius,poshalf=1,width=width)
        if dconinput:
            e.addonchannel(Ldc)
        e.addontaper(taper,outwidth=2*width)
        x2,y2 = e.x,e.y # print('x2',x2,'y2',y2)
        e.x,e.y = x1,y1
        e.addondoublesbend(dx=outersbend,dy=outerpitch,splitradius=splitradius,neghalf=1,width=width)
        if dconinput:
            e.addonchannel(Ldc)
        e.addontaper(taper,outwidth=2*width)
        x3,y3 = e.x,e.y # print('x3',x3,'y3',y3)
        def addinnermz(e,x,y):
            e.x,e.y = x,y
            e.addondoublesbend(dx=innersbend,dy=innerpitch,splitradius=splitradius,poshalf=1,width=width)
            e.addonchannel(Lrf)
            e.x,e.y = x,y
            e.addondoublesbend(dx=innersbend,dy=innerpitch,splitradius=splitradius,neghalf=1,width=width)
            e.addonchannel(Lrf)
            e.y = y
            e.addondoublesbend(dx=innersbend,dy=innerpitch,splitradius=splitradius,outputside=True)
            e.addontaper(taper,inwidth=2*width,outwidth=width)
            if not dconinput:
                e.addonchannel(Ldc)
        addinnermz(e,x3,y3)
        addinnermz(e,x2,y2)
        e.y = y1
        e.addondoublesbend(dx=outersbend,dy=outerpitch,splitradius=splitradius,outputside=True,width=width)
        e.addontaper(taper,inwidth=2*width,outwidth=width)
        e.addonchannel(inx)
        # print(dx-4*taper-2*innersbend-2*outersbend-2*inx-Ldc,26600+25000)
        if verbose: print('innerpitch',innerpitch,'outerpitch',outerpitch,'Ldc',Ldc,'Lrf',Lrf)
        self.info.mzlength,self.info.dclength = Lrf,Ldc
        return self
    def addchiplabels(self,chipx=None,text=None,x0=1000,y0=200,extratext=''): # self is the group, self.parent is the chip
        chipx = chipx if chipx is not None else self.finddefaultinfo('chiplength') # self.parent.info.chipx
        text = text if text is not None else self.parent.name+' '+self.finddefaultinfo('chipid')
        e = Element('metric',parent=self)
        for x in range(0,chipx,5000):
            #for i,n in enumerate([4,3,2,1]): e.addmetric(x-75*(i+1),0,25,n,2,6)
            e.addtext(text+extratext,x,0)
        e.translate(x0,y0)
        return self
    def addguidelabels(self,chipx=None,text=None,x0=None,y0=None,extratext='',dy=-25,skip=[],polingfill=0,repx=5000,xlast=None,metrics=None): # self is the group, self.parent is the chip
        chipx = chipx if chipx is not None else self.finddefaultinfo('chiplength') # self.parent.info.chipx
        text = text if text is not None else self.finddefaultinfo('chipid')+'-G'+str(self.finddefaultinfo('groupcount'))+'.1'  # self.parent.info.chipid+'-G'+str(self.parent.info.groupcount)+'.1'#+str(self.info.guidenumber) # G1.1,G2.1
        minfeature = self.finddefaultinfo('minfeature',fail=False) or self.finddefaultinfo('maskres')
        metrics = metrics if metrics is not None else [4,3,2,1]+[0.6]*(minfeature<=0.6)
        x0 = x0 if x0 is not None else self.x
        y0 = y0 if y0 is not None else self.y
        if polingfill:
            g = Element('metric',parent=self)
        for x in range(repx,int(chipx)-repx+1,repx):
            if x not in skip:
                e = Element('metric',parent=self)
                for i,n in enumerate(metrics):
                    e.addmetric(x-75*(i+1),dy,25,n,2,6)
                e.addtext('|'+str(int(x/1000))+'mm '+text+extratext,x,dy,fity=25)
                bx,by,bdx,bdy = e.boundingbox()
                if polingfill and xlast:
                    g.addlayergrating(polingfill,gx=bx-xlast-200,x0=xlast+100,y0=dy-15,note='')
                xlast = bx+bdx
                e.translate(x0,y0)
            # else:
            #     print('skipped guide label',x)
        if polingfill:
            g.translate(x0,y0)
        return self
    def addlayergrating(self,period,gx,gy=25,x0=0,y0=0,dc=None,note=None,apodize=None,layer='POLING'):
        g = Element('grating',layer=layer,parent=self) # g.addrect( x0,y0-gy/2.,gx,gy )
        dc = dc if dc is not None else polingdc(period)
        g.info.period,g.info.dc,g.info.bar = period,'%.3f' % dc, period*dc
        g.info.gratingstart,g.info.gratinglength,g.info.gratingwidth = x0,gx,gy
        ss = '%sΛ %d%% %d'%(float2string(period),100*dc,int(gx)) + ' apod'*bool(apodize)
        self.addnote(x0+gx/4,y0,grating=(('' if note=='' else ss+' '+str(note)) if note is not None else ss))
        # self.addnote(x0+gx/4,y0,grating='%s period, %.2f bar (%.2f%%), %d bars'%(period,period*dc,100*dc,len(grating(period,dc,gx,x0=x0)[0])))
        if dev: period *= periodmag
        barstarts,barends = grating(period,dc,gx,x0=x0,apodize=apodize)
        for xi,xj in zip(barstarts,barends):
            g.addrect( xi,y0-gy/2., xj-xi,gy )
        return self
    def addinterleavedlayergrating(self,period1,period2,gx,gy=20,x0=0,y0=0,overpole=0):
        print('warning - double-check overpole, min gap, and min bar in this implementation before using in production')
        g = Element('grating',layer='POLING',parent=self) # g.addrect( x0,y0-gy/2.,gx,gy )
        g.info.period1,g.info.period2 = period1,period2
        g.info.gratingstart,g.info.gratinglength,g.info.gratingwidth = x0,gx,gy
        self.addnote(x0+gx/4,y0,grating="%s Λ, %s Λ'"%(float2string(period1),float2string(period2)))
        if dev: period1,period2 = periodmag*period1,periodmag*period2
        barstarts,barends = interleavedgrating(period1,period2,padcount=1,gx=gx,overpole=overpole) # grating(period,dc,gx,x0=x0)
        barstarts,barends = shrinkbars(barstarts,barends,op=overpole)
        for xi,xj in zip(barstarts,barends):
            g.addrect( x0+xi,y0-gy/2., xj-xi,gy )
        return self
    def addbragggrating(self,width,dx,period=8,dc=1,enddc=None,x0=0,y0=0):
        angle = self.finddefaultinfo('braggangle',fail=False)
        angle = angle if angle is not None else 0
        if dev: period,angle = periodmag*period,arctan(periodmag*tan(angle))
        barcount = int(ceil(1.*dx/period))
        period = 1.*dx/barcount
        g = Element('bragg',parent=self)
        if enddc:
            g.info.braggperiod,g.info.braggstartdc,g.info.braggenddc,g.info.braggbar = '%.2f' % period, '%.3f' % dc, '%.3f' % enddc, '%.2f' % (period*min(dc,enddc))
        else:
            g.info.braggperiod,g.info.braggdc,g.info.braggbar = '%.2f' % period, '%.3f' % dc, '%.2f' % (period*dc)
        enddc = enddc if enddc is not None else dc
        barmiddles = [x*period for x in range(barcount+1)] # first and last bars will be half width
        from numpy import linspace
        barwidths = list(linspace(dc*period,enddc*period,barcount+1))
        for n,b0,bx in zip(range(barcount+1),barmiddles,barwidths):
            ys = [-width/2., -width/2., width/2., width/2.]
            xs = [b0-bx/2.-angle*width/2., b0+bx/2.-angle*width/2., b0+bx/2.+angle*width/2., b0-bx/2.+angle*width/2.]
            if 0==n:
                xs = [b0, b0+bx/2.-angle*width/2., b0+bx/2.+angle*width/2., b0]; assert bx/2.-angle*width/2.>0, 'too much angle on skew grating, redesign to account for barwidth shortening'
            if barcount==n:
                xs = [b0-bx/2.-angle*width/2., b0, b0, b0-bx/2.+angle*width/2.]; assert bx/2.-angle*width/2.>0, 'too much angle on skew grating, redesign to account for barwidth shortening'
            ps = [(xx+x0,yy+y0) for xx,yy in zip(xs,ys)]
            g.addpoly(ps,autoclose=True)
        return self
    def addpibragg(self,guidewidth,braggwidth,period,dc,dx=None):
        return self.addcorrugatedbragg(guidewidth=guidewidth,braggwidth=braggwidth,period=period,dc=dc,dx=dx,piphaseshift=True)
    def addcorrugatedbragg(self,guidewidth,braggwidth,period,dc,dx=None,piphaseshift=False,note=True):
        e = Guide(parent=self)
        dx = dx if dx is not None else self.finddefaultinfo('chiplength')
        x,y = e.x,e.y
        if 1==dc:
            e.addonchannel(dx,guidewidth)
        else:
            e.addoncorrugatedbragg(dx,guidewidth,braggwidth,period,dc,piphaseshift=piphaseshift)
            if note:
                e.addnote(x+dx/2,y,width=f"{guidewidth:g}")
                e.addnote(x+dx/2,y,ewidth0=f"  {braggwidth:g}")
                e.addnote(x+dx/2,y,length=dx)
                e.addnote(x+dx/2,y,bragg='Δφ=π '*piphaseshift+'%.3gΛ,%d%%'%(period,100*dc))
        return self
    def addbragg(self,width,dc,period=None,dx=None):
        e = Guide(parent=self)
        dx = dx if dx is not None else self.finddefaultinfo('chiplength')
        period = period if period is not None else self.finddefaultinfo('braggperiod')
        e.info.guidewidth = width
        if 1==dc:
            e.addonchannel(dx,width)
        else:
            e.addonbragg(dx,width,period,dc)
        return self
    def addbraggmodefilter(self,width,dx=None,indc=1,outdc=1,period=None,modefilterx=None,taperx=None,swapdcs=False):
        if swapdcs: indc,outdc = outdc,indc
        e = Guide(parent=self)
        if indc: e.info.inputbraggdutycycle = indc
        if outdc: e.info.outputbraggdutycycle = outdc
        if not indc and not outdc: modefilterx,taperx = 0,0
        if modefilterx: e.info.modefilterlength = modefilterx
        modefilterx = modefilterx if modefilterx is not None else e.finddefaultinfo('modefilterlength')
        if taperx: e.info.taperlength = taperx
        taperx = taperx if taperx is not None else e.finddefaultinfo('taperlength')
        dx = dx if dx is not None else self.finddefaultinfo('chiplength')
        period = period if period is not None else self.finddefaultinfo('braggperiod')
        e.info.guidewidth = width
        e.addonbraggmodefilter(width,period,dx,indc,outdc,modefilterx,taperx)
        return self
    def addtaperedbragggrating(self,inwidth,outwidth,dx,period=8,dc=1,enddc=None,x0=0,y0=0,note=True):
        x,y = self.x,self.y
        angle = self.finddefaultinfo('braggangle',fail=False)
        angle = angle if angle is not None else 0
        if dev: period,angle = periodmag*period,np.arctan(periodmag*np.tan(angle))
        barcount = int(ceil(1.*dx/period))
        period = 1.*dx/barcount
        g = Element('bragg',parent=self)
        if enddc:
            g.info.braggperiod,g.info.braggstartdc,g.info.braggenddc,g.info.braggbar = '%.2f' % period, '%.3f' % dc, '%.3f' % enddc, '%.2f' % (period*min(dc,enddc))
        else:
            g.info.braggperiod,g.info.braggdc,g.info.braggbar = '%.2f' % period, '%.3f' % dc, '%.2f' % (period*dc)
        enddc = enddc if enddc is not None else dc
        barmiddles = [x*period for x in range(barcount+1)] # first and last bars will be half width
        from numpy import linspace
        barwidths = list(linspace(dc*period,enddc*period,barcount+1))
        guidewidths = list(linspace(inwidth,outwidth,barcount+1))
        for n,b0,bx,width in zip(range(barcount+1),barmiddles,barwidths,guidewidths):
            #width = inwidth/2.+outwidth/2.
            ys = [-width/2., -width/2., width/2., width/2.]
            xs = [b0-bx/2.-angle*width/2., b0+bx/2.-angle*width/2., b0+bx/2.+angle*width/2., b0-bx/2.+angle*width/2.]
            if 0==n:
                xs = [b0, b0+bx/2.-angle*width/2., b0+bx/2.+angle*width/2., b0]; assert bx/2.-angle*width/2.>0, 'too much angle on skew grating, redesign to account for barwidth shortening'
            if barcount==n:
                xs = [b0-bx/2.-angle*width/2., b0, b0, b0-bx/2.+angle*width/2.]; assert bx/2.-angle*width/2.>0, 'too much angle on skew grating, redesign to account for barwidth shortening'
            ps = [(xx+x0,yy+y0) for xx,yy in zip(xs,ys)]
            g.addpoly(ps,autoclose=True)
        if note:
            self.addnote(x+dx/2,y,taper=dx)
        return self
    def addtaperedbraggmodefilter(self,mfwidth,width,dx=None,indc=1,outdc=1,braggperiod=None,modefilterx=None,taperx=None,swapdcs=False):
        if swapdcs: indc,outdc = outdc,indc
        e = Guide(parent=self)
        if indc: e.info.inputbraggdutycycle = indc
        if outdc: e.info.outputbraggdutycycle = outdc
        # if not indc and not outdc and mfwidth==width: modefilterx,taperx = 0,0
        if modefilterx: e.info.modefilterlength = modefilterx
        modefilterx = modefilterx if modefilterx is not None else e.finddefaultinfo('modefilterlength')
        if taperx: e.info.taperlength = taperx
        taperx = taperx if taperx is not None else e.finddefaultinfo('taperlength')
        dx = dx if dx is not None else self.finddefaultinfo('chiplength')
        braggperiod = braggperiod if braggperiod is not None else self.finddefaultinfo('braggperiod')
        e.info.guidewidth = width
        e.info.modefilterwidth = mfwidth
        e.addontaperedbraggmodefilter(mfwidth,width,braggperiod,dx,indc,outdc,modefilterx,taperx)
        return self
    def addtaperedbraggmodefilterwithshg(self,mfwidth,width,dx=None,indc=1,outdc=1,braggperiod=None,modefilterx=None,taperx=None,swapdcs=False,polingperiod=None):
        if swapdcs: indc,outdc = outdc,indc
        e = Guide(parent=self)
        if indc: e.info.inputbraggdutycycle = indc
        if outdc: e.info.outputbraggdutycycle = outdc
        # if not indc and not outdc and mfwidth==width: modefilterx,taperx = 0,0
        if modefilterx: e.info.modefilterlength = modefilterx
        modefilterx = modefilterx if modefilterx is not None else e.finddefaultinfo('modefilterlength')
        if taperx: e.info.taperlength = taperx
        taperx = taperx if taperx is not None else e.finddefaultinfo('taperlength')
        dx = dx if dx is not None else self.finddefaultinfo('chiplength')
        braggperiod = braggperiod if braggperiod is not None else self.finddefaultinfo('braggperiod')
        e.info.guidewidth = width
        e.info.modefilterwidth = mfwidth
        x0,y0 = self.x,self.y # x,y are saved after e created but before being modified
        e.addontaperedbraggmodefilter(mfwidth,width,braggperiod,dx,indc,outdc,modefilterx,taperx)
        gx = dx-(modefilterx+taperx)*2
        self.addlayergrating(polingperiod,gx=gx,x0=x0+modefilterx+taperx,y0=y0)
        return self
    def adddarpaqfcchip(self,maskname,chipx,ww):
        # 2250(out)+1550(pump)->920(in) DFG chip#   mfpump=3,mfin=2.5,mfout=4,pumponsbend=0
        # 2250(pump)+1550(in)->920(out) chip    #   mfpump=4,mfin=3,mfout=2.5,pumponsbend=0
        # 1950(pump)+1550(in)->864(out) chip    #   mfpump=3.5,mfin=3,mfout=2.5,pumponsbend=0
        # 1950(pump)+1520(in)->854(out) chip    #   mfpump=3.5,mfin=3,mfout=2.5,pumponsbend=0
        # 1340(pump)+780(out)->493(in)          #   mfpump=2.5,mfin=2.5,mfout=2.5,pumponsbend=1
        # 1550(pump)+1064(in)->631(out) chip    #   mfpump=3,mfin=2.5,mfout=2.5,pumponsbend=1
        # [lambda1=2250,lambda2=1550,period1=21.3,period2=21.9,wdmwidth=7,gap1=14,gap2=15,L1=4804,pumpmfwidth=4] ,shgperiod=16.0 ,mfpump=4,mfin=3,mfout=2.5
        # [lambda1=1950,lambda2=1550,period1=19.1,period2=19.3,wdmwidth=4,gap1=17,gap2=18,L1=9740,pumpmfwidth=3.5] ,shgperiod=16.0 ,mfpump=3.5,mfin=3,mfout=2.5
        # [lambda1=1950,lambda2=1520,period1=18.8,period2=19,wdmwidth=4,gap1=17,gap2=18,L1=9740,pumpmfwidth=3.5] ,shgperiod=16.0 ,mfpump=3.5,mfin=3,mfout=2.5
        # [lambda1=1340,lambda2=780,period1=5.04,period2=5.1,wdmwidth=5,gap1=8.5,gap2=9,L1=1869,pumpmfwidth=2.5] ,shgperiod=6.05 ,mfpump=2.5,mfin=2.5,mfout=2.5
        if '918DFG'==ww:
            self.addqfcchip(maskname,ww,chipx,lambda1=2250,lambda2=1550,period1=21.3,period2=21.9,wdmwidth=7,gap1=14,gap2=15,L1=4804,mfpump=3,mfin=2.5,mfout=4,pumponsbend=True,gratingonsbend=False)
        if '918'==ww:
            self.addqfcchip(maskname,ww,chipx,lambda1=2250,lambda2=1550,period1=21.3,period2=21.9,wdmwidth=7,gap1=14,gap2=15,L1=4804,mfpump=4,mfin=3,mfout=2.5)
        if '864'==ww:
            self.addqfcchip(maskname,ww,chipx,lambda1=1950,lambda2=1550,period1=19.1,period2=19.3,wdmwidth=4,gap1=17,gap2=18,L1=9740,mfpump=3.5,mfin=3,mfout=2.5)
        if '864OAB'==ww:
            self.addqfcchip(maskname,ww,chipx,lambda1=1950,lambda2=1550,period1=19.1,period2=19.3,wdmwidth=6,gap1=9,gap2=9.5,L1=1554,L2=2114,mfpump=3.5,mfin=3,mfout=2.5,gratingonsbend=False)
        if '854'==ww:
            self.addqfcchip(maskname,ww,chipx,lambda1=1950,lambda2=1520,period1=18.8,period2=19.0,wdmwidth=4,gap1=17,gap2=18,L1=9740,mfpump=3.5,mfin=3,mfout=2.5)
        if '493'==ww:
            self.addqfcchip(maskname,ww,chipx,lambda1=1340,lambda2=780,period1=5.04,period2=5.1,wdmwidth=5,gap1=8.5,gap2=9.0,L1=1869,mfpump=2.5,mfin=2.5,mfout=2.5,pumponsbend=True,gratingonsbend=False)
        if '631OBO'==ww:
            self.addqfcchip(maskname,ww,chipx,lambda1=1550,lambda2=1064,period1=10.0,period2=10.1,wdmwidth=8,gap1=9.0,gap2=9.0,L1=1286,L2=1486,mfpump=3.0,mfin=2.5,mfout=2.5,pumponsbend=True,gratingonsbend=False)
        if '532'==ww:
            self.addgreenmodefiltertestingchip(maskname,chipx)
        if '1588+1064 WDM'==ww:             # 637
            self.addqfc2chip(maskname,ww,chipx,lambda1=1588,lambda2=1064,period1=10.2,period2=10.4,wdmwidth=7,gap1=9.5,gap2=10.5,L1=2877,mfpump=3.5,mfin=2.5,mfout=2.5,pumponsbend=True,gratingonsbend=False)
        if '1064+637 A DFG WDM'==ww:        # 1588
            self.addqfc2chip(maskname,ww,chipx,lambda1=1064,lambda2=637,period1=10.2,period2=10.4,wdmwidth=6,gap1=7.7,gap2=8.3,L1=1906,mfpump=2.5,mfin=2.5,mfout=3.5,pumponsbend=True,gratingonsbend=False)
        if '1588+1064 OBO WDM'==ww:         # 637OBO
            self.addqfc2chip(maskname,ww,chipx,lambda1=1588,lambda2=1064,period1=10.2,period2=10.4,wdmwidth=8,gap1=8.7,gap2=9.1,L1=1386,mfpump=3.5,mfin=2.5,mfout=2.5,pumponsbend=True,gratingonsbend=False)
        if '1064+637 B DFG WDM'==ww:        # 1588B
            self.addqfc2chip(maskname,ww,chipx,lambda1=1064,lambda2=637,period1=10.2,period2=10.4,wdmwidth=6,gap1=8.7,gap2=9.3,L1=9906,mfpump=2.5,mfin=2.5,mfout=3.5,pumponsbend=True,gratingonsbend=False)
        if '1064+637 DFG'==ww:              # TNO
            self.addtnochip(maskname,chipx)
        if '1550+920 WDM'==ww:
            self.addqfc2chip(maskname,ww,chipx,lambda1=1550,lambda2=920,period1=8.1,period2=8.2,wdmwidth=5,gap1=9.2,gap2=10.0,L1=1845,mfpump=3.5,mfin=2.5,mfout=2.5,pumponsbend=True,gratingonsbend=False)
        if 'DEGOPO'==ww:
            self.adddegopochip(maskname,chipx)
        return self
    def addqfc2chip(self,maskname,ww,chiplength,lambda1,lambda2,period1,period2,wdmwidth,gap1,gap2,L1,mfpump,mfin,mfout,pumponsbend=False,gratingonsbend=True,L2=0):
        # differences compared to addqfcchip:
        #  ~ 5 WDMs have same L,P varying G (L=L1, P=period1, G=[gap1,,gap1/2+gap2/2,,gap2])
        #  ~ 1550 MF is bigger (3.5 not 3)
        #  ~ shgperiod=16.0 for 1064+1550
        #  ~ mf range 2..4.5 (not 1.5..4)
        #  ~ max(mfin,mfpump), not mfin for group 4&5
        #  ~ samemfouts=True on WDMs
        #  ~ label argument is now same as chip label

        L2 = L1 if 0==L2 else L2
        shgperiod,shgmf = 16.0,3.5
        if period1<7: shgperiod,shgmf = 6.0,2.5
        sfgtext = '  '+ww
        # sfgtext = '  '+str(lambda1)+'+'+str(lambda2)
        # if 'OBO'==ww[-3:] or 'OAB'==ww[-3:] or 'DFG'==ww[-3:]: sfgtext += ' '+ww[-3:]
        # if 'B'==ww[-1:]: sfgtext += ' '+ww[-1:]
        # sfgtext += ' WDM'
        self.addtextrelative(maskname+' '+self.info.chipid+sfgtext,x=2000,y=200)
        self.info.chiplength = chiplength
        self.info.chipwidth = 2500
        self.info.guidespacing = 38.5
        self.info.groupspacing = 79
        self.info.modefilterlength = 2500
        self.info.taperlength = 3000
        self.info.sbendlength = 2400
        self.info.vgroovepitch = 127
        self.adddiceguides()
        self.x,self.y = 0,300

        #     group 1: 2.0 to 4.5um mode filters (input and output)
        g1 = Element('group',parent=self)
        g1.addguidelabels()
        for wi in [2.0,2.5,3.0,3.5,4.0,4.5]:
            g1.addmodefilter(width=8,inwidth=wi,outwidth=wi)

        #    group 2: 2.0 to 4.5um channels (ie. whole chip length mode filters)
        g2 = Element('group',parent=self)
        g2.addguidelabels()
        for wi in [2.0,2.5,3.0,3.5,4.0,4.5]:
            g2.addchannel(width=wi)

        #    group 3: SHG channels (1550 16um or 1064 6um period), 4,6,8,10um width
        g3 = Element('group',parent=self)
        g3.addguidelabels()
        for wi in [4,6,8,10]:
            g3.addshgmodefilter(width=wi,period=shgperiod,inwidth=shgmf,outwidth=2.5)

        #    group 4: lambda1+lambda2 SFG channels (20um period), 4,6,8,10um width
        g4 = Element('group',parent=self)
        g4.addguidelabels()
        for wi in [4,6,8,10]:
            g4.addshgmodefilter(width=wi,period=period1,dx=chiplength,inwidth=max(mfin,mfpump),outwidth=mfout)

        #    group 5: lambda1+lambda2 SFG channels (8um width), ranging period
        g5 = Element('group',parent=self)
        g5.addguidelabels()
        deltaperiod = (period2-period1)/2.
        periods = [period1 + (n-1)*deltaperiod for n in range(6)]
        for ppi in periods:
            g5.addshgmodefilter(width=8,period=ppi,dx=chiplength,inwidth=max(mfin,mfpump),outwidth=mfout)

        #    group 6: 5 WDM devices, 5 different gaps, same period and length for each gap
        g6 = Element('group',parent=self)
        g6.addguidelabels(extratext=sfgtext)
        self.info.interaction = sfgtext
        gaps = [gap1 + n/4.*(gap2-gap1) for n in range(5)]
        for gap in gaps:
            g6.addqfc(mfpump,mfin,mfout,width=8,couplergap=gap,couplerx=L1,couplerwidth=wdmwidth,period=period1,pumponsbend=pumponsbend,gratingonsbend=gratingonsbend,samemfouts=True)
        print(self.info.chipid+sfgtext,'gaps:'+str(gaps),'period:'+str(period1),'L:'+str(L1))
        ### Leff = L + sqrt(pi/2*d0*R), µ=Aexp(d/d0) where T=sin²(µz), Lc = pi/2/µ
        ### for 1550nm, 7um width, 10um gap, L(10)=3.16, L(8)=.394, d0=(10-8)/ln(3.16/.394)=.96, R=10mm, sqrt(pi/2*d0*R)=123um
        ### for 1064nm, 6um width, 9um gap, L(10)=60.51, L(8)=1.725, d0=.56, R=10mm, sqrt(pi/2*d0*R)=94um

        return self
    def addsubmountmask(self,periods,dcs,numx=4,numy=20,padgaps=False): # todo: add dicing streets
        dicegap = 750
        chipx,chipy,padcount = 25000,5000,10
        if 60==len(periods):
            numx,numy = 3,20
            chipx,chipy,padcount = 30000,5000,12
        assert len(periods)==len(dcs)==numx*numy
        self.info.rows = numy
        nx,ny,x,y = 0,0,0,0
        for i in range(0,len(periods),1):
            nx,ny = i//numy,i%numy
            x,y = nx*(chipx+dicegap),ny*(chipy+dicegap)
            chip = Element('chip',parent=self)
            # chip.info.chiplength,chip.info.chipwidth = chipx,chipy
            chip.addsubmount(period=periods[i],dc=dcs[i],padcount=padcount,gapx=(100 if padgaps else 0),inputconnected=False,outputconnected=(False if padgaps else True))
            chip.translate(x-(chipx+dicegap)*numx/2.+dicegap/2.,y-(chipy+dicegap)*numy/2.+dicegap/2.)
            print(self.name,i,[nx,ny],periods[i],dcs[i],(x,y),chip.boundingbox()[:2],(chipx,chipy),chip.boundingbox()[2:],chip.info.text)
            if dev: break
        return self
    def adduagratingmask(self,period0s,period1s,dcs,gaps,ops,padgaps=True):
        numx,numy,chipx,chipy,dicegap,padcount = 5,20,30000,5000,750,12
        self.info.chiplength,self.info.chipwidth,self.info.dicegap,self.info.rows = chipx,chipy,dicegap,numy
        assert len(period0s)==len(period1s)==len(dcs)==numx*numy
        print(self)
        def chipxy(i):
            nx,ny = i//numy,i%numy
            xs,ys = [chipx+dicegap]*3+[10200],[chipy+dicegap]*(numy-1)
            return sum(xs[:nx]),sum(ys[:ny])
        for i in range(0,len(period0s),1):
            chip = Chip(parent=self)
            if dev and i not in [0,3,19,20,23,39,40,43,59,60,63,79,80,83,99,100,103]: continue
            if period0s[i]==period1s[i]:
                chip.addsubmount(period=period0s[i],dc=dcs[i],padcount=padcount,gapx=gaps[i],inputconnected=False,outputconnected=(False if padgaps else True))
            elif ops[i] is not None: # UA grating
                chip.adduagratingsubmount(expectedoverpole=ops[i],padcount=4,gapx=gaps[i],inputconnected=False,outputconnected=True)
            else:
                chip.addchirpsubmount(period0=period0s[i],period1=period1s[i],dc=dcs[i],padcount=4,gapx=gaps[i],inputconnected=False,outputconnected=(False if padgaps else True))
            chip.translate(*chipxy(i))
            print(self.name,i,period0s[i],period1s[i],dcs[i],gaps[i],chipxy(i),chip.boundingbox()[:2],(chipx,chipy),chip.boundingbox()[2:],chip.info.text)
        bx,by,bdx,bdy = self.boundingbox()
        self.translate(-bx-bdx//2,-by-bdy//2)
        return self
    def addchirpsubmountmask(self,period0s,period1s,dcs,gaps,ops,padgaps=True):
        dicegap = 750
        numx,numy,chipx,chipy,padcount = 5,20,25000,5000,10
        assert len(period0s)==len(period1s)==len(dcs)==numx*numy
        self.info.rows = numy
        self.info.chiplength,self.info.chipwidth,self.info.dicegap = chipx,chipy,dicegap
        nx,ny,x,y = 0,0,0,0
        for i in range(0,len(period0s),1):
            nx,ny = i//numy,i%numy
            x,y = nx*(chipx+dicegap),ny*(chipy+dicegap)
            if i in [80,120]: self.x += 10000-chipx
            # if i in [80,120]: self.x += 10200-chipx-dicegap
            # if i in [100,140]: self.x += 10000-chipx
            chip = Chip(parent=self)
            if i not in [0,3,19,20,23,39,40,43,59,60,63,79,80,83,99,100,103]: continue
            if period0s[i]==period1s[i]:
                chip.addsubmount(period=period0s[i],dc=dcs[i],padcount=padcount,gapx=gaps[i],inputconnected=False,outputconnected=(False if padgaps else True))
            else:
                chip.addchirpsubmount(period0=period0s[i],period1=period1s[i],dc=dcs[i],padcount=4,gapx=gaps[i],inputconnected=False,outputconnected=(False if padgaps else True))
            chip.translate(x-(chipx+dicegap)*numx/2.+dicegap/2.,y-(chipy+dicegap)*numy/2.+dicegap/2.)
            print(self.name,i,[nx,ny],period0s[i],period1s[i],dcs[i],gaps[i],(x,y),chip.boundingbox()[:2],(chipx,chipy),chip.boundingbox()[2:],chip.info.text)
        bx,by,bdx,bdy = self.boundingbox()
        self.translate(-bx-bdx//2,-by-bdy//2)
        return self
    def addnanjingsubmountmask(self,period0s,period1s,dcs,gaps,temperatures,labels,padgaps=False):
        dicegap = 750
        numx,numy,chipx,chipy,padcount = 5,20,25000,5000,10
        assert len(period0s)==len(period1s)==len(dcs)==numx*numy
        self.info.rows = numy
        self.info.chiplength,self.info.chipwidth,self.info.dicegap = chipx,chipy,dicegap
        nx,ny,x,y = 0,0,0,0
        for i in range(0,len(period0s),1):
            nx,ny = i//numy,i%numy
            x,y = nx*(chipx+dicegap),ny*(chipy+dicegap)
            if i in [80,120]: self.x += 10000-chipx
            chip = Chip(parent=self)
            if dev and i not in [0,3,19,20,23,39,40,43,59,60,63,72,79,80,83,92,99,100]: continue
            if period0s[i]==period1s[i]:
                chip.addsubmount(period=period0s[i],dc=dcs[i],padcount=padcount,gapx=gaps[i],inputconnected=False,outputconnected=(False if padgaps else True))
            else:
                chip.addentwinedsubmount(period0=period0s[i],period1=period1s[i],padcount=4,temperature=temperatures[i],label=labels[i],gapx=gaps[i],inputconnected=False,outputconnected=(False if padgaps else True))
            chip.translate(x-(chipx+dicegap)*numx/2.+dicegap/2.,y-(chipy+dicegap)*numy/2.+dicegap/2.)
            print(self.name,i,[nx,ny],period0s[i],period1s[i],dcs[i],gaps[i],(x,y),chip.boundingbox()[:2],(chipx,chipy),chip.boundingbox()[2:],chip.info.text)
        bx,by,bdx,bdy = self.boundingbox()
        self.translate(-bx-bdx//2,-by-bdy//2)
        return self
    def addelectrodemask(self,chipxys,electrodes=None):
        def electrodelist(infile,outfile):
            with open(infile,'r') as f:
                es = []
                for line in f:
                    if line.startswith('qpmtype'):
                        n,e = 1,OrderedDictFromCSV(line)
                        if 'quantity' in e:
                            n = int(e.quantity)
                            del e.quantity
                        for i in range(n): es.append(e.copy())
            return es
        if not 'test'==self.name:
            infile,outfile = self.name+'in.csv',self.name+'out.csv'
            electrodes = electrodes if electrodes is not None else electrodelist(infile,outfile)
        else:
            numx,numy = 1,2
            electrodes = [OrderedDictFromCSV([('qpmtype', 'standard'), ('period', 6.93), ('dutycycle', 0.3), ('padgap', 100), ('padcount', 12), ('inputconnected', 0), ('outputconnected', 0)]), OrderedDictFromCSV([('qpmtype', 'standard'), ('period', 6.93), ('dutycycle', 0.3), ('padgap', 100), ('padcount', 12), ('inputconnected', 0), ('outputconnected', 0)])]
        # assert len(electrodes)==numx*numy, f'electrode list:{len(electrodes)} expected:{numx}x{numy}={numx*numy} '
        assert len(electrodes)==len(chipxys), f'electrode list count:{len(electrodes)} chipxys list count:{len(chipxys)} '
        for i,e in enumerate(electrodes):
            submount = Submount(parent=self)
            # if dev and i not in [0,19,20,30,40,50,51,60,70,71,72,73,74,75,76,77,78,79,79,80,81,82,83,84,90,99,100,119]: continue # if dev and i not in [0,8,19,20,28,39,40,48,59,60,68,79,80,88,99]: continue
            e.padcount,e.padgap = e.get('padcount',10),e.get('padgap',0)
            e.inputconnected,e.outputconnected = e.get('inputconnected',False),e.get('outputconnected',False if e.padgap else True) # outputconnected=True if padgap==0 by default
            e.chipid = submount.info.chipid
            if 'standard'==e.qpmtype:
                e.dutycycle = max(self.finddefaultinfo('maskres'),e.dutycycle*e.period)/e.period
                kwargs = {'gy':e.padwidth} if 'padwidth' in e else {}
                kwargs = {**kwargs, 'paddesign':e.paddesign} if 'paddesign' in e else kwargs
                kwargs = {**kwargs, 'omitpads':e.omitpads} if 'omitpads' in e else kwargs
                submount.addsubmount(period=e.period,dc=e.dutycycle,padcount=e.padcount,gapx=e.padgap,
                    breakupgapsize=e.breakupgapsize if 'breakupgapsize' in e else 0,
                    inputconnected=e.inputconnected,outputconnected=e.outputconnected,apodize=e.get('apodize',None),**kwargs)
            elif 'short'==e.qpmtype:
                e.dutycycle = max(self.finddefaultinfo('maskres'),e.dutycycle*e.period)/e.period
                submount.addshortsubmount(period=e.period,dc=e.dutycycle,poledlength=e.poledlength,padcount=e.padcount,gapx=e.padgap,inputconnected=e.inputconnected,outputconnected=e.outputconnected)
            # elif 'chirped'==e.qpmtype:
            #     e.dutycycle = max(self.finddefaultinfo('maskres'),e.dutycycle*min(e.startperiod,e.endperiod))/min(e.startperiod,e.endperiod)
            #     submount.addchirpsubmount(period0=e.startperiod,period1=e.endperiod,dc=e.dutycycle,padcount=e.padcount,gapx=e.padgap,inputconnected=e.inputconnected,outputconnected=e.outputconnected)
            elif 'chirped'==e.qpmtype:
                e.dutycycle = max(self.finddefaultinfo('maskres'),e.dutycycle*min(e.startperiod,e.endperiod))/min(e.startperiod,e.endperiod)
                submount.addchirpsubmount(period0=e.startperiod, period1=e.endperiod, dc=e.dutycycle, padcount=e.padcount, gapx=e.padgap, 
                        inputconnected=e.inputconnected, outputconnected=e.outputconnected, apodize=e.get('apodize',None), 
                        opmin=e.get('expectedoverpole',e.get('expectedoverpolemin',None)), opmax=e.get('expectedoverpolemax',None))
            elif 'custom'==e.qpmtype:
                e.outputconnected = True
                e.label = 'UA'+e.filename[9] if e.filename.startswith('uagrating') else (e.filename[:-4] if e.filename.endswith('.dat') else e.filename).title()
                labels = e.get('labels',[])
                breakupgapsize = e.get('breakupgapsize',0)
                # submount.adduagratingsubmount(expectedoverpole=e.expectedoverpole,padcount=e.padcount,gapx=e.padgap,inputconnected=e.inputconnected,outputconnected=e.outputconnected)
                submount.addcustomsubmount(filename=e.filename,label=e.label,expectedoverpole=e.expectedoverpole,padcount=e.padcount,gapx=e.padgap,inputconnected=e.inputconnected,outputconnected=e.outputconnected,breakupgapsize=breakupgapsize,breakupgapbar=1,labels=labels,minfeature=e.minfeature)
            elif 'fixeddomain'==e.qpmtype:
                e.outputconnected = True
                # e.label = 'UA'+e.filename[9] if e.filename.startswith('uagrating') else (e.filename[:-4] if e.filename.endswith('.dat') else e.filename).title()
                # # submount.adduagratingsubmount(expectedoverpole=e.expectedoverpole,padcount=e.padcount,gapx=e.padgap,inputconnected=e.inputconnected,outputconnected=e.outputconnected)
                # submount.addfixeddomainsubmount(filename=e.filename,expectedoverpole=e.expectedoverpole,padcount=e.padcount,gapx=e.padgap,inputconnected=e.inputconnected,outputconnected=e.outputconnected)
                submount.addfixeddomainsubmount(period=e.period,padcount=e.padcount,gapx=e.padgap,
                    inputconnected=e.inputconnected,outputconnected=e.outputconnected)
            elif 'interleaved'==e.qpmtype:
                e.outputconnected = True
                if 'order' in e: e.period0,e.period1 = e.order*e.period0,e.order*e.period1
                e.label = e.label if 'label' in e else f'{"+" if e.period1-e.period0>=0 else ""}{e.period1-e.period0:.4f}'
                submount.addinterleavedsubmount(period0=e.period0, period1=e.period1, padcount=e.padcount, label0=e.label, gapx=e.padgap, 
                    inputconnected=e.inputconnected, outputconnected=e.outputconnected,
                    smallestbar=e.smallestbar if 'smallestbar' in e else 1,
                    breakupgapsize=e.breakupgapsize if 'breakupgapsize' in e else 0,
                    apodize=e.apodize if 'apodize' in e else None)
            elif 'phaseflip'==e.qpmtype:
                e.outputconnected = True
                submount.addphaseflipsubmount(period=e.period, n=e.n, padcount=e.padcount, gapx=e.padgap, 
                    inputconnected=e.inputconnected, outputconnected=e.outputconnected, apodize=e.get('apodize',None))
            elif 'alternating'==e.qpmtype:
                e.outputconnected = True
                e.repeatlength = e.repeatlength if 'repeatlength' in e else 0
                if 'order' in e: e.period0,e.period1 = e.order*e.period0,e.order*e.period1
                e.label = e.label if 'label' in e else f'{"+" if e.period1-e.period0>=0 else ""}{e.period1-e.period0:.4f}'
                e.dc0 = max(self.finddefaultinfo('maskres'),e.dc0*e.period0)/e.period0
                e.dc1 = max(self.finddefaultinfo('maskres'),e.dc1*e.period1)/e.period1
                submount.addalternatingsubmount(period0=e.period0, period1=e.period1, dc0=e.dc0, dc1=e.dc1, repeatlength=e.repeatlength, padcount=e.padcount, label0=e.label, gapx=e.padgap, inputconnected=e.inputconnected, outputconnected=e.outputconnected)
            else:
                assert 0, f'submount qpmtype:{e.qpmtype} not found'
            print('e',e)
            submount.translate(*chipxys[i]) # print(self.name,i,period0s[i],period1s[i],dcs[i],gaps[i],chipxy(i),chip.boundingbox()[:2],(chipx,chipy),chip.boundingbox()[2:],chip.info.text)
        if not 'test'==self.name:
            with open(outfile,'w') as f:
                for e in electrodes: print(str(e).replace(':',','),file=f)
        return self
    def adddicelanes(self,xx,yy,layers=['MASK','DICE'],maxradius=None,honly=False,stripping=False):
        def addcenteredrect(self,x0,y0,dx,dy,swapxy=False):
            return self.addrect(y0-dy/2,x0-dx/2,dy,dx) if swapxy else self.addrect(x0-dx/2,y0-dy/2,dx,dy)
        ee = [Element('dicinglanes',layer=layer,parent=self) for layer in layers]
        for e in ee:
            for y in yy:
                e.adddashedline(xx[0]/2+xx[-1]//2, y, xx[-1]-xx[0], 100, 10000, 1000, xaxis=1, maxradius=maxradius)
            if not honly:
                for x in xx:
                    e.adddashedline(x, yy[0]/2+yy[-1]//2, 100, yy[-1]-yy[0], 10000, 1000, xaxis=0, maxradius=maxradius)
        if stripping:
            eee = [Element('dicinglanes',layer='STRIPPING',parent=self) for layer in layers]
            for e in eee:
                for y in yy:
                    e.addcenteredrect(xx[0]/2+xx[-1]//2, y, xx[-1]-xx[0], 200)
                if not honly:
                    for x in xx:
                        e.addcenteredrect(x, yy[0]/2+yy[-1]//2, 200, yy[-1]-yy[0])
    def addsplitter(self,width,pitch,mfin=0,mfout=0,mzlength=None,sbendlength=None,splitradius=0.5,singlemfout=False,taperx=None,mftaperx=None):
        dx = self.finddefaultinfo('chiplength')
        modefilterx = self.finddefaultinfo('modefilterlength') if mfin or mfout else 0
        taperx = self.finddefaultinfo('taperlength') if taperx is None else taperx
        mftaperx = self.finddefaultinfo('taperlength') if mftaperx is None else mftaperx
        sbendlength = sbendlength if sbendlength is not None else self.finddefaultinfo('sbendlength')
        # e = Element('ysplitter-guide',parent=self)
        e = Guide('ysplitter-guide',parent=self)
        e.y += pitch/2 # self.y lines up with first arm, e.y will line up with input
        e.info.inputmodefilter,e.info.outputmodefilter,e.info.guidewidth = mfin,mfout,width
        cx = (dx -(2*modefilterx*(mfin>0) + 2*mftaperx*(mfin>0) + 2*taperx + 2*sbendlength + mzlength))/2 if mzlength is not None else modefilterx
        if mfin:
            e.addonchannel(modefilterx,width=mfin).addontaper(mftaperx,outwidth=width)
        e.addonchannel(cx,width=width)
        e.addontaper(taperx,outwidth=2*width)
        x0, y0 = e.x, e.y
        dx, dy = dx - (modefilterx*(mfin>0) + mftaperx*(mfin>0) + modefilterx*(mfout>0) + mftaperx*(mfout>0) + cx + taperx + sbendlength), width/2
        e.addondoublesbend(sbendlength,pitch,width,x=x0,y=y0,neghalf=True,splitradius=splitradius).addonchannel(dx)
        if mfout: e.addontaper(mftaperx,outwidth=mfout).addonchannel(modefilterx)
        e.addondoublesbend(sbendlength,pitch,width,x=x0,y=y0,poshalf=True,splitradius=splitradius).addonchannel(dx)
        if mfout:
            if singlemfout:
                e.addonchannel(mftaperx).addonchannel(modefilterx)
            else:
                e.addontaper(mftaperx,outwidth=mfout).addonchannel(modefilterx)
        e.info.ysplitterpitch = pitch
        self.parent.y = self.y = e.y # self.y and self.parent.y will line up with second arm (for placement of next guide)
        e.y -= pitch/2 # e.y will line up with output
        return self
    # def adddualmfsplitter(self,width,pitch,mfin=0,mfout=0,mzlength=None,sbendlength=None,splitradius=0,singlemfout=False,taperx=None,mftaperx=None):
    #     dx = self.finddefaultinfo('chiplength')
    #     modefilterx = self.finddefaultinfo('modefilterlength')
    #     taperx = self.finddefaultinfo('taperlength') if taperx is None else taperx
    #     mftaperx = self.finddefaultinfo('taperlength') if mftaperx is None else mftaperx
    #     sbendlength = sbendlength if sbendlength is not None else self.finddefaultinfo('sbendlength')
    #     channelx = 4000
    #     e = Element('ysplitter-guide',parent=self)
    #     e.y += pitch/2 # self.y lines up with first arm, e.y will line up with input
    #     e.info.inputmodefilter,e.info.outputmodefilter,e.info.guidewidth = mfin,mfout,width
    #     # cx = (dx -(2*modefilterx + 2*mftaperx + 2*taperx + 2*sbendlength + mzlength))/2 if mzlength is not None else modefilterx
    #     cx = (dx -(2*channelx + 2*modefilterx + 4*mftaperx + 2*taperx + 2*sbendlength + mzlength))/2 if mzlength is not None else modefilterx
    #     if mfin:
    #         e.addonchannel(modefilterx,width=mfin).addontaper(mftaperx,outwidth=mfin/2+width/2)
    #         e.addonchannel(channelx,width=mfin/2+width/2).addontaper(mftaperx,outwidth=width)
    #     e.addonchannel(cx,width=width)
    #     e.addontaper(taperx,outwidth=2*width)
    #     x0, y0 = e.x, e.y
    #     dx, dy = dx - (2*channelx + 2*modefilterx + 4*mftaperx + cx + taperx + sbendlength), width/2
    #     e.addondoublesbend(sbendlength,pitch,width,x=x0,y=y0,neghalf=True,splitradius=splitradius).addonchannel(dx)
    #     e.addontaper(mftaperx,outwidth=mfin/2+width/2).addonchannel(channelx)
    #     e.addontaper(mftaperx,outwidth=mfout).addonchannel(modefilterx)
    #     e.addondoublesbend(sbendlength,pitch,width,x=x0,y=y0,poshalf=True,splitradius=splitradius).addonchannel(dx)
    #     if singlemfout:
    #         e.addonchannel(2*mftaperx+channelx+modefilterx)
    #     else:
    #         e.addontaper(mftaperx,outwidth=mfin/2+width/2).addonchannel(channelx)
    #         e.addontaper(mftaperx,outwidth=mfout).addonchannel(modefilterx)
    #     e.info.ysplitterpitch = pitch
    #     self.parent.y = self.y = e.y # self.y and self.parent.y will line up with second arm (for placement of next guide)
    #     e.y -= pitch/2 # e.y will line up with output
    #     return self
    def addscurves(self,width,mf=0,sbendcount=1,sbendpitch=127/2.,sbendlength=None):
        dx = self.finddefaultinfo('chiplength')
        modefilterx = self.finddefaultinfo('modefilterlength')
        taperx = self.finddefaultinfo('taperlength')
        sbendlength = sbendlength if sbendlength is not None else self.finddefaultinfo('sbendlength')
        # e = Element('scurve-guide',parent=self)
        e = Guide('scurve-guide',parent=self)
        e.info.guidewidth = width
        if mf and not mf==width: e.info.inputmodefilter,e.info.outputmodefilter = mf,mf
        dx = (dx - (2*modefilterx*(mf>0) + 2*taperx*(mf>0) + 2*sbendcount*sbendlength))/2.
        dy = width/2.
        if mf:
            e.addonchannel(modefilterx,width=mf).addontaper(taperx,outwidth=width)
        e.addonchannel(dx,width=width)
        for n in range(sbendcount):
            e.addonsbend(sbendlength,+sbendpitch-dy,width).addonsbend(sbendlength,-sbendpitch+dy,width)
        e.addonchannel(dx,width=width)
        if mf:
            e.addontaper(taperx,outwidth=mf).addonchannel(modefilterx)
        e.info.sbendpitch = sbendpitch
        e.info.sbendcount = sbendcount
        return self
    def addmfsplittermachzehnder(self,width,pitch,mzlength=0,mf=0,sbendlength=None,splitradius=0,xoffset=0,y=None,**kwargs):
        # for an unbalanced mz, supply either deltawidth, deltalength, or deltabragg # first arm is normal, second arm is modified
        dx = self.finddefaultinfo('chiplength')
        modefilterx = self.finddefaultinfo('modefilterlength')
        taperx = self.finddefaultinfo('taperlength')
        sbendlength = sbendlength if sbendlength is not None else self.finddefaultinfo('sbendlength')
        e = Guide('machzehnder-guide',parent=self)
        # e.y += pitch/2 # self.y lines up with first arm, e.y will line up with input
        e.y = e.y+pitch/2 if y is None else y # self.y lines up with first arm, e.y will line up with input
        e.info.guidewidth = width
        if mf and not mf==width: e.info.inputmodefilter,e.info.outputmodefilter = mf,mf
        cx = modefilterx
        dx,dy = dx - (2*modefilterx*(mf>0) + 2*taperx*(mf>0) + 2*cx + 2*taperx + 2*sbendlength), width/2.
        if mzlength: cx,dx = cx+(dx-mzlength)/2.,mzlength
        if mf:
            e.addonchannel(modefilterx,width=mf).addontaper(taperx,outwidth=width)
        e.addonchannel(cx+xoffset,width=width).addontaper(taperx,outwidth=2*width)
        x0,y0 = e.x,e.y
        self.info.mzx0,self.info.mzy0 = e.x+sbendlength,e.y
        e.addondoublesbend(sbendlength,pitch,width,x=x0,y=y0,neghalf=True,splitradius=splitradius).addonchannel(dx)
        if 'deltawidth' in kwargs:
            deltawidth = round(kwargs['deltawidth']/0.005)*0.005 # ALTA3900 grid size for the 0.6µm mask process
            e.info.mzwidth1,e.info.mzwidth2 = width,width+deltawidth
            e.addondoublesbend(sbendlength,pitch,width,x=x0,y=y0,poshalf=True,splitradius=splitradius)
            # e.addontaper(1000,width+deltawidth).addonchannel(dx-2000).addontaper(1000,width); e.info.mzimbalanceeffectivelength = dx-1000 # full length
            e.addonchannel(dx/2-6000/2).addontaper(1000,width+deltawidth).addonchannel(4000).addontaper(1000,width).addonchannel(dx/2-6000/2) # 6000 length (5000 effective)
            e.info.mzimbalanceeffectivelength = 5000
        elif 'deltalength' in kwargs:
            deltalength = kwargs['deltalength']
            def biasfreelength(lambdainum,index): # length required for pi/2 phase shift, deltalength = 0.1814 for pi/2 at 1.556um (n=2.145)
                return lambdainum/4/index
            def displacement(dl,s,mzx):
                # mzx = 25000, d =  5, s = 16, dl = 0.157899935, ROC = ~100mm
                # mzx = 25000, d = 20, s =  2, dl = 0.039475124
                # dl = 0.039475124*(d/20)**2*(12500/ds)**2 = 0.039475124*(d/20)**2*(s*12500/mzx)**2
                return 20*sqrt(dl/0.039475124)/(s*12500/mzx)
            sbendcount = 16 # must be even # 2x bendcount = 4x pathlength = 0.25x ROC
            dy = -5 # 2x displacement = 4x pathlength = 0.5x ROC
            #dy = -displacement(0.1814,sbendcount,mzlength)
            dy = -displacement(deltalength,sbendcount,mzlength)
            xs,ys,roc,pathlength = sbend(dx/sbendcount,dy)
            #print('roc,pathlength*sbendcount,mzarmrocinmm',roc,pathlength*sbendcount,abs(roc/1000))
            e.info.mzdeltalength,e.info.mzsbendcount,e.info.mzydisplacement,e.info.mzarmrocinmm = deltalength,sbendcount,dy,abs(roc/1000)
            #print('mzdeltalength,mzsbendcount,mzydisplacement,mzarmrocinmm',e.info.mzdeltalength,e.info.mzsbendcount,e.info.mzydisplacement,e.info.mzarmrocinmm)
            e.addondoublesbend(sbendlength,pitch,width,x=x0,y=y0,poshalf=True,splitradius=splitradius)
            ee = Element('segment',parent=self,layer='MASK')
            ee.width,ee.x,ee.y = e.width,e.x,e.y
            ee.x1 = e.x
            for n in range(sbendcount//2):
                e.addonsbend(dx/sbendcount,dy,note=0).addonsbend(dx/sbendcount,-dy,note=(1==n))
                ee.addonsbend(dx/sbendcount,dy,note=0).addonsbend(dx/sbendcount,-dy,note=(1==n))
            ee.x2 = e.x
            e.addnote(ee.x1/2+ee.x2/2,e.y,mzul='%.3f ΔL, %d S-bends'%(deltalength,sbendcount))
            ee.addnote(ee.x1/2+ee.x2/2,e.y,mzul='%.3f ΔL, %d S-bends'%(deltalength,sbendcount))
            self.seg = ee # ee = copy of wiggly segment to incorporate into a channel waveguide later
        elif 'deltabragg' in kwargs:
            deltabragg = kwargs['deltabragg']
            def openfraction(lambdainum,mzlength,deltaindex): # openfraction = 1-braggdutycycle = 'deltabragg' required for pi/2 phase shift 
                return lambdainum/4/mzlength/deltaindex  # nwg = 2.145, nb = 2.137 at 1.556um
            # openfraction(1.556,25000,2.145-2.137) = 0.001945  # nwg = 2.145, nb = 2.137 at 1.556um
            openx = 2 # target open segment length
            braggcount = 1+int(mzlength*deltabragg/openx)
            braggperiod,braggdc = mzlength/braggcount,1-deltabragg
            e.info.mzbraggopenfraction,e.info.mzbraggopenlength,e.info.mzbraggperiod,e.info.mzbraggdc,e.info.mzbraggopensegmentlength = deltabragg,deltabragg*mzlength,braggperiod,1-deltabragg,braggperiod*deltabragg
            e.addondoublesbend(sbendlength,pitch,width,x=x0,y=y0,poshalf=True,splitradius=splitradius)
            ee = Element('segment',parent=self,layer='MASK')
            ee.width,ee.x,ee.y = e.width,e.x,e.y
            ee.x1 = e.x
            e.addonbragg(dx,period=braggperiod*(1 if not dev else 1/periodmag),dc=1-deltabragg*(1 if not dev else periodmag)) # for cartoon version, bragg gap is magnified, not braggperiod
            ee.addonbragg(dx,period=braggperiod*(1 if not dev else 1/periodmag),dc=1-deltabragg*(1 if not dev else periodmag))
            ee.x2 = e.x
            e.addnote(ee.x1/2+ee.x2/2,e.y,mzub='%.1f period, %.2f gap (%.2f%%), %d gaps'%(braggperiod,e.info.mzbraggopensegmentlength,100*deltabragg,braggcount))
            ee.addnote(ee.x1/2+ee.x2/2,e.y,mzub='%.1f period, %.2f gap (%.2f%%), %d gaps'%(braggperiod,e.info.mzbraggopensegmentlength,100*deltabragg,braggcount))
            self.seg = ee # ee = copy of wiggly segment to incorporate into a channel waveguide later
            #print('mzbraggopenfraction,mzbraggopenlength,mzbraggperiod,mzbraggdc,mzbraggopensegmentlength',e.info.mzbraggopenfraction,e.info.mzbraggopenlength,e.info.mzbraggperiod,e.info.mzbraggdc,e.info.mzbraggopensegmentlength)
        else:
            e.addondoublesbend(sbendlength,pitch,width,x=x0,y=y0,poshalf=True,splitradius=splitradius).addonchannel(dx)
        e.addondoublesbend(sbendlength,pitch,width,y=y0,outputside=True,splitradius=splitradius) # final self.info.roc is set here
        e.y = y0
        e.addontaper(taperx,inwidth=2*width,outwidth=width).addonchannel(cx-xoffset,width=width)
        if mf:
            e.addontaper(taperx,outwidth=mf).addonchannel(modefilterx)
        e.info.mzpitch = pitch
        e.info.mzlength = dx
        self.parent.y = self.y = e.y + pitch/2 # self.y and self.parent.y will line up with second arm (for placement of next guide), e.y will line up with output
        self.ey = e.y # hack to get e.y for centering electrode
        return self
    def addmfsplitterdcmachzehnder(self,width,pitch,mzlength=None,sbendlength=None,splitradius=0,xoffset=0,y=None,inx=None,split=None,Lc=None,outpitch=None):
        dx = self.finddefaultinfo('chiplength')
        # modefilterx = self.finddefaultinfo('modefilterlength')
        taperx = self.finddefaultinfo('taperlength')
        sbendlength = sbendlength if sbendlength is not None else self.finddefaultinfo('sbendlength')
        e = Guide('machzehnder-guide',parent=self)
        e.y = e.y+pitch/2 if y is None else y # self.y lines up with first arm, e.y will line up with input
        e.info.guidewidth = width
        e.addonchannel(inx,width=width).addontaper(taperx,outwidth=2*width)
        x0,y0 = e.x,e.y
        self.info.mzx0,self.info.mzy0 = e.x+sbendlength, e.y
        e.addondoublesbend(sbendlength,pitch,width,x=x0,y=y0,neghalf=True,splitradius=splitradius).addonchannel(mzlength)
        e.addondoublesbend(sbendlength,pitch,width,x=x0,y=y0,poshalf=True,splitradius=splitradius).addonchannel(mzlength)
        xx = dx-(inx+taperx+sbendlength+mzlength) # xx = length of remaining section
        e.addondcoutput(xx,pitch,width,y=y0,split=split,Lc=Lc,sbendx=sbendlength,outpitch=outpitch,outx=inx)
        e.info.mzpitch = pitch
        e.info.mzlength = dx
        self.parent.y = self.y = e.y + pitch/2 # self.y and self.parent.y will line up with second arm (for placement of next guide), e.y will line up with output
        self.ey = e.y # hack to get e.y for centering electrode
        self.info.outsplitx = x0 + sbendlength + mzlength + sbendlength # x start of directional coupler straight section
        self.info.outsplity = y0                                        # y start of directional coupler straight section
        return self
    def addmfmachzehnder(self,width,pitch,mzlength=0,mf=0,sbendlength=None,splitradius=0,xoffset=0,y=None):
        # for an unbalanced mz, supply either deltawidth, deltalength, or deltabragg # first arm is normal, second arm is modified
        dx = self.finddefaultinfo('chiplength')
        taperx = self.finddefaultinfo('taperlength')
        sbendlength = sbendlength if sbendlength is not None else self.finddefaultinfo('sbendlength')
        e = Guide('machzehnder-guide',parent=self)
        e.y = e.y+pitch/2 if y is None else y # self.y lines up with first arm, e.y will line up with input
        e.info.guidewidth = width
        if mf and not mf==width: e.info.inputmodefilter,e.info.outputmodefilter = mf,mf
        mf = mf if mf else width
        mfx = 0.5*(self.finddefaultinfo('chiplength') - (mzlength + 2*taperx + 2*sbendlength))
        # cx = modefilterx
        # dx,dy = dx - (2*modefilterx*(mf>0) + 2*taperx*(mf>0) + 2*cx + 2*taperx + 2*sbendlength), width/2.
        # dx,dy = dx - (2*modefilterx*(mf>0) + 2*cx + 2*taperx + 2*sbendlength), width/2.
        # if mzlength: cx,dx = cx+(dx-mzlength)/2.,mzlength
        # if mf: e.addonchannel(modefilterx,width=mf).addontaper(taperx,outwidth=width)
        e.addonchannel(mfx+xoffset,width=mf).addontaper(taperx,outwidth=2*width)
        x0,y0 = e.x,e.y
        e.addondoublesbend(sbendlength,pitch,width,x=x0,y=y0,neghalf=True,splitradius=splitradius).addonchannel(mzlength)
        e.addondoublesbend(sbendlength,pitch,width,x=x0,y=y0,poshalf=True,splitradius=splitradius).addonchannel(mzlength)
        e.addondoublesbend(sbendlength,pitch,width,y=y0,outputside=True,splitradius=splitradius) # final self.info.roc is set here
        e.y = y0
        e.addontaper(taperx,inwidth=2*width,outwidth=mf).addonchannel(mfx-xoffset,width=mf)
        # if mf: e.addontaper(taperx,outwidth=mf).addonchannel(modefilterx)
        e.info.mzpitch = pitch
        e.info.mzlength = mzlength
        self.parent.y = self.y = e.y + pitch/2 # self.y and self.parent.y will line up with second arm (for placement of next guide), e.y will line up with output
        self.ey = e.y # hack to get e.y for centering electrode
        self.info.mzlength = mzlength
        return self
    def addmetriconblockedchip(self,dx=None,dy=None,x0=0,y0=0,filldicegap=False):
        dx = dx if dx is not None else self.finddefaultinfo('chiplength')
        dy = dy if dy is not None else self.finddefaultinfo('chipwidth')
        if filldicegap:
            dg = self.finddefaultinfo('dicegap')
            x0,y0,dx,dy = x0,y0-dg/2,dx,dy+dg
        Element('witness',parent=self).addrect(x0,y0,dx,dy)
        # self.addnote(dx/2,dy/2,dy=100,note=f'↕ {dy} µm')
        return self
    def addwaferflatwindowchip(self,yflat=700,filldicegap=False):
        x0,y0,dx,dy = 0,0,self.finddefaultinfo('chiplength'),self.finddefaultinfo('chipwidth')
        print('wafer flat window size',yflat)
        assert 0<yflat<dy, f'yflat:{yflat}'
        # Element('witness',parent=self).addrect(0,0,dx,0.25*dy).addrect(0,0.75*dy,dx,0.25*dy)
        # for i in range(6): Element('witness',parent=self).addrect(0,200*i,dx,100)
        if filldicegap:
            dg = self.finddefaultinfo('dicegap')
            x0,y0,dx,yflat = x0,y0-dg/2,dx,yflat+dg/2
        Element('window',parent=self).addrect(x0,y0,dx,yflat) # wafer flat at (yflat), lined up to bottom of window
        self.addnote(dx/2,yflat/2,dy=100,note=f'↕ {yflat} µm')
        return self
    def addtestchip(self):
        # label = ' '+self.info.chipid
        # self.addtext(self.finddefaultinfo('maskname')+label,x=2000,y=200)
        # self.addtext(self.finddefaultinfo('maskname')+label,x=self.finddefaultinfo('chiplength')/2,y=200)
        self.info.guidespacing = 30
        self.info.groupspacing = 60
        # self.adddiceguides()
        self.y += 500
        g1 = Group(parent=self).addguidelabels(dy=-50)
        g1.addchannel(5)
        for wi in [5,6,7,8]:
            g1.addchannel(wi)
        return self
    def addwaveguidechip(self):
        self.info.guidespacing = 100
        self.info.groupspacing = 200
        self.adddiceguides()
        self.addmetric(x=100,y=400,dx=25,dy=25,nx=2,ny=6)
        self.addtext('chip:'+self.info.chipid,x=500,y=400,scale=2)
        self.y += 500
        g1 = Group(parent=self).addguidelabels(dy=-50)
        for wi in [2,3,4]:
            g1.addchannel(wi)
        g2 = Group(parent=self).addguidelabels(dy=-50)
        for wi in [2,3,4]:
            g2.addmodefilter(wi,inwidth=6,modefilterx=1000,taperx=2500)
        g3 = Group(parent=self,showguidespacing=True,showgroupspacing=True).addguidelabels(dy=-50)
        for wi in [4,6,8]:
            g3.addmodefilter(wi,inwidth=2,outwidth=2,modefilterx=1000,taperx=2500)
        self.addscalenote(poling=False)
        return self
    def addtestelectrodewaveguidechip(self):
        label = ' '+self.info.chipid
        self.addtext(self.finddefaultinfo('maskname')+label,x=2000,y=200)
        self.addtext(self.finddefaultinfo('maskname')+label,x=self.finddefaultinfo('chiplength')/2,y=200)
        self.info.guidespacing = 50
        self.info.groupspacing = 100
        self.adddiceguides()
        self.y += 500
        g1 = Group(parent=self).addguidelabels(dy=-50)
        g1.addchannel(5)
        for wi in [5,6,7,8]:
            g1.addchannel(wi)

        # elabel = self.finddefaultinfo('electrodemaskname')+label
        L,dx = 10000,40000
        self.addcpwelectrode(L,10,20,100,200,dx,1000,xin=2000,xout=2000)

        return self
    def addcarrots(self,length,gap,xoffset=0,size=200):
        sx,sy,stext = size,size/4,200
        dx,dy = self.finddefaultinfo('chiplength'),self.finddefaultinfo('chipwidth')
        g = Element('electrode',layer='ELECTRODE',parent=self)
        x,y = self.x+self.finddefaultinfo('chiplength')/2, self.ey if hasattr(self,'ey') else self.y
        cx,cy = np.array([0,sx,sx,0,0]),np.array([0,0,gap/2,sy,0])
        g.addxypoly( cx+length/2+x+xoffset,  cy+y+gap/2)
        g.addxypoly(-cx-length/2+x+xoffset,  cy+y+gap/2)
        g.addxypoly( cx+length/2+x+xoffset, -cy+y-gap/2)
        g.addxypoly(-cx-length/2+x+xoffset, -cy+y-gap/2)
        return self
    def addcarrotlabels(self,length,gap,xoffset=0,size=200):
        sx,sy,stext = size,0,200
        dx,dy = self.finddefaultinfo('chiplength'),self.finddefaultinfo('chipwidth')
        g = Element('electrode',layer='ELECTRODE',parent=self)
        x,y = self.x+self.finddefaultinfo('chiplength')/2, self.ey if hasattr(self,'ey') else self.y
        g.addtext('I',length/2+x+xoffset,y-gap/2-sy,fitx=sx,fity=stext,center=True,scale=0).addtext('O',-length/2+x+xoffset-sx,y-gap/2-sy,fitx=sx,fity=stext,center=True,scale=0)
        g.addtext('I',length/2+x+xoffset,y+gap/2+sy+stext,fitx=sx,fity=stext,center=True,scale=0).addtext('O',-length/2+x+xoffset-sx,y+gap/2+sy+stext,fitx=sx,fity=stext,center=True,scale=0)
        return self
    def addcentertapelectrode(self,gap,label=None,length=40000,xoffset=0,edgey=200,ingap=None,metric=False): # edgey = gap distance between metal and edge of chip
        hot,gnd = 50,50
        inputy = 100 # length of straight part before tapering out
        inhot,ingnd = hot,gnd
        ingap = ingap if ingap is not None else inhot
        outgap,outhot,outgnd = 500,500,500
        label = label if label is not None else f'{hot}-{gap}-{gnd}'
        dx,dy = self.finddefaultinfo('chiplength'),self.finddefaultinfo('chipwidth')
        g = Element('electrode',layer='ELECTRODE',parent=self)
        x,y = self.x+self.finddefaultinfo('chiplength')/2, self.ey if hasattr(self,'ey') else self.y
        res = self.finddefaultinfo('minfeature',fail=False) or self.finddefaultinfo('maskres')
        g.addtext(label,x=2000+x+xoffset,y=y+500).addmetric(x=2000+x+xoffset-107,y=y+500,dx=25,dy=gap,nx=2,ny=6).addmetric(x=2000+x+xoffset-107*2,y=y+500,dx=25,dy=res,nx=2,ny=6)
        g.info.gap,g.info.hot,g.info.gnd = gap,hot,gnd
        g.info.length = length
        def chot(): # hot, center T-shape electrode: ┬
            a,b,c,d,e = inhot/2,gap/2,inputy,length/2,-gap/2-hot
            j,m = outhot/2,dy-y-edgey
            tx,ty = [0,j,a,a,a,d,d,0,-d,-d,-a,-a,-a,-j,0],[m,m,c,c,-b,-b,e,e,e,-b,-b,c,c,m,m]
            c0 = [(cx+x,cy+y) for cx,cy in zip(tx,ty)]
            return roundoffcorners(c0)
        c0 = chot()
        def cgnds(): # gnd, two outer L-shape electrodes: ┐┌
            a,b,c,d,e,f = inhot/2+ingap,inhot/2+ingap+ingnd,inputy,length/2,gap/2+gnd,gap/2
            j,k,m = outhot/2+outgap,outhot/2+outgap+outgnd,dy-y-edgey
            c1 = [(cx+x,cy+y) for cx,cy in zip([j,k,b,b,b,d,d,a,a,j],[m,m,c,c,e,e,f,f,c,m])]
            c2 = [(cx+x,cy+y) for cx,cy in zip([-j,-k,-b,-b,-b,-d,-d,-a,-a,-j],[m,m,c,c,e,e,f,f,c,m])]
            return roundoffcorners(c1),roundoffcorners(c2)
        c1,c2 = cgnds()
        c0,c1,c2 = [(cx+xoffset,cy) for cx,cy in c0],[(cx+xoffset,cy) for cx,cy in c1],[(cx+xoffset,cy) for cx,cy in c2]
        g.addpoly(c0).addpoly(c1).addpoly(c2)
        p = (x,y-gap/2-hot/2); g.addnote(p[0]+1500+xoffset,p[1],ewidth=hot)
        p = (x,y+gap/2+gnd/2); g.addnote(p[0]+1500+xoffset,p[1],ewidth=gnd)
        p = (x,y); g.addnote(p[0]+1000+xoffset,p[1],egap=gap)
        p = np.array(c0[-2])/2+np.array(c0[1])/2; w = (np.array(c0[1])-np.array(c0[-2]))[0]; g.addnote(p[0],p[1],outwidth=w)
        p = np.array(c1[1])/2+np.array(c1[0])/2; w = (np.array(c1[1])-np.array(c1[0]))[0]; g.addnote(p[0],p[1],outwidth=w)
        p = np.array(c2[0])/2+np.array(c2[1])/2; w = (np.array(c2[0])-np.array(c2[1]))[0]; g.addnote(p[0],p[1],outwidth=w)
        p = np.array(c1[0])/2+np.array(c0[1])/2; w = (np.array(c1[0])-np.array(c0[1]))[0]; g.addnote(p[0],p[1],outgap=w)
        p = np.array(c2[0])/2+np.array(c0[-2])/2; w = -(np.array(c2[0])-np.array(c0[-2]))[0]; g.addnote(p[0],p[1],outgap=w)
        p = np.array(c0[-3])/2+np.array(c0[2])/2; w = (np.array(c0[2])-np.array(c0[-3]))[0]; g.addnote(p[0],p[1],inwidth=w)
        p = np.array(c1[2])/2+np.array(c1[-2])/2; w = (np.array(c1[2])-np.array(c1[-2]))[0]; g.addnote(p[0],p[1],inwidth=w)
        p = np.array(c2[-2])/2+np.array(c2[2])/2; w = (np.array(c2[-2])-np.array(c2[2]))[0]; g.addnote(p[0],p[1],inwidth=w)
        p = np.array(c1[-2])/2+np.array(c0[2])/2; w = (np.array(c1[-2])-np.array(c0[2]))[0]; g.addnote(p[0],p[1],ingap=w)
        p = np.array(c2[-2])/2+np.array(c0[-3])/2; w = -(np.array(c2[-2])-np.array(c0[-3]))[0]; g.addnote(p[0],p[1],ingap=w)
        if metric:
            for xi in [x-length/2+2000,x-2000,x+length/2-2000]:
                g.addoversizemetrics(5,xi,y+500,i0=11,dx=10,nx=8,ny=6,w0=None,dw=1,textscale=1,layer='ELECTRODE')
        return self
    def addphidlcpwelectrode(self,L,gap,hot,fangap,fanhot,dx,dy,xin,xout,id=None,metric=None,label=None):
        g = Element('electrode',layer='MASK',parent=self)
        import phidl,phidls
        text = f"{self.info.chipid if id is None else id} {L/1000:g}mm {gap}-{hot}-{gap}"
        label = label if label is not None else text
        D = phidls.horseshoecpw(L=L,gap=gap,hot=hot,fangap=fangap,fanhot=fanhot,chipx=dx-xin-xout,chipy=dy,label=label,mirror=True)
        D.movex(xin)
        polys = phidls.layerpolygons(D,layers=[0],closed=True)
        g.addpolys(polys)
        g.showorientations()

        m = Element(layer='TEMP')
        w0s = [8,9,10,11,12,13,14,15,16,17,18,19]
        m.addoversizemetrics(width=10,xmetric=dx/2,ymetric=1000,i0=3,dx=50,nx=8,ny=6,w0=w0s,textscale=3,relative=True)
        m.convertrectstopolys()
        m.showorientations()
        m.orient(-1)
        m.showorientations()
        g.insetelement(m)

        for k,s,(x,y) in D.notes:
            g.addnote(x=x+xin,y=y,dy=20,**{k:s})
        return self
    def addcpselectrode(self,L,hot,gap,gnd,dx,dy,label=1):
        g = Element('cps',layer='MASK',parent=self)
        if -1!=label:
            g.addrect(0,0,L,gnd)
            g.addrect(0,gnd+gap,L,hot)
        if 1==label:
            g.addtext(f'{hot}-{gap}-{gnd}',x=50,y=-50,scale=0.5)
            if L>2000:
                g.addtext(f'{hot}-{gap}-{gnd}',x=L-500+50,y=-50,scale=0.5)
        if -1==label:
            def polyrect(x0,y0,dx,dy):
                return [(x0,y0),(x0+dx,y0),(x0+dx,y0+dy),(x0,y0+dy),(x0,y0)]
            poly = polyrect(0,0,L,gnd)
            g.addinsettext([poly[::-1]],f'{hot}-{gap}-{gnd}',x0=50,y0=100,scale=0.5)
            g.addpoly(polyrect(0,gnd+gap,L,hot))
        g.translate((dx-L)/2,(dy-(hot+gap+gnd))/2)
        self.addnotesframe(margin=(0,0),size=(dx,dy))
        # print('L,hot,gap,gnd,dx,dy',L,hot,gap,gnd,dx,dy)
        return self
    def addcpwelectrode(self,ghg,y0=500,L=1000,label='',scaleifdev=False,diceinset=25,xghg=None):
        dx,dy = self.finddefaultinfo('chiplength'),self.finddefaultinfo('chipwidth')
        r = dy-y0
        g = Element('electrode',layer='ELECTRODE',parent=self)
        from waves import Wave
        gapsvshots = {
            # 50Ω designs from mglnapemask(gridsize=4000,gridnum=360000,iters=3,xghg=900)
            '23-10-23': Wave([23,35.9684,50.3551,65.6613,81.6565,114.677,148.62,183.064,217.792,252.655,287.57,322.458,357.273,390.177],[10,15,20,25,30,40,50,60,70,80,90,100,110,119.481]),
            '51-20-51': Wave([51,66.516,82.7314,116.2,150.595,185.499,220.679,255.999,291.365,326.699,361.96,390.659],[20,25,30,40,50,60,70,80,90,100,110,118.169]),
            '16-10-16': Wave([16,24.7612,35.1649,46.8643,59.5648,87.1243,116.633,147.452,179.044,211.188,243.706,276.472,309.376,342.345,375.326,383.494],[10,15,20,25,30,40,50,60,70,80,90,100,110,120,130,132.48]),
            '36-20-36': Wave([36,48.0163,61.0539,89.3424,119.615,151.223,183.614,216.565,249.89,283.459,317.161,350.925,384.688,384.95],[20,25,30,40,50,60,70,80,90,100,110,120,130,130.078]),
            }
        gvh = gapsvshots[ghg]
        if xghg is not None:
            ghg = Wave(gvh.x,2*gvh.y+gvh.x)
            fanhot = ghg(xghg,extrapolate='log')
            fangap = gvh(fanhot) # print('fanhot',fanhot,'fangap',fangap)
            gvh = gvh(0,fanhot) # print('gvh',gvh)
            gvh = Wave(list(gvh.y)+[fangap],list(gvh.x)+[fanhot]) # print('gvh',gvh)
        hot,gap = int(round(gvh.x[0])), int(round(gvh[0]))
        fhot,fgap = int(round(gvh.x[-1])), int(round(gvh[-1]))
        dg,fdg = 0.5*hot+0.5*gap, 0.5*fhot+0.5*fgap
        cc = cpwelectrode(L=L,r=r,dx=L+2*r+4000,dy=dy,gvhwave=gvh,mirrory=True,diceinset=diceinset)
        # g.addpolys(cc)
        cc = [reorient(c,0) for c in cc]
        g.addinsettext(cc,label,x0=0.5*L,y0=r-200,scale=2,justify='center',scaleifdev=scaleifdev)
        g.insetshapes(advrlogo(x=-3400,y=-100,scale=3),sort=False)
        for x in [L*a/8 for a in [1,3,5,7]]:
            g.insetshapes(checkerboardmetric(dx=2*hot,dy=hot,nx=4,ny=8,x=x,y=r-200,centered=True))
        for x in (-r,L+r):
            g.addnote(x,r-50,inwidth=fhot).addnote(x-fdg,r-50,ingap=fgap).addnote(x+fdg,r-50,ingap2=fgap)
        for x in (100,):
            g.addnote(x,0,ewidth=hot).addnote(x,-dg,egap=gap).addnote(x,dg,egap=gap)
        for x in (L-100,):
            g.addnote(x,0,ewidth2=hot).addnote(x,-dg,egap2=gap).addnote(x,dg,egap2=gap)
        g.translate(0.5*dx-0.5*L,dy-r)
    def addcpwelectrode2(self,ghg,y0=500,L=1000,label='',scaleifdev=False,diceinset=25,fanhot=None,fantaper=None,xoffset=0):
        dx,dy = self.finddefaultinfo('chiplength'),self.finddefaultinfo('chipwidth')
        r = dy-y0
        g = Element('electrode',layer='ELECTRODE',parent=self)
        from waves import Wave
        gapsvshots = {
            # 50Ω designs from mglnapemask(gridsize=4000,gridnum=360000,iters=3,xghg=900)
            '23-10-23': Wave([23,35.9684,50.3551,65.6613,81.6565,114.677,148.62,183.064,217.792,252.655,287.57,322.458,357.273,390.177],[10,15,20,25,30,40,50,60,70,80,90,100,110,119.481]),
            '51-20-51': Wave([51,66.516,82.7314,116.2,150.595,185.499,220.679,255.999,291.365,326.699,361.96,390.659],[20,25,30,40,50,60,70,80,90,100,110,118.169]),
            '16-10-16': Wave([16,24.7612,35.1649,46.8643,59.5648,87.1243,116.633,147.452,179.044,211.188,243.706,276.472,309.376,342.345,375.326,383.494],[10,15,20,25,30,40,50,60,70,80,90,100,110,120,130,132.48]),
            '36-20-36': Wave([36,48.0163,61.0539,89.3424,119.615,151.223,183.614,216.565,249.89,283.459,317.161,350.925,384.688,384.95],[20,25,30,40,50,60,70,80,90,100,110,120,130,130.078]),
            }
        gvh = gapsvshots[ghg]
        assert fanhot<gvh.x.max()
        fangap = gvh(fanhot) # print('fanhot',fanhot,'fangap',fangap)
        gvh = gvh(0,fanhot) # print('gvh',gvh)
        gvh = Wave(list(gvh.y)+[fangap],list(gvh.x)+[fanhot])
        cc,notes = cpwelectrode2(L=L,y0=r,y1=150,y2=fantaper,chipx=L+2*r+4000,chipy=dy,gvhwave=gvh,mirrory=True,diceinset=diceinset)
        cc = [reorient(c,0) for c in cc]
        # g.addpolys(cc)
        # g.addinsettext(cc,label,x0=0.5*L,y0=r-200,scale=2,justify='center',scaleifdev=scaleifdev)
        g.addinsettext(cc,label,x0=0.5*L-420,y0=r-180,scale=1,justify='left',scaleifdev=scaleifdev)
        g.insetshapes(advrlogo(x=0.5*L-1095,y=r-110,scale=3),sort=False)
        hot,gap = int(round(gvh.x[0])), int(round(gvh[0]))
        for x in [L*a/8 for a in [1,3,5,7]]:
            g.insetshapes(checkerboardmetric(dx=2*hot,dy=hot,nx=4,ny=8,x=x,y=r-250,centered=True))
        for args in notes:
            g.addnote(**args)
        g.translate(0.5*dx-0.5*L+xoffset,dy-r)
    def addgrating(self,barstarts,barends,y0,gy):
        g = Element('grating',layer='MASK')
        for x0,x1 in zip(barstarts,barends):
            g.addrect( x0,y0, x1-x0,gy )
        g.translate(self.x,self.y)
        self.addelem(g)
        return self
    def pickle(self,name=None):
        name = name if name is not None else self.name
        with open(name+'.pickle','wb') as f:
            pickle.dump(self,f,-1)
    @staticmethod
    def getpickle(name):
        with open('pickle/'+name+'.pickle', 'rb') as f:
            return pickle.load(f)
    def addboeingpmelectrode2(self,length,style,hot=305,gap=7,gap2=307,dicegap=209,translate=(0,0)):
        assert style=='H'
        def widthnotes(self,xys,dim=1): # dim=0 for horizontal widths, dim=1 for vertical widths
            for p,pp in zip(xys[0::2],xys[1::2]):
                xy,dz = 0.5*(np.array(p)+np.array(pp)),abs(p[dim]-pp[dim])
                self.addnote(*xy,ewidth0=dz)
            for p,pp in zip(xys[1::2],xys[2::2]):
                xy,dz = 0.5*(np.array(p)+np.array(pp)),abs(p[dim]-pp[dim])
                self.addnote(*xy,egap0=dz)
        def hstrips(self,xys,L):
            assert 0==len(xys)%2, 'need two points to define each hstrip'
            def strip(x0,y0,x1,y1):
                return np.array([(x0,y0),(x0+L,y0),(x1+L,y1),(x1,y1),(x0,y0)])
            self.addpolys([strip(*p,*pp) for p,pp in zip(xys[0::2],xys[1::2])])
            widthnotes(self,xys,dim=1)
            widthnotes(self,xys+(L,0),dim=1)
        def bendstrips(self,xy0s,xy1s,d0=(0,0),d1=(0,0),squaredoff=False,debug=False): # d0,d1 = straight section delta at start and end of bend
            assert len(xy0s)==len(xy1s)
            def arc(p,pp,squaredx=False,squaredy=False):
                (x0,y0),(x1,y1) = np.array(p)+d0,np.array(pp)+d1
                # if squaredx: return [p]+[(x1,y0)]+[pp]
                if squaredy: return [p]+[(x0,y1)]+[pp]
                θs = np.linspace(0,pi/2,201)
                return [p]+[(x0+(x1-x0)*(1-cos(θ)),y0+(y1-y0)*sin(θ)) for θ in θs]+[pp]
            arcs = [arc(p,pp,squaredy=(i==0 or i==len(xy0s)-1 if squaredoff else False))  for i,(p,pp) in enumerate(zip(xy0s,xy1s))]
            self.addpolys([a0+a1[::-1]+a0[:1] for a0,a1 in zip(arcs[0::2],arcs[1::2])])
            widthnotes(self,xy0s,dim=0)
            if debug:
                for p in xy0s: self.addcenteredrect(*p,10,10)
                for pp in xy1s: self.addcenteredrect(*pp,5,5)
        assert dicegap==209 # 1004.5==900+dicegap/2
        ingap,inhot,ingap2 = 55,600,160
        outgap,outhot,outgap2 = ingap,inhot,ingap2
        x0,y0 = (3750,1004.5) # position of start of electrode straight section
        dy = 870 # ±vertical bounds of metal, 1800 wide minus 30 margin
        dxin,dxout = 325,765+200 # center of input,output fanouts relative to start and end of straight section (200 offset of output fanout relative to PMG electrode)
        bxin,bxout = 1137.5+500,1425 # horizontal bounds of metal relative to center of input,output fanouts (500 loonger input than PMG)
        ypad = 400 # fanout straight length
        yframe = 765.5
        ybuffer = 40 # width of waveguide protecting buffer
        xy0s = (x0-dxin,y0+dy) + np.array([(dxin,0),(0.5*ingap,0),(-0.5*ingap,0),(-0.5*ingap-inhot,0),(-0.5*ingap-inhot-ingap2,0),(-bxin,0)])
        xy1s = (x0,y0) + np.array([(0,dy),(0,0.5*gap),(0,-0.5*gap),(0,-0.5*gap-hot),(0,-0.5*gap-hot-gap2),(0,-dy)])
        xy2s = (x0+length+dxout,y0-dy) + np.array([(outhot+0.5*outgap,0),(0.5*outgap,0),(-0.5*outgap,0),(-0.5*outgap-outhot,0),(-0.5*outgap-outhot-outgap2,0),(-dxout,0)])
        g = Element('electrode',layer='ELECTRODE',parent=self)
        bendstrips(g,xy0s,xy1s,d0=(0,-ypad),squaredoff=True)
        hstrips(g,xy1s,length)
        bendstrips(g,xy2s,xy1s+(length,0),d0=(0,ypad-100))
        g.addrect(x0+length+dxout+outhot+0.5*outgap-5,y0-dy,bxout-outhot-0.5*outgap+5,ypad)
        g.addrect(x0+length,y0+dy-yframe,bxout+dxout,yframe)
        bx0,by0,bdx,bdy = g.boundingbox()
        g.addsubmountmetric(x=bx0+bdx-500,y=by0+600)
        g.translate(*translate)
        b = Element('buffer',layer='BUFFER',parent=self)
        b.addrect(x0-dxin-bxin-ybuffer,y0-0.5*ybuffer,dxin+bxin+ybuffer,ybuffer)
        b.addrect(x0+length,y0-0.5*ybuffer,dxout+0.5*outgap+outhot,ybuffer)
        b.translate(*translate)
        return self

    def addboeingpmvivaldi(self,length=40000,style='I',gap=7,dicegap=209,translate=(0,0),mirrored=False):
        assert style in 'IJK'
        assert dicegap==209 # 1004.5==900+dicegap/2
        x0,y0 = (6000,1004.5) # position of start of electrode straight section
        dy = 2500-30 # ±vertical bounds of metal, 1800 wide minus 30 margin
        dw = 500-0.5*gap if style in 'JK' else dy
        xtaper,xexp = 7350,4895
        (xpad,ypad),xease = (500,200),100
        dx = self.finddefaultinfo('chiplength')
        length = 0.5*dx-7000 if mirrored else length
        assert xtaper<length, 'no room for vivaldi linear taper section'
        def vivaldi(self,length,gap,dw,dy,xexp,xtaper,x0,y0,style):
            def f(x):
                return (0.147)*0.25*exp(0.123*(1/0.147)*np.where(0<x,x,0)) # from Anguel
            xs = np.linspace(-xexp,0,xexp+1)
            ys = 1000*f(abs(xs)/1000)
            v0s = list(zip(xs,ys))
            v1s = [v0s[-1],(xtaper,0.5*gap)]
            v2s = [(xtaper,0.5*gap),(length-xpad-xease,0.5*gap)]
            v3s = [(length-xpad-xease,0.5*gap),(length-xpad,0.5*gap)]
            v4s = [(length-xpad,0.5*gap),(length,0.5*gap)]
            yy0,yy1,yy2 = v0s[-1][1],v1s[-1][1],v2s[-1][1]
            def uppery(x,y):
                # return dy if style=='I' or (style=='K' and x<-3000) else min(y+dw,dy) # version 1.1
                # return dy if style=='I' or (style=='K' and x<-3000) else min(y+dw,dy) if 0<x else min(1000*f(-x/1000)+dw,dy) # same as version 1.1
                return dy if style=='I' or (style=='K' and x<-3000) else min(y+dw,dy) if 0<x else min(1000*f(-x/800)+dw,dy) # version 1.2 # width at end is 1086
            def uppertrace(vs):
                return [(x,uppery(x,y)) for x,y in vs[::-1]]
            cs = [ vs + uppertrace(vs) + vs[:1] for vs in [v0s,v1s,v2s,v3s,v4s] ]
            ccs = [[(x+x0,-y+y0) for x,y in c] for c in cs]
            cs = [[(x+x0,y+y0) for x,y in c] for c in cs]
            self.addpolys(cs)
            self.addpolys(ccs)
            self.addnote(x0-xexp,y0,egap0=f"{2*v0s[0][1]:.3f}")
            self.addnote(x0-xexp,y0-0.5*v0s[0][1]-0.5*dy,ewidth0=f"{dy-v0s[0][1]:.3f}")
            self.addnote(x0-xexp,y0+0.5*v0s[0][1]+0.5*dy,ewidth0=f"{dy-v0s[0][1]:.3f}")
            self.addnote(x0,y0+0.5*uppery(0,yy0),ewidth0=uppery(0,yy0)-yy0)
            self.addnote(x0,y0-0.5*uppery(0,yy0),ewidth0=uppery(0,yy0)-yy0)
            self.addnote(x0,y0,egap0=2*yy0)
            self.addnote(x0+xtaper,y0+0.5*uppery(xtaper,yy1),ewidth0=uppery(xtaper,yy1)-yy1)
            self.addnote(x0+xtaper,y0-0.5*uppery(xtaper,yy1),ewidth0=uppery(xtaper,yy1)-yy1)
            self.addnote(x0+xtaper,y0,egap0=gap)
            self.addnote(x0+length,y0+0.5*uppery(length,yy2),ewidth0=uppery(length,yy2)-yy2)
            self.addnote(x0+length,y0-0.5*uppery(length,yy2),ewidth0=uppery(length,yy2)-yy2)
            self.addnote(x0+length,y0,egap0=gap)
            def lengthnote(self,c,below=True):
                (x0,y0,dx,dy) = curveboundingbox(c)
                self.addnote(x0+0.5*dx,y0+dy,length=dx)
            for c in cs:
                lengthnote(self,c)
            if style in 'JK': # add pads for termination resistor
                ( t3x, t3y),( t4x, t4y),( t5x, t5y) =  cs[-2][-2], cs[-1][-2], cs[-1][-3]
                (tt3x,tt3y),(tt4x,tt4y),(tt5x,tt5y) = ccs[-2][-2],ccs[-1][-2],ccs[-1][-3]
                assert t3x==tt3x and t4x==tt4x and t5x==tt5x and t3y==t4y==t5y and tt3y==tt4y==tt5y
                # print(t3x,t3y,t4x,t4y,t5x,t5y,tt3x,tt3y,tt4x,tt4y,tt5x,tt5y)
                zs = np.linspace(0,1,101)
                xs = t3x + (t4x-t3x)*zs
                ys  =  t3y + ypad*sin(0.5*pi*zs)**2
                yys = tt3y - ypad*sin(0.5*pi*zs)**2
                vt0,vt1 = list(zip(xs,ys*0+t3y)), list(zip(xs,ys))
                vtt0,vtt1 = list(zip(xs,yys*0+tt3y)), list(zip(xs,yys))
                ct  =  vt0 +  vt1[::-1] +  vt0[:1]
                ctt = vtt0 + vtt1[::-1] + vtt0[:1]
                self.addpoly(ct).addpoly(ctt)
                self.addrect(t4x,t4y,t5x-t4x,ypad).addrect(tt4x,tt4y-ypad,tt5x-tt4x,ypad)
                self.addnote(t5x,t5y+0.5*ypad,ewidth0=ypad).addnote(tt5x,tt5y-0.5*ypad,ewidth0=ypad)
            return self
        g = Element('electrode',layer='ELECTRODE',parent=self)
        vivaldi(g,length,gap,dw,dy,xexp,xtaper,x0,y0,style)
        bx0,by0,bdx,bdy = g.boundingbox()
        g.addsubmountmetric(x=bx0+bdx+200,y=by0+600)
        g.translate(*translate)
        if mirrored:
            gg = Element('electrode',layer='ELECTRODE',parent=self)
            vivaldi(gg,length,gap,dw,dy,xexp,xtaper,x0,y0,style)
            gg.rotate(pi,x0=0.5*dx,y0=y0)
            gg.translate(*translate)
        return self
    def addboeingpmelectrode(self,length=40000,style='A',hot=305,gap=7,dicegap=209,translate=(0,0)):
        g = Element('electrode',layer='ELECTRODE',parent=self)
        import phidl,phidls
        pm = phidls.boeingpm if style in 'ABCDE' else phidls.boeingpm2
        D = pm(length,style=style,hot=hot,mirror=1).movey(dicegap/2) if style in 'ABCDEFG' else pm(length,style=style,hot=hot,gap=gap,mirror=1).movey(dicegap/2)
        polys = phidls.layerpolygons(D,layers=[0],closed=True)
        g.addpolys(polys)
        x0,y0,dx,dy = g.boundingbox()
        print('g.boundingbox() x0,y0,dx,dy',x0,y0,dx,dy)
        g.addsubmountmetric(x=x0+dx-500,y=y0+600)
        b = Element('buffer',layer='BUFFER',parent=self)
        bufferpolys = phidls.layerpolygons(D,layers=[1],closed=True)
        b.addpolys(bufferpolys)
        for k,s,(x,y) in D.notes:
            print(k,s)
            g.addnote(x=x,y=y+dicegap/2,dy=10,**{k:s})
        g.translate(*translate)
        b.translate(*translate)
        return self
    def addetchtestchip(self,chiptype,dicegap):
        def addsbendgroup(self,dx,pitch,bendcounts=(2,10,50)):
            g = Group(parent=self)
            g.info.sbendlength = dx
            g.addguidelabels(dy=-25,extratext=' '+','.join([str(b)for b in bendcounts])+' S-bends',metrics=[3])
            for n in bendcounts:
                g.addsbendguide(sbendlength=dx,sbendcount=n,pitch=pitch,width=3,mf=0,invert=False,res=None)
            self.y += pitch
            return self
        def addubendgroup(self,radius,width,bendcounts=(4,20,100)):
            g = Group(parent=self)
            g.addguidelabels(dy=-25,extratext=' '+','.join([str(b)for b in bendcounts])+' U-bends',metrics=[3])
            for i,n in enumerate(bendcounts):
                g.addubendguide(radius=radius,ubendcount=n,width=width,mf=0,xinner=6*radius*(i+1),period=4*radius*2*len(bendcounts))
            self.y += 4*radius
            return self
        def addqigroup(self,width,innerpitch=100,outerpitch=200):
            self.y += 0.5*(outerpitch+innerpitch)
            g = Group(parent=self).addguidelabels(dy=-25-0.5*(outerpitch+innerpitch),metrics=[3])
            g.addqi(width,innerpitch=innerpitch,outerpitch=outerpitch,splitradius=1.0,verbose=0)
            self.y += 125 # 25
            return self
        label = f" {self.info.chipid} {chiptype}"
        elabel = self.finddefaultinfo('electrodemaskname')+label
        # print(label,chiptype)
        self.info.guidespacing = 25
        self.info.groupspacing = 75
        dx,dy = self.finddefaultinfo('chiplength'),self.finddefaultinfo('chipwidth')
        self.addtextrelative(self.finddefaultinfo('maskname')+label,x=2000,y=110+0.5*dicegap)
        self.addtextrelative(self.finddefaultinfo('maskname')+label,x=0.5*dx,y=110+0.5*dicegap)
        self.addoversizemetrics(3,0.5*dx-2000,110+0.5*dicegap)
        self.adddiceguides(x=0,y=0.5*dicegap,chipx=dx,chipy=dy-dicegap,s=10)
        self.y = 0.5*dicegap + 200
        g1 = Group(parent=self).addguidelabels(dy=-25,metrics=[3])
        for wi in [2.6,2.8,3.0,3.2]:
            g1.addchannel(wi)
        addqigroup(self,3)
        g3 = Group(parent=self)
        for wi in [1,2,3,4,5]:
            g3.addchannel(wi)
        if chiptype=='SBEND':
            # addsbendgroup(self,1200,pitch=100)
            # addsbendgroup(self,1200,pitch=50)
            # addsbendgroup(self, 600,pitch=50)
            # addsbendgroup(self, 300,pitch=50)
            # addsbendgroup(self, 150,pitch=50)
            addsbendgroup(self,1000,pitch=12.5)
            addsbendgroup(self,1000,pitch=50)
            addsbendgroup(self, 500,pitch=50)
            addsbendgroup(self, 250,pitch=50)
            addsbendgroup(self, 125,pitch=50)
        if chiptype=='UBEND':
            # addubendgroup(self,radius=500,width=3,bendcounts=(4,12,20))
            # addubendgroup(self,radius=200,width=3,bendcounts=(4,12,36))
            addubendgroup(self,radius=80,width=3)
            addubendgroup(self,radius=40,width=3)
            addubendgroup(self,radius=20,width=3)
        if chiptype=='RING':
            g4 = Group(parent=self).addguidelabels(dy=-25,metrics=[3])
            g4.addspiralguide(width=3,n=5,gap=25,taperx=10000,res=200)
            g4.addspiralguide(width=3,n=15,gap=25)
            self.y += (15+1)*25
            g5 = Group(parent=self).addguidelabels(dy=-25,metrics=[3])
            g5.addringguide(width=3,radius=50,yoffset=6,xinner=   0,xoffset=1000); g5.y += 2*50+6
            g5.addringguide(width=3,radius=50,yoffset=6,xinner=1000,xoffset=2000,invert=True)
            g5.addringguide(width=3,radius=50,yoffset=6,xinner= 200,xoffset=11000); g5.y += 2*50+6
            g5.addringguide(width=3,radius=50,yoffset=6,xinner=5000,xoffset=12000,invert=True)
        if chiptype=='BRAGG':
            g4 = Group(parent=self).addguidelabels(dy=-25,metrics=[3])
            for dc,period in zip((0.80,0.90,0.95,0.98),(5,10,20,50)):
               g4.addbragg(width=3,dc=dc,period=period)
            g5 = Group(parent=self).addguidelabels(dy=-25,metrics=[3])
            for period in (5,10,20,40,80):
                g5.addcorrugatedbragg(guidewidth=3,braggwidth=2,dc=0.5,period=period,dx=None)
            g6 = Group(parent=self)
            for period in (5,10,20,40,80):
                g6.addpibragg(guidewidth=3,braggwidth=2,dc=0.5,period=period,dx=None)
            g7 = Group(parent=self).addguidelabels(dy=-25,metrics=[3])
            for period in (5,10,20,40,80):
                g7.addcorrugatedbragg(guidewidth=3,braggwidth=2.8,dc=0.5,period=period,dx=None)
            g8 = Group(parent=self)
            for period in (5,10,20,40,80):
                g8.addpibragg(guidewidth=3,braggwidth=2.8,dc=0.5,period=period,dx=None)
        if chiptype=='SPLIT':
            for r in (0.5,1.0,2.0):
                g = Group(parent=self)
                g.info.modefilterlength = 1000
                g.info.taperlength = 2000
                g.info.sbendlength = 5000
                g.addsplitter(width=3,pitch=50,mzlength=1000,splitradius=r)
            for y in (6,7,8,9):
                g = Group(parent=self)
                g.info.modefilterlength = 20000
                g.info.taperlength = 2000
                g.info.sbendlength = 5000
                g.info.vgroovepitch = 50
                g.addqfc(mfpump=3,mfin=3,mfout=3,width=3,couplergap=y,couplerx=1000,couplerwidth=3,
                    period=None,pumponsbend=False,gratingonsbend=True,grating=0,verbose=0)
    def addetchchip(self,pmwidth,dicegap):
        label = f" {self.info.chipid} {pmwidth:g}w"
        elabel = self.finddefaultinfo('electrodemaskname')+label
        self.info.guidespacing = 25
        self.info.groupspacing = 300
        self.info.modefilterlength = 3000
        self.info.taperlength = 3000
        dx,dy = self.finddefaultinfo('chiplength'),self.finddefaultinfo('chipwidth')
        def addheader(self):
            e = Element(parent=self)
            def polyrect(x0,y0,dx,dy):
                return reorient([(x0,y0),(x0+dx,y0),(x0+dx,y0+dy),(x0,y0+dy),(x0,y0)],False)
            for x in (0,dx/2):
                e.addinsettext([polyrect(x,0.5*dicegap+50,dx/2,200)],self.finddefaultinfo('maskname')+label,x0=x+2000,y0=0.5*dicegap+50+150,
                    fitx=0,fity=0,margin=0,scale=1,fitcenter=True,vertical=False,scaleifdev=True,font=None,justify='left')
            return e
        def addetchblock(self,y=25,dy=self.info.groupspacing-50):
            e = Element(parent=self)
            e.addrect(0,self.y+y,dx,dy)
            e.addnote(200,self.y+y+dy/2,width=dy).addnote(200+dx/2,self.y+y+dy/2,width=dy)
            return e

        self.adddiceguides(x=0,y=0.5*dicegap,chipx=dx,chipy=dy-dicegap,s=10)
        addheader(self)
        for x in (0.25*dx,0.50*dx,0.75*dx,):
            self.addoversizemetrics(3,x-2000,300+0.5*dicegap)

        self.y = 0.5*dicegap + 350
        g1 = Group(parent=self).addguidelabels(dy=-25,metrics=[4,3,2])
        # g1sb0,g1sb1 = 1750,1750
        g1sb0,g1sb1 = 5500,7800
        couplerx = dx-g1sb0-g1sb1-2*(3000+3000)-2000
        for wi,split,fL in zip([pmwidth,pmwidth,pmwidth],[pmwidth+2,pmwidth+2,pmwidth+2],[0.25,0.5,1.0]):
            g1.addwdm(width=wi,couplergap=split,couplerx=fL*couplerx,couplerwidth=wi,
                mfsbendin=3,mfsbendout=3,mfchannelin=3,mfchannelout=3,
                modefilterx=3000,taperx=3000,mirror=False,vgroovepitch=127,metrics=True,sb0=g1sb0,sb1=g1sb1)

        addetchblock(self)
        g2 = Group(parent=self)
        # g2sb0,g2sb1 = 2500,2500
        g2sb0,g2sb1 = 5500,7800
        couplerx = dx-g2sb0-g2sb1-2*(3000+3000)-2000
        for wi,split,fL in zip([pmwidth-3,pmwidth-3,pmwidth-3],[pmwidth-1,pmwidth-1,pmwidth-1],[0.25,0.5,1.0]):
            g2.addwdm(width=wi,couplergap=split,couplerx=fL*couplerx,couplerwidth=wi,
                mfsbendin=3,mfsbendout=3,mfchannelin=3,mfchannelout=3,
                modefilterx=3000,taperx=3000,mirror=False,vgroovepitch=127,metrics=True,sb0=g2sb0,sb1=g2sb1)

        addetchblock(self)
        g3 = Group(parent=self)
        for wi in [3,3,4,4,5,5,6,6]:
            g3.addmodefilter(wi,outwidth=3)

        addetchblock(self)
        g4 = Group(parent=self)
        for wi in [6,6,7,7,8,8,9,9]:
            g4.addmodefilter(wi,outwidth=3)

        # self.y -= 200
        # g5 = Group(parent=self).addmodefilter(pmwidth,outwidth=3)
        addetchblock(self)
        self.y += 150
        g5 = Group(parent=self).addmodefilter(pmwidth,outwidth=3)
        self.y += 300

        addetchblock(self,dy=200)
        g6 = Group(parent=self).addguidelabels(dy=-25,metrics=[4,3,2])
        g6.addsplitter(width=pmwidth,pitch=127,mfin=3,mfout=3,mzlength=1000,sbendlength=1250,splitradius=2.0)
        
        addetchblock(self)
        g7 = Group(parent=self)
        g7.addsplitter(width=pmwidth,pitch=127,mfin=3,mfout=3,mzlength=1000,sbendlength=1750,splitradius=2.0)

        addetchblock(self)
        g8 = Group(parent=self)
        for wi in [3,4,5,6,7,8,9,10]:
            g8.addmodefilter(wi,outwidth=3)

        yend = dy - 0.5*dicegap - 50
        addetchblock(self,y=25,dy=yend-(self.y+25))

    def addboeingxcutpmchip(self,targetwidth,pmlength,chiptype,chiphot,chipgap,wgextraoxide=False,
            pitch=113,yrad=0.5,mf=0,sbendx=5000,taperx=None,mfx=0,dicegap=None,suppresswaveguides=False,translate=(0,0),mirrored=False):
        label = f" {self.info.chipid}" # {hot}-{gap}-{hot} {targetwidth}W"
        elabel = self.finddefaultinfo('electrodemaskname')+label
        self.info.guidespacing = 127/4
        self.info.groupspacing = 127
        self.info.modefilterlength = mfx
        self.info.taperlength = 500 if taperx is None else taperx
        self.info.sbendlength = sbendx
        dx,dy = self.finddefaultinfo('chiplength'),self.finddefaultinfo('chipwidth')
        if not suppresswaveguides:
            self.addtextrelative(self.finddefaultinfo('maskname')+label,x=2000,y=480+dicegap)
            self.addtextrelative(self.finddefaultinfo('maskname')+label,x=self.finddefaultinfo('chiplength')/2,y=480+dicegap)
            self.adddiceguides(x=0,y=dicegap/2,chipx=dx,chipy=dy-dicegap,s=10)
            self.y = dy/2-570
            print(label)

            g1 = Group(parent=self).addguidelabels(dy=-50,skip=[55000,60000])
            sy0 = self.y
            for wi in [6,6,6,6,6]:
                g1.addchannel(wi)
            sy1 = self.y
            if wgextraoxide:
                Element('buffer',layer='BUFFER',parent=self).addrect(0,sy0-20,dx,sy1-sy0+40)

        self.y = dy/2-self.info.groupspacing
        assert dicegap/2+900-127==self.y, self.y
        g2 = Group(parent=self)
        if not suppresswaveguides:
            # print(g2.y); exit() # 1004.5
            g2.addchannel(targetwidth)
        if chiptype=='PMH':
            assert not mirrored
            g2.addboeingpmelectrode2(
                length=pmlength,style=chiptype[-1],hot=chiphot,gap=chipgap,dicegap=dicegap,translate=translate)
        if chiptype in ('PMI','PMJ','PMK'):
            g2.addboeingpmvivaldi(
                length=pmlength,style=chiptype[-1],gap=chipgap,dicegap=dicegap,translate=translate,mirrored=mirrored)
        self.y += 127/2

        if not suppresswaveguides:
            g3 = Group(parent=self)
            sy0 = self.y
            for wi in [5,6,7,8,9]:
                g3.addchannel(wi)
            sy1 = self.y
            if wgextraoxide:
                Element('buffer',layer='BUFFER',parent=self).addrect(0,sy0-20,dx,sy1-sy0+40)
            g4 = Group(parent=self)
            sy0 = self.y
            g4.addsplitter(width=targetwidth,pitch=pitch,mfin=mf,mfout=mf,mzlength=1000,sbendlength=self.info.sbendlength,splitradius=yrad)
            sy1 = self.y
            if wgextraoxide:
                Element('buffer',layer='BUFFER',parent=self).addrect(0,sy0-20,dx,sy1-sy0+40)

        self.addscalenote(dicegap=dicegap)
        return self

class Group(Element):
    def __init__(self,name='group',layer=None,parent=None,showgroupspacing=None,showguidespacing=None,**kwargs):
        assert parent, 'Group must have parent'
        # assert parent.info.groupspacing, 'Group groupspacing must be defined in parent'
        assert parent.finddefaultinfo('groupspacing') is not None, 'Group groupspacing must be defined in parent'
        super(Group,self).__init__(name,layer,parent,x=None,y=None,**kwargs)
        parent.info.groupcount = self.info.groupnumber = getattr(parent.info,'groupcount',0) + 1
        if parent.info.groupcount>1: parent.y += parent.info.groupspacing
        if showgroupspacing or (2==self.info.groupnumber and showgroupspacing is None): # show group spacing note between g1.last and g2.1
            parent.addnote(parent.x+1000,parent.y-parent.info.groupspacing/2,separation=parent.info.groupspacing)
        if showguidespacing is not None:
            self.info.showguidespacing = showguidespacing
        self.x,self.y = parent.x,parent.y
class Guide(Element):
    def __init__(self,name='guide',layer=None,parent=None,x=None,y=None,guidespacing=None,showspacing=None,width=None,**kwargs):
        assert parent, 'Guide must have parent'
        assert isinstance(parent,Group), 'Guide must have Group as parent'
        super(Guide,self).__init__(name,layer,parent,x=x,y=y,**kwargs)
        guidespacing = guidespacing if guidespacing is not None else self.finddefaultinfo('guidespacing')
        parent.info.guidecount = getattr(parent.info,'guidecount',0) + 1
        parent.info.groupnumber = getattr(parent.info,'groupnumber',1)
        if parent.info.guidecount>1:
            parent.y = parent.parent.y = parent.y + guidespacing
            # both group and chip are updated with y position of current waveguide
            # chip.y should probably be updated when adding wide guides (spltter,mz) but isn't yet
        showspacing = showspacing if showspacing is not None else (2==parent.info.guidecount and 1==parent.info.groupnumber)
        if hasattr(parent.info,'showguidespacing'):
            showspacing = (2==parent.info.guidecount and parent.info.showguidespacing)
        if showspacing: # show guide spacing note between g1.1 and g1.2
            parent.addnote(parent.x+1000,parent.y-guidespacing/2,separation=guidespacing)
        self.info.guidenumber = f"{parent.info.groupnumber}.{parent.info.guidecount}"
        self.x = x if x is not None else parent.x
        self.y = y if y is not None else parent.y
        if hasattr(parent,'width'):
            self.width = parent.width  # for keeping track of current waveguide width
        if width is not None:
            self.width = width
    def addonchannel(self,dx,width=None,note=True,x=None,y=None):
        assert 0<=dx
        x = x if x is not None else self.x
        y = y if y is not None else self.y
        width = width if width is not None else self.width
        self.addrect(x,y-width/2.,dx,width)
        self.x,self.y,self.width = x+dx,y,width
        if note:
            self.addnote(x+dx/2,y,width=width)
            self.addnote(x+dx/2,y,length=dx)
        return self
    def addontaper(self,dx,outwidth,inwidth=0,linear=False,note=True,dy=0):
        x,y = self.x,self.y
        if 0==dx: return self
        assert dx>0
        if 0==inwidth: inwidth = self.width
        self.x,self.width = self.x+dx,outwidth
        if linear:
            self.addpoly([(x+xx,y+yy) for xx,yy in zip([0,0,dx,dx,0],[-inwidth/2.,inwidth/2.,outwidth/2.,-outwidth/2.,-inwidth/2.])])
        else:
            from numpy import linspace
            # xs = list(linspace(0,dx,101))
            # ys = [inwidth/2.+ (outwidth/2.-inwidth/2.)*(1-cos(pi*xx/dx))/2. for xx in xxs]
            xs = linspace(0,dx,101)
            ys = inwidth/2.+ (outwidth/2.-inwidth/2.)*(1-np.cos(np.pi*xs/dx))/2.
            zs = dy*xs/dx
            if dy:
                xs,ys = list(xs) + list(xs[::-1]), list(zs+ys) + list(zs-ys)[::-1]
            else:
                xs,ys = list(xs) + list(xs[::-1]), [yy for yy in ys] + [-yy for yy in ys[::-1]]
            c = [(x+xx,y+yy) for xx,yy in zip(xs,ys)] + [(x+xs[0],y+ys[0])]
            self.addpoly(c)
        if note:
            self.addnote(x+dx/2,y,taper=dx)
        return self
    def addonhalfring(self,r,yoffset=0,open=True,invert=False,minangle=0.1,width=None,x=None,y=None,note=True):
        x = x if x is not None else self.x
        y = y if y is not None else self.y
        width = width if width is not None else self.width
        j,k = -1 if open else +1,-1 if invert else +1
        from geometry import Arch
        A = Arch(r,-pi/2,0,width=width,minangle=minangle)
        B = Arch(r,0,pi/2,width=width,minangle=minangle)
        self.addpoly((j,k)*(A.curve()+(+0,+0))+(x,y+k*yoffset))
        self.addpoly((j,k)*(B.curve()+(+r,+r))+(x,y+k*yoffset))
        if note and open:
            self.addnote(x,y+k*r,uradius=f"{r:g}r")
        return self
    def addondoubleubend(self,r,invert=False,minangle=0.1,width=None,x=None,y=None,note=True):
        x = x if x is not None else self.x
        y = y if y is not None else self.y
        width = width if width is not None else self.width
        k = -1 if invert else +1
        from geometry import Arch
        A = Arch(r,-pi/2,0,width=width,minangle=minangle)
        B = Arch(r,0,pi/2,width=width,minangle=minangle)
        C = Arch(r,-pi/2,-pi,width=width,minangle=minangle)
        D = Arch(r,pi,pi/2,width=width,minangle=minangle)
        self.addpoly((1,k)*(A.curve()+(+0,+0))+(x,y))
        self.addpoly((1,k)*(B.curve()+(+r,+r))+(x,y))
        self.addpoly((1,k)*(C.curve()+(+0,+2*r))+(x,y))
        self.addpoly((1,k)*(D.curve()+(-r,+3*r))+(x,y))
        self.x,self.y,self.width = x,y+k*4*r,width
        if note:
            self.addnote(x,y+k*r,uradius=f"{r:g}r")
        return self
    def addonsbend(self,dx,dy,width=None,x=None,y=None,note=True,res=None):
        assert dx>0
        x = x if x is not None else self.x
        y = y if y is not None else self.y
        width = width if width is not None else self.width
        if 0==dy: return self.addonchannel(dx)
        xs,ys,roc,pathlength = sbend(dx,dy,res=res)
        xs = x + np.append(xs,xs[::-1])
        ys = y + np.append(ys-width/2,ys[::-1]+width/2)
        c = [(xx,yy) for xx,yy in zip(xs,ys)] + [(xs[0],ys[0])]
        self.addpoly(c)
        self.info.rocinmm = '%.2f' % (roc/1000)
        self.x,self.y,self.width = x+dx,y+dy,width
        if note:
            self.addnote(x+dx/2,y+dy/2,length=dx)
            self.addnote(x+dx,y+dy/2,displacement=abs(dy))
            self.addnote(x+dx/2,y+dy/2,roc=abs(roc)/1000)
        return self
    def addondoublesbend(self,dx,dy,width=None,x=None,y=None,poshalf=0,neghalf=0,splitradius=10,outputside=False): # negative radius = flat \_/ (flat V shape), positive radius = circle (U shape)
        # dy = center-to-center separation
        assert dx>0
        x = x if x is not None else self.x
        y = y if y is not None else self.y
        width = width if width is not None else self.width
        xs,ys,roc,pathlength = sbend(dx,dy/2-width/2)
        r0 = np.abs(splitradius)
        x0 = np.interp(r0,ys,xs)
        xrs = np.where( r0<=ys, xs, np.where(splitradius>0,x0-np.sqrt(np.abs(r0**2-ys**2)),x0) ) # y=r0 -> x=x0, y=0 -> x=x0-r0
        if splitradius and not neghalf:
           self.addnote(x+dx-x0 if outputside else x+x0,y,splitradius=r0)
        if poshalf:
            xs = np.concatenate(( np.array([0]), xrs, xs[::-1] ))
            ys = np.concatenate(( np.array([0]), ys, ys[::-1]+width ))
            self.y = y+dy/2
        elif neghalf:
            xs = np.concatenate(( np.array([0]), xs, xrs[::-1] ))
            ys = np.concatenate(( np.array([0]), -ys-width, -ys[::-1] ))
            self.y = y-dy/2
        else:
            xs = np.concatenate(( xrs, xs[::-1], xs, xrs[::-1] ))
            ys = np.concatenate(( ys, ys[::-1]+width, -ys-width, -ys[::-1] ))
            self.y = y
        if outputside: xs = dx-xs
        xs,ys = xs+x,ys+y
        c = [(xx,yy) for xx,yy in zip(xs,ys)] + [(xs[0],ys[0])]
        self.addpoly(c)
        self.info.rocinmm = '%.2f' % (roc/1000)
        self.x,self.width = x+dx,width
        self.addnote(x+dx/2,y+dy/2,length=dx)
        self.addnote(x if outputside else x+dx,y,separation=dy)
        if not neghalf: self.addnote(x+dx/2,y-dy/2,roc=roc/1000)
        return self
    def addondcoutput(self,dx,dy,width=None,x=None,y=None,split=None,sbendx=None,Lc=None,outpitch=None,outx=None):
        # dy = center-to-center separation
        # sbendx + Lc + sbendoutx + outx == dx
        sbendoutx = dx - sbendx - Lc - outx # second sbend length chosen to make length come out right
        assert sbendoutx>0
        x = x if x is not None else self.x
        y = y if y is not None else self.y
        width = width if width is not None else self.width
        outpitch = outpitch if outpitch is not None else dy
        x0,y0 = self.x,self.y
        self.x,self.y = x0,y0-dy
        for s in (-1,+1):
            self.x,self.y = x0,y0+(s-1)/2*dy
            self.addonsbend(sbendx,-0.5*s*(dy-split),width,note=(outpitch!=dy))
            self.addonchannel(Lc,width)
            self.addonsbend(sbendoutx,+0.5*s*(outpitch-split),width)
            self.addonchannel(outx,width)
        self.addnote(x0+sbendx+Lc/2,y0-dy/2,separation=split)
        return self
    def addonmodefilter(self,width,dx,inwidth=0,outwidth=0,modefilterx=0,taperx=0):
        if inwidth: self.addonchannel(modefilterx,inwidth).addontaper(taperx,outwidth=width)
        assert 0<dx-(modefilterx+taperx)*(bool(inwidth)+bool(outwidth)), 'chip not long enough for mode filters'
        self.addonchannel(dx-(modefilterx+taperx)*(bool(inwidth)+bool(outwidth)),width)
        if outwidth: self.addontaper(taperx,outwidth=outwidth).addonchannel(modefilterx,outwidth)
        return self
    def addonsplittapersbend(self,dx,dy,taperx,sx,sy,inwidth=None,outwidth=None,x=None,y=None,reverse=False,note=True):
        # starts with tight ROC at wide guide width, then in middle of s-bend tapers down to smaller guide width, and finishes s-bend with loose ROC
        assert dx>0
        x = x if x is not None else self.x
        y = y if y is not None else self.y
        inwidth = inwidth if inwidth is not None else self.width
        if 0==dy: return self.addonchannel(dx)
        # xs,ys,roc,pathlength = sbend(dx,dy)
        # x1s,y1s,x2s,y2s,roc1,roc2,ds = doublesbend(11000,127,2500,3400,10)
        x1s,y1s,x2s,y2s,roc1,roc2,ds = splittapersbend(dx,dy,taperx,sx,sy)
        self.info.splittaperdx = x2s[-1] - x1s[-1] # print('x1s[0]',x1s[0],'x2s[0]',x2s[0],'x1s[-1]',x1s[-1],'x2s[-1]',x2s[-1])
        if reverse:
            x2s,x1s = dx - x2s,dx - x1s
            xs = x + np.append(x2s,x2s[::-1])
            ys = -dy + y + np.append(y2s-outwidth/2,y2s[::-1]+outwidth/2)
            cc = [(xx,yy) for xx,yy in zip(xs,ys)] + [(xs[0],ys[0])]
            self.addpoly(cc)
            self.x,self.y = x+x2s[0],y-dy+y2s[0]
            self.addontaper(taperx,outwidth=inwidth,dy=y1s[-1]-y2s[0])
            xs = x + np.append(x1s,x1s[::-1])
            ys = -dy + y + np.append(y1s-inwidth/2,y1s[::-1]+inwidth/2)
            c = [(xx,yy) for xx,yy in zip(xs,ys)] + [(xs[0],ys[0])]
            self.addpoly(c)
            self.x,self.y,self.width = x+dx,y-dy,inwidth
        else:
            xs = x + np.append(x1s,x1s[::-1])
            ys = y + np.append(y1s-inwidth/2,y1s[::-1]+inwidth/2)
            c = [(xx,yy) for xx,yy in zip(xs,ys)] + [(xs[0],ys[0])]
            self.addpoly(c)
            self.x,self.y = x+x1s[-1],y+y1s[-1]
            self.addontaper(taperx,outwidth=outwidth,dy=y2s[0]-y1s[-1])
            xs = x + np.append(x2s,x2s[::-1])
            ys = y + np.append(y2s-outwidth/2,y2s[::-1]+outwidth/2)
            cc = [(xx,yy) for xx,yy in zip(xs,ys)] + [(xs[0],ys[0])]
            self.addpoly(cc)
            self.x,self.y,self.width = x+dx,y+dy,outwidth
        self.info.rocinmm = '%.2f %.2f' % (roc1/1000,roc2/1000)
        if note and not reverse:
            self.addnote(x+dx/2,y+dy/2,length=dx)
            self.addnote(x+dx,y+dy/2,displacement=abs(dy))
            self.addnote(x+0.5*dx,y+0.5*dy,roc=abs(roc1 if reverse else roc2)/1000)
            self.addnote(x+dx,y+dy,roc=abs(roc2 if reverse else roc1)/1000)
        return self
    def addoncorrugatedbragg(self,dx,guidewidth,braggwidth,period,dc,piphaseshift=False):
        x,y = self.x,self.y
        if dev:
            period = periodmag*period
        barcount = int(dx/period)-1
        barcount = 2*(barcount//2) if piphaseshift else barcount
        xpi = 0.5*period if piphaseshift else 0
        xin = 0.5*(dx-barcount*period-xpi)
        def addbar(self,x,y,L,w):
            if 0<L and 0<w:
                self.addcenteredrect(x+0.5*L,y,L,w)
        addbar(self,x,y,xin,guidewidth)
        for i in range(barcount):
            xi = x+xin+i*period+xpi*(barcount//2-1<i)
            xg,xb = 0.5*dc*period,(1-dc)*period
            # addbar(self,      xi,y,xg,guidewidth)
            # addbar(self,   xg+xi,y,xb,braggwidth)
            # addbar(self,xb+xg+xi,y,xg,guidewidth)
            addbar(self,      xi,y,xg if 0==i else 0,guidewidth)
            addbar(self,   xg+xi,y,xb,braggwidth)
            addbar(self,xb+xg+xi,y,xg if i==barcount-1 else 2*xg+xpi*(i==barcount//2-1),guidewidth)
        addbar(self,x+xin+barcount*period+xpi,y,xin,guidewidth)
        self.x = self.x+dx
        # if note:
        #     self.addnote(x+dx/2,y,width=width)
        #     self.addnote(x+dx/2,y,length=dx)
        #     self.addnote(x+dx/2,y,bragg='%.3gΛ,%d%%'%(period,100*dc)) # mzub # print('%.2gΛ, %.2gdc'%(period,dc),(period,dc))
        #     if enddc:
        #         self.addnote(x+dx/2,y,taper=dx)
        #         assert 0, 'test implementation of this note'
        return self
    def addonbragg(self,dx,width=None,period=8,dc=1.0,enddc=None,note=True):
        x,y = self.x,self.y
        width = width if width is not None else self.width
        self.addbragggrating(width,dx,period,dc,enddc,x0=x,y0=y) # self.addrect(x,y-width/2.,dx,width)
        self.x,self.width = self.x+dx,width
        if note:
            self.addnote(x+dx/2,y,width=width)
            self.addnote(x+dx/2,y,length=dx)
            self.addnote(x+dx/2,y,bragg='%.3gΛ,%d%%'%(period,100*dc)) # mzub # print('%.2gΛ, %.2gdc'%(period,dc),(period,dc))
            if enddc:
                self.addnote(x+dx/2,y,taper=dx)
                assert 0, 'test implementation of this note'
        return self
    def addonbraggmodefilter(self,width,period,dx,indc=1,outdc=1,modefilterx=None,taperx=None):
        inmf,outmf = bool(indc<1),bool(outdc<1)
        if inmf: self.addonbragg(modefilterx,width,period,dc=indc).addonbragg(taperx,width,period,dc=indc,enddc=1)
        assert 0<dx-(modefilterx+taperx)*(bool(inmf)+bool(outmf)), 'chip not long enough for bragg mode filters'
        self.addonchannel(dx-(modefilterx+taperx)*(bool(inmf)+bool(outmf)),width)
        if outmf: self.addonbragg(taperx,width,period,dc=1,enddc=outdc).addonbragg(modefilterx,width,period,dc=outdc)
        return self
    def addontaperedbragg(self,dx,outwidth,inwidth=None,period=8,dc=1.0,enddc=None):
        x,y = self.x,self.y
        inwidth = inwidth if inwidth is not None else self.width
        self.addtaperedbragggrating(inwidth,outwidth,dx,period,dc,enddc,x0=x,y0=y)
        self.x,self.width = self.x+dx,outwidth
        return self
    def addontaperedbraggmodefilter(self,mfwidth,width,period,dx,indc=1,outdc=1,modefilterx=None,taperx=None):
        inmf,outmf = bool(indc<1 or not mfwidth==width),bool(outdc<1 or not mfwidth==width)
        if inmf:
            if 1==indc:
                self.addonchannel(modefilterx,width=mfwidth).addontaper(taperx,inwidth=mfwidth,outwidth=width)
            else:
                self.addonbragg(modefilterx,mfwidth,period,dc=indc).addontaperedbragg(taperx,inwidth=mfwidth,outwidth=width,period=period,dc=indc,enddc=1)
        assert 0<dx-(modefilterx+taperx)*(bool(inmf)+bool(outmf)), 'chip not long enough for bragg mode filters'
        self.addonchannel(dx-(modefilterx+taperx)*(bool(inmf)+bool(outmf)),width)
        if outmf:
            if 1==outdc:
                self.addontaper(taperx,inwidth=width,outwidth=mfwidth).addonchannel(modefilterx,width=mfwidth)
            else:
                self.addontaperedbragg(taperx,inwidth=width,outwidth=mfwidth,period=period,dc=1,enddc=outdc).addonbragg(modefilterx,mfwidth,period,dc=outdc)
        return self
class Chip(Element):
    # def __init__(self,parent,name=None,layer=None,**kwargs): #  chip must have parent
    def __init__(self,name=None,layer=None,parent=None,x=None,y=None,id=None,**kwargs):
        n = getattr(parent.info,'chipcount',0)
        name = name if name is not None else 'chip'
        super(Chip,self).__init__(name,layer,parent,x,y,**kwargs)
        if id is not None:
            self.info.chipid,self.info.chipnumber,parent.info.chipcount = id,n+1,n+1
            return
        nx,ny = 0,n
        # assert hasattr(parent.info,'rows'), 'chip id needs mask.info.rows'
        if hasattr(parent.info,'rows'):
            nx,ny = n//parent.info.rows,n%parent.info.rows
            parent.info.columns = max(getattr(parent.info,'columns',0),nx+1)
        self.info.chipid,self.info.chipnumber,parent.info.chipcount = f'{ny+1:02d}'+'ABCDEFGHIJ'[nx],n+1,n+1
    def addscalenote(self,extra='',poling=True,bragg=False,dicegap=0):
        dx,dy = self.finddefaultinfo('chiplength'),self.finddefaultinfo('chipwidth')
        s = f"chip length = {dx}, chip width = {dy-dicegap}{' after dicing'*bool(dicegap)}{(', dice gap = '+str(dicegap))*bool(dicegap)}, units in µm"
        if dev and not 1==scalemag: s += ', %s× vertical scaling' % scalemag
        if dev and not 1==periodmag and poling: s += ', %s× poling period scaling' % periodmag
        if bragg: s = s.replace('poling','poling and bragg')
        if extra: s = s+', '+extra
        self.addnote(dx/2,75,note=s)
class Submount(Chip):
    def addpads(self,padstarts,padends,padtext,padcount,padinputy,gx,gy,inputconnected,outputconnected,design=None):
        def addpad(self,padnum,text,startx,endx,padinputy,gx,gy,inputconnected,outputconnected,design=None,mock=False):
            assert (startx==startx and endx==endx) or (startx!=startx and endx!=endx) # both are nan or neither
            hasbars = (startx==startx) # hasbars if not nan
            tabx,spiney = 1000,100
            taby = padinputy-spiney
            # if not endx: endx = gx 
            endx = endx if (endx and endx==endx) else gx # todo:case for not exactperiod, dicegapx/y
            spineendx = gx if inputconnected else endx
            startx = startx if startx==startx else 0
            startx = 0 if inputconnected else startx
            x0,y0,dx,dy = gx-tabx,0,max(tabx-gx+endx,500),taby # top tab is at least 500 wide
            self.addtext(text, x=startx,y=taby, fitx=gx-tabx-startx,fity=taby, margin=100, scale=5,scaleifdev=False)
            # self.addtext(text, x=0,y=taby, fitx=gx-tabx,fity=taby, margin=100, scale=5,scaleifdev=False)
            if not hasbars:
                return self
            if not mock: self.addsubmountmetric(startx+12,taby-12)
            if mock: self.addrect(min(x0,startx),taby+spiney, max(spineendx,x0+dx)-min(x0,startx),gy)
            if design is None or mock: # top tab
                self.addrect(x0,y0,dx,dy)
            elif 'circle'==design[0]:
                r = design[1]/2 # diameter = grating period
                # self.addpoly([(x0,y0),(x0+dx,y0),(x0+dx,y0+dy),(x0,y0+dy),(x0,y0)]) # top tab outline
                def polycircle(x0,y0,rx,ry=None,num=40,ccw=True): # ccw circle with start at min y (x→ y↓ axes)
                    q = np.linspace(0,2*np.pi,num,endpoint=False)
                    xs,ys = (-1 if ccw else 1)*rx*np.sin(q),-(rx if ry is None else ry)*np.cos(q)
                    c = [(x+x0,y+y0) for x,y in zip(xs,ys)]
                    return c+c[:1]
                # c = [(0,0),(dx/2,0)] + polycircle(dx/2,dy/2,r) + [(dx/2,0),(dx,0),(dx,dy),(0,dy),(0,0)] # single circle
                # λ 1064nm, index 1.449631, brewster's angle in SiO2 55.400895°, aspect 1.761088 # 1% difference for 405 vs 1550
                dxs,rxs,rys = list(dx/2+np.linspace(-250,250,5)),[2*r,r,r,r,2*r],[2*r,1.761*r,r,1.761*r,2*r] # 5 holes
                # dxs,rxs,rys = list(dx/2+np.linspace(-250,250,5))[1:-1],[r,r,r],[1.761*r,r,1.761*r] # 3 holes
                cs =  [ [(xi,0)] + polycircle(xi,dy/2,ri,rj) + [(xi,0)] for xi,ri,rj in zip(dxs,rxs,rys) ]
                c = [(0,0)] + [p for c in cs for p in c] + [(dx,0),(dx,dy),(0,dy),(0,0)]
                if r<=25: # don't make holes if they are too big
                    self.addpoly([(x+x0,y+y0) for x,y in c])
                else:
                    self.addpoly([(x0,y0),(x0+dx,y0),(x0+dx,y0+dy),(x0,y0+dy),(x0,y0)])
            else:
                assert 0
            # self.addrect(startx,taby, spineendx-startx,spiney) # top spine
            self.addrect(min(x0,startx),taby, max(spineendx,x0+dx)-min(x0,startx),spiney) # top spine, extends at least to front and back end of top tab
            self.addrect(startx,taby+spiney+gy, spineendx-startx,spiney) # bottom spine
            self.addrect((0 if outputconnected else gx-tabx),taby+spiney+gy+spiney, (gx if outputconnected else tabx-gx+endx),taby) # bottom tab
            return self
        for n in range(padcount):
            s = padtext[n%len(padtext)]
            # e = Element('pad'+str(n),layer='MASK')
            e = Element('pad'+str(n),layer='MASK',parent=self)
            # print('padstarts,padends',padstarts,padends)
            mock = Element('pad'+str(n),layer='MOCKUP',parent=e)
            addpad(e,   n,s,padstarts[n]-gx*n,padends[n]-gx*n,padinputy,gx,gy,inputconnected,outputconnected,design=design)
            addpad(mock,n,s,padstarts[n]-gx*n,padends[n]-gx*n,padinputy,gx,gy,inputconnected,outputconnected,design=design,mock=True)
            e.translate(gx*n)
            e.translate(self.x,self.y)
            # self.addelem(e)
        self.info.text = getattr(self.info,'text','') + ','.join(padtext)
        return self
    def savegrating(self,barstarts,barends):
        assert len(barstarts)==len(barends), f"unequal barstarts,barends: {len(barstarts)},{len(barends)}"
        for a,b in zip(barstarts,barends): assert a<b, 'invalidbar:'+str(a)+' '+str(b)
        name = self.info.chipid
        file = f'{self.parent.name}gratings.py'
        with open(file,'w' if '01A'==name else 'a') as f:
            f.write(f'barstarts{name},barends{name} = {list(barstarts)},{list(barends)}\n')
    def addsubmount(self,period=333,dc=0.5,phase=0,padcount=10,padinputy=1000,gx=2500,gy=3000,gapx=0,breakupgapsize=0,breakupgapbar=1,
        inputconnected=False,outputconnected=True,paddesign=True,apodize=None,padtext=None,omitpads=()):
        # print('period,dc,gx,padcount,gapx,phase',period,dc,gx,padcount,gapx,phase)
        # add bars for all pads as single grating
        barstarts,barends = grating(period,dc,gx,padcount,gapx,phase,apodize=apodize,omitpads=omitpads)
        if breakupgapsize:
            from grating import breakupgaps
            barstarts,barends = breakupgaps(barstarts,barends,maxgap=breakupgapsize,barsize=breakupgapbar)
        self.addgrating(barstarts,barends,padinputy,gy)
        self.info.averageperiod = averageperiod(barstarts,barends) # print( 'averageperiod',self.info.averageperiod )
        # add each pad
        padstarts,padends = findpadboundarys(barstarts,barends,gx,padcount)
        padtext = padtext if padtext is not None else makepadtext(self.parent.name,self.info.chipid,period,dc,padcount)
        if breakupgaps: padtext[9]='AA' # note anti-arc protection bars
        if apodize:
            padtext[3] = {'asingauss23':'AGAU','asintriangle':'ATRI','trapezoid':'TRAP'}[apodize.replace('inverse','')] # replace duty cycle with type
        self.addpads(padstarts,padends,padtext,padcount,padinputy,gx,gy,inputconnected,outputconnected,design=['circle',period] if paddesign else None)
        self.savegrating(barstarts,barends)
        return self
    def addfixeddomainsubmount(self,period,padcount,padinputy=1000,gx=2500,gy=3000,gapx=0,inputconnected=False,outputconnected=True,paddesign=False):
        # gaussian electrode from sellmeiertests.fixeddomain() 12/29/20 project 7111
        #   fixeddomain(gaussfunc,L=100*129+50,Λ=100,name='electrode15mm')
        #   fixeddomain(gaussfunc,L=100*172+50,Λ=100,name='electrode20mm')
        electrode15mmsym2 = [1,1,1,1,1,1,1,1,1,-1,1,1,-1,1,1,-1,1,1,1,-1,1,1,-1,1,1,-1,1,1,1,1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,-1,1,1,1,-1,1,-1,1,1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,1,-1,1,1,1,-1,1,-1,1,1,1,-1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,1,1,1,1,-1,1,1,-1,1,1,-1,1,1,1,-1,1,1,-1,1,1,-1,1,1,1,1,1,1,1,1,1]
        electrode20mmsym2 = [1,1,1,1,1,1,1,1,1,-1,1,1,-1,1,1,-1,1,1,1,1,1,-1,1,1,-1,1,1,-1,1,1,1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,1,1,-1,1,1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,-1,1,1,1,-1,1,1,1,-1,1,-1,1,1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,1,-1,1,-1,1,1,1,-1,1,-1,1,1,1,-1,1,-1,1,1,1,-1,1,1,1,-1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,1,1,-1,1,1,1,1,1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,1,1,-1,1,1,1,-1,1,1,-1,1,1,-1,1,1,1,1,1,-1,1,1,-1,1,1,-1,1,1,1,1,1,1,1,1,1]
        flips = electrode20mmsym2 if 8==padcount else electrode15mmsym2 if 6==padcount else None
        # print('period,dc,gx,padcount,gapx,phase',period,dc,gx,padcount,gapx,phase)
        # add bars for all pads as single grating
        from grating import flipgratingbars
        barstarts,barends = flipgratingbars(flips,period)
        # grating(period,dc,gx,padcount,gapx,phase,apodize=apodize)
        def recenter(barstarts,barends,gx,padcount):
            Δx = gx*padcount/2 - (barends[-1]+barstarts[0])/2
            return [b+Δx for b in barstarts],[b+Δx for b in barends]
        barstarts,barends = recenter(barstarts,barends,gx,padcount)
        self.addgrating(barstarts,barends,padinputy,gy)
        self.info.averageperiod = averageperiod(barstarts,barends) # print('flipgrating averageperiod',self.info.averageperiod)
        # add each pad
        padstarts,padends = findpadboundarys(barstarts,barends,gx,padcount)
        padtext = makepadtext(self.parent.name,self.info.chipid,period,0.5,padcount)
        padtext[3] = 'FIX' # replace duty cycle with type
        self.addpads(padstarts,padends,padtext,padcount,padinputy,gx,gy,inputconnected,outputconnected,design=['circle',period] if paddesign else None) #None)
        self.savegrating(barstarts,barends)
        return self
    def addshortsubmount(self,period,dc,poledlength,poledstart=None,phase=0,padcount=10,padinputy=1000,gx=2500,gy=3000,gapx=0,inputconnected=False,outputconnected=True):
        # add bars for all pads as single grating
        assert 0==gapx, 'gap>0 not yet tested'
        poledstart = poledstart if poledstart is not None else (gx*padcount-poledlength)//2
        barstarts,barends = grating(period,dc,padx=poledlength,padcount=1,gapx=gapx,phase=phase,x0=poledstart)
        self.addgrating(barstarts,barends,padinputy,gy)
        self.info.averageperiod = averageperiod(barstarts,barends)
        # add each pad
        padstarts,padends = defaultpadboundarys(gx,padcount) #{i:gx*i for i in range(padcount)},{i:gx*(i+1) for i in range(padcount)} # print('padstarts,padends',padstarts,padends)
        padtext = makepadtext(self.parent.name,self.info.chipid,period,dc,padcount)
        padtext = padtext[:3] + [f'{poledlength/1000:.1f}mm']
        self.addpads(padstarts,padends,padtext,padcount,padinputy,gx,gy,inputconnected,outputconnected,design=['circle',period])
        self.savegrating(barstarts,barends)
        return self
    def addalternatingsubmount(self,period0=33,period1=123,dc0=0.5,dc1=0.5,repeatlength=0,label0='AG',label1='AG',padcount=10,padinputy=1000,gx=2500,gy=3000,gapx=0,inputconnected=False,outputconnected=True):
        # add bars for all pads as single grating
        repeats = (2 if 0==repeatlength else round(padcount*gx/repeatlength) )
        # print('repeats',repeats)
        from grating import alternatinggrating
        barstarts,barends = alternatinggrating(period1,period0,dc0,dc1,padcount,gx,repeats=repeats)
        self.addgrating(barstarts,barends,padinputy,gy)
        # add each pad
        padstarts,padends = findpadboundarys(barstarts,barends,gx,padcount)
        padtext0 = makepadtext(self.parent.name,self.info.chipid,period0,dc=dc0,padcount=padcount)
        padtext1 = makepadtext(self.parent.name,self.info.chipid,period1,dc=dc1,padcount=padcount)
        # padtext = padtext0[:3] + padtext1[2:3] + padtext0[4:-4] + padtext0[3:4] + padtext1[3:4] + [str(label0)] + [str(label1)]
        padtext = padtext1[:4] + [str(label0)] + padtext0[:4] + [str(label1)]
        self.addpads(padstarts,padends,padtext,padcount,padinputy,gx,gy,inputconnected,outputconnected)
        self.savegrating(barstarts,barends)
        return self
    def addinterleavedsubmount(self,period0,period1,label0,label1=None,padcount=10,padinputy=1000,gx=2500,gy=3000,gapx=0,inputconnected=False,outputconnected=True,
            smallestbar=1,overpole=0,breakupgapsize=0,apodize=None):
        label1 = label1 if label1 is not None else ('AI' if apodize else 'IL')
        from grating import interleavedgrating,shrinkbars,mergetouchingbars,dropsmallbars,breakupgaps,apodizedinterleavedgrating
        # add bars for all pads as single grating
        barstarts,barends = interleavedgrating(period1,period0,padcount,gx) if apodize is None else apodizedinterleavedgrating(period1,period0,padcount,gx,apodize=apodize)
        barstarts,barends = shrinkbars(barstarts,barends,dx=overpole)
        barstarts,barends = mergetouchingbars(barstarts,barends,tolerance=smallestbar)
        barstarts,barends = dropsmallbars(barstarts,barends,tolerance=smallestbar)
        barstarts,barends = breakupgaps(barstarts,barends,maxgap=breakupgapsize,barsize=smallestbar)
        self.addgrating(barstarts,barends,padinputy,gy)
        # add each pad
        padstarts,padends = findpadboundarys(barstarts,barends,gx,padcount)
        padtext0 = makepadtext(self.parent.name,self.info.chipid,period0,dc=0,padcount=padcount)
        padtext1 = makepadtext(self.parent.name,self.info.chipid,period1,dc=0,padcount=padcount)
        padtext = padtext0[:3] + padtext1[2:3] + padtext0[4:-3] + padtext1[2:3] + [str(label0)] + [str(label1)]
        self.addpads(padstarts,padends,padtext,padcount,padinputy,gx,gy,inputconnected,outputconnected)
        self.savegrating(barstarts,barends)
        return self
    def addphaseflipsubmount(self,period=123,n=7,label0='PF',padcount=10,padinputy=1000,gx=2500,gy=3000,gapx=0,inputconnected=False,outputconnected=True,apodize=None):
        # note - spectral peaks have unequal intensity 
        from grating import phaseflipgrating,apodizebars
        barstarts,barends = phaseflipgrating(period,n,padcount,gx) # add bars for all pads as single grating
        if apodize is not None:
            barstarts,barends = apodizebars(barstarts,barends,apodize=apodize)
        # barstarts,barends = mergetouchingbars(barstarts,barends,tolerance=smallestbar)
        # barstarts,barends = dropsmallbars(barstarts,barends,tolerance=smallestbar)
        self.addgrating(barstarts,barends,padinputy,gy)
        # add each pad
        padstarts,padends = findpadboundarys(barstarts,barends,gx,padcount)
        padtext0 = makepadtext(self.parent.name,self.info.chipid,period,dc=0,padcount=padcount)
        padtext1 = [f"N={n}"]
        padtext = padtext0[:3] + padtext1[2:3] + padtext0[4:-3] + padtext1[2:3] + [str(label0)]
        self.addpads(padstarts,padends,padtext,padcount,padinputy,gx,gy,inputconnected,outputconnected)
        self.savegrating(barstarts,barends)
        return self
    def addchirpsubmount(self,period0,period1,dc,padcount=4,padinputy=1000,gx=2500,gy=3000,gapx=0,
            inputconnected=False,outputconnected=True,apodize=None,opmin=None,opmax=None):
        # add bars for all pads as single grating
        # barstarts,barends = chirpgrating(period0,period1,dc,gx,padcount,gapx)
        from grating import kchirpgrating,apodizebars,invertbarsgaps,expandbars,dropsmallbars,mergetouchingbars
        barstarts,barends = kchirpgrating(period0,period1,dc,gx,padcount,gapx)
        def tri(x): return 1-2*abs(x-0.5)
        def trap(x): return np.minimum(1,2*tri(x))
        if apodize in ['trapezoid','inversetrapezoid']:
            barstarts,barends = apodizebars(barstarts,barends,trap)
        elif apodize in ['triangle','inversetriangle']:
            barstarts,barends = apodizebars(barstarts,barends,tri)
        else:
            assert 0, f'apodize function not defined: {apodize}'
        if apodize.startswith('inverse'):
            barstarts,barends = invertbarsgaps(barstarts,barends)
        if opmin is not None:
            if opmax is not None:
                def opfunc(b):
                    return -(opmin + (opmax-opmin)*b/(period0/2+period1/2))
                barstarts,barends = expandbars(barstarts,barends,op=opfunc) # op is negative to make bars smaller (opmin & opmax are positive)
            else:
                barstarts,barends = expandbars(barstarts,barends,op=-opmin)
        # barstarts,barends = dropsmallbars(barstarts,barends,0)
        barstarts,barends = dropsmallbars(barstarts,barends,self.finddefaultinfo('maskres'))
        # barstarts,barends = mergetouchingbars(barstarts,barends,0)
        barstarts,barends = mergetouchingbars(barstarts,barends,self.finddefaultinfo('maskres'))
        self.addgrating(barstarts,barends,padinputy,gy)
        self.info.averageperiod = averageperiod(barstarts,barends)
        self.info.firstperiod,self.info.lastperiod = barstarts[1]-barstarts[0],barends[-2]-barends[-1]
        # add each pad
        padstarts,padends = findpadboundarys(barstarts,barends,gx,padcount)
        padtext = makepadtext(self.parent.name,self.info.chipid,period0,dc,padcount)
        if not period0==period1:
            padtext = [self.parent.name,self.info.chipid,'%.2f/'%period0,'/%.2f'%period1]
        self.addpads(padstarts,padends,padtext,padcount,padinputy,gx,gy,inputconnected,outputconnected)
        self.savegrating(barstarts,barends)
        return self
    def adduagratingsubmount(self,expectedoverpole,padcount=4,padinputy=1000,gx=2500,gy=3000,gapx=0,inputconnected=False,outputconnected=True):
        # add bars for all pads as single grating
        file = {1:'uagrating.dat',2:'uagrating2.dat',3:'uagrating3.dat'}[1]
        barstarts,barends =  customgrating(file,expectedoverpole,minfeature=0.6)
        # barstarts,barends = uagrating(1,expectedoverpole,minfeature=0.6)
        self.addgrating(barstarts,barends,padinputy,gy)
        self.info.averageperiod = averageperiod(barstarts,barends)
        self.info.firstperiod,self.info.lastperiod = barstarts[1]-barstarts[0],barends[-2]-barends[-1]
        self.info.expectedoverpole = expectedoverpole
        # add each pad
        padstarts,padends = findpadboundarys(barstarts,barends,gx,padcount)
        # print('padstarts,padends',padstarts,padends)
        # padtext = makepadtext(self.parent.name,self.info.chipid,period0,dc,padcount)
        padtext = [self.parent.name,self.info.chipid,'OP%.1f'%expectedoverpole,'UA']
        self.addpads(padstarts,padends,padtext,padcount,padinputy,gx,gy,inputconnected,outputconnected)
        self.savegrating(barstarts,barends)
        return self
    def addcustomsubmount(self,filename,label='',expectedoverpole=0,padcount=4,padinputy=1000,gx=2500,gy=3000,gapx=0,inputconnected=False,outputconnected=True,centered=True,breakupgapsize=0,breakupgapbar=1,labels=[],minfeature=0.6):
        from grating import breakupgaps,shrinkbars,dropsmallbars
        def customgrating(file,expectedoverpole,minfeature):
            with open(file if file.endswith('.dat') else file+'.dat') as f:
                bars = [float(s) for s in f.readlines()]
            starts,ends = [a for a in bars[::2]],[b for b in bars[1::2]]
            starts,ends = shrinkbars(starts,ends,expectedoverpole)
            starts,ends = dropsmallbars(starts,ends,minfeature)
            return starts,ends
        # add bars for all pads as single grating
        barstarts,barends =  customgrating(filename,expectedoverpole,minfeature)
        if centered:
            bx = (padcount*gx - barends[-1])/2.
            barstarts,barends = [b+bx for b in barstarts],[b+bx for b in barends]
        if breakupgapsize:
            barstarts,barends = breakupgaps(barstarts,barends,maxgap=breakupgapsize,barsize=breakupgapbar)
        self.addgrating(barstarts,barends,padinputy,gy)
        self.info.averageperiod = averageperiod(barstarts,barends)
        self.info.firstperiod,self.info.lastperiod = barstarts[1]-barstarts[0],barends[-2]-barends[-1]
        self.info.expectedoverpole = expectedoverpole
        # add each pad
        padstarts,padends = defaultpadboundarys(gx,padcount) if inputconnected else findpadboundarys(barstarts,barends,gx,padcount) # print('padstarts,padends',padstarts,padends)
        # print('inputconnected',inputconnected,'padstarts,padends',padstarts,padends,'findpadboundarys',findpadboundarys(barstarts,barends,gx))
        # padtext = makepadtext(self.parent.name,self.info.chipid,period0,dc,padcount)
        padtext = [self.parent.name,self.info.chipid,'OP%.1f'%expectedoverpole]
        padtext += labels if labels else [label]
        self.addpads(padstarts,padends,padtext,padcount,padinputy,gx,gy,inputconnected,outputconnected)
        self.savegrating(barstarts,barends)
        return self
    def addshortwidebulksubmount(self,period,dc):
        xpoled,xsubmount,xbar,xchip,ypoled,ysubmount = 2000,20000,100,10000,8000,11000
        self.addsubmount(period=7.907,dc=dc,padcount=1,padinputy=0.5*(ysubmount-ypoled),
            gx=xpoled,gy=ypoled,gapx=0,inputconnected=0,outputconnected=0,paddesign=0,padtext=[f'{100*dc:g}%'])
        self.addtext(f"{period:g}", x=2000,y=1000, margin=100, scale=5)
        self.translate(0.5*(xsubmount-xpoled),0)
        self.addrect(0.5*(xsubmount-xchip)-0.5*xbar,0,xbar,ysubmount)
        self.addrect(0.5*(xsubmount+xchip)-0.5*xbar,0,xbar,ysubmount)
        return self
def roundoffcorners(ps,r=20):
    def rightangle(a,b,c):
        if np.array_equal(a,b) or np.array_equal(b,c): return False # numpy.allclose if tolerance
        #v np.dot(a-b,c-b) # any right angle
        v = (a-b)*(c-b)   # must be x/y aligned right angle # print v, False==any(v)
        return False==any(v)
    def quartercircle(start,end): # assumes center is (0,0)
        a,c,r = start,end,np.linalg.norm(start)
        phia,phic = arctan2(a[1],a[0]),arctan2(c[1],c[0])
        if phic-phia>pi: phic-=2*pi
        if phic-phia<-pi: phic+=2*pi
        return [np.array([r*cos(phi),r*sin(phi)]) for phi in np.linspace(phia,phic,11)]
    v = [np.array(list(p)) for p in ps]
    vv = [v[0]]
    for n in range(1,len(v)-1):
        a,b,c = v[n-1],v[n],v[n+1]
        if rightangle(a,b,c):
            aa,cc = (a-b),(c-b)
            if not (np.linalg.norm(aa)>=r and np.linalg.norm(cc)>=r):
                raise ValueError('radius too big:'+str(r)+' '+str(np.linalg.norm(aa))+' '+str(np.linalg.norm(cc)))
            # if not (np.linalg.norm(aa)>2*r and np.linalg.norm(cc)>2*r):
            #     import warnings
            #     warnings.warn('radius too big:'+str(r)+' '+str(np.linalg.norm(aa))+' '+str(np.linalg.norm(cc)))
            # assert np.linalg.norm(aa)>=r and np.linalg.norm(cc)>=r, 'radius too big:'+str(r)+' '+str(np.linalg.norm(aa))+' '+str(np.linalg.norm(cc)) # not sure why I set this to 2*r before, seems like r is ok
            aa,cc = aa*r/np.linalg.norm(aa),cc*r/np.linalg.norm(cc)
            b0 = aa+cc # b+b0 is center of circle
            vv += [b+b0+bi for bi in quartercircle(start=aa-b0,end=cc-b0)]
        else:
            vv += [b]
    vv += [v[-1]]
    return [tuple(vi) for vi in vv]
def polingdc(period): # valid on Oct17B and Oct17D LN poling masks
    # replace with max(0.5-1.*overpole/period, 0.7/period) ?
    if period>11:
        overpole = 3
        return 0.5-1.*overpole/period
    if period>7.4: # 12% dc for 8um, 920+1550→580
        return 1.0/period
    return 0.7/period # 12% dc for 6.05um, 1064→532, 10% dc for 7.25um, 810→405
def mglnpolingdc(period): # valid on Dec17,Dec17B submount mask
    if period>4: return 0.25
    return 0.6/period
def ktppolingdc(period):
    if period>20: return 0.50
    if period>10: return 0.45 # 0.5um overpole
    if period>5:  return 0.40 # 0.5um overpole
    return 1.0/period
def savespreadsheetsummary(d,folder,maskname,skip=('chip','chipmap','chipenable'),verbose=False):
    def valid(k):
        return k.startswith('chip') and hasattr(d[k],'__len__') and d['rw']==len(d[k]) and k not in skip
    def vstr(v):
        return ' '.join(map(str,v)) if isinstance(v,(list, tuple)) else str(v) if v is not None else ' '
    assert d['rw']==len(d['chipid'])
    title = ','.join([k[4:] for k,v in d.items() if valid(k)])
    if verbose: print(title)
    print(title, file=open(f'{folder}{maskname}.csv','w'))
    for i in range(d['rw']):
        # line = ','.join([','.join([str(k),vstr(v[i])]) for k,v in d.items() if k.startswith('chip') and k not in ['chip','chipmap','chipnum','chipxy','chipenable','chipdx']])
        # line = ','.join([','.join([str(k),vstr(v[i])]) for k,v in d.items() if k.startswith('chip') and hasattr(d[k],'__len__') and d['rw']==len(d[k])])
        line = ','.join([vstr(v[i]) for k,v in d.items() if valid(k)])
        if verbose: print(line)
        print(line, file=open(f'{folder}{maskname}.csv','a'))
def standardtwolayerlithomask(draftfolder,chipnum=None,chipmap=False):
    maskname,polingmaskname = 'Dec99B','Dec99C'
    maskinfo = [maskname,'LN waveguide and poling litho mask','EM 6200','5" mask','1.0um process']
    waferradius,flatradius,workingradius = 38100,36400,34000 # 34000 o-ring radius, 31000 tfln workingradius
    folder = 'masks 2023/'+maskname+'/'+draftfolder+'/'
    os.makedirs(folder,exist_ok=True)
    mask = Element(maskname,layer='MASK')
    mask.info.font = r'Interstate-Bold.ttf'
    mask.info.minfeature = 1.0
    chipname = ['' for _ in range(19)]
    chiptype = ['BLOCK']*4 + ['WDM']*16
    chipname = [c+cc for (c,cc) in zip(chiptype,chipname)]
    mask.info.rows = rw = len(chipname)
    mask.info.chiplength = dx = 62000
    mask.info.chipwidth = dy = 3000
    chipid = [None for i in range(rw)]
    chipenable = [True for i in range(rw)]
    if chipnum is not None:
        chipenable = [i==chipnum for i in range(rw)]
    if isinstance(chipnum, (list, tuple)):
        chipenable = [True for i in range(rw)] if 0==len(chipnum) else [i in chipnum for i in range(rw)]
    def chipxy(i):
        dys = [dy for i in range(rw)]
        dx = mask.info.chiplength
        return (-dx/2, sum(dys[:i]) - sum(dys)/2)
    for i in range(rw):
        chip = Chip(parent=mask)
        chipid[i] = chip.info.chipid
        # chip.info.chiplength = dx
        # chip.info.chiplength = dy
        if chipenable[i]:
            if chiptype[i]=='BLOCK':
                chip.addmetriconblockedchip() # leave 4 chip gap at bottom of wafer for metricon witness
            elif chiptype[i]=='WDM':
                chip.addmetriconblockedchip()
            chip.translate(*chipxy(i))
            if chipmap:
                chip.addnotescircle(waferradius,bb=chip.boundingbox()).addnotescircle(workingradius,bb=chip.boundingbox())
                chip.addnotesframe(margin=(1000,100))
                chipfolder = folder+'chipmaps/' if scalemag>1 else folder+'chips/'
                os.makedirs(chipfolder,exist_ok=True)
                chip.savemask(svg=True,txt=True,filename=f'{chipfolder}{maskname}-{chipid[i]}',png=savepng)
    if chipmap or not all(chipenable):
        return
    mask.addstanfordfiducialchip(x=0,y=+workingradius-2500,rotation=0)
    mask.addstanfordfiducialchip(x=0,y=-workingradius+2500,rotation=0)
    if not dev:
        mask.polingarea(overpole=3,maxarea=18*30) # make sure polingarea calculated before poling windows are added
    for layer in ['MASK']:
        e = Element('window',parent=mask,layer=layer)
        e.addrect(-waferradius,-9000,1000,2*9000).addrect(waferradius-1000,-9000,1000,2*9000)
    mask.addnotescircle(waferradius,bb=mask.boundingbox()).addnotescircle(workingradius,bb=mask.boundingbox()) # actual diameter = 76.2mm, old working diameter = 70mm, poling diameter = 68mm
    mask.roundoff(res=0.1,decimal=False)
    # mask.checkpolygonvalidity()
    for i in range(rw):
        def vstr(v): return ' '.join(map(str,v)) if isinstance(v,(list, tuple)) else str(v)
        line = ','.join([','.join([str(k),vstr(v[i])]) for k,v in locals().items() if k.startswith('chip') and k not in ['chip','chipmap','chipnum','chipxy','chipenable','chipdx']])
        # print(line)
        print(line, file=open(f'{folder}{maskname}.csv','a' if i else 'w'))
    maskfiles = mask.savemask(maskname,layers=['MASK','POLING'],layernames=[maskname,polingmaskname],
        svg=True,txt=True,png=savepng,folder=folder)
    for line in ['']+maskinfo+maskfiles+['']:
        print(line)
def gen(maskfunc,mag=10): # decorator: gen(func)(args) or @gen/ndef func
    assert callable(maskfunc), 'usage: gen(maskfunc)(args)'
    def wrapper(draftfolder, *args, **kwargs):
        global dev,periodmag,scalemag,savepng
        savepng = 1
        dev,periodmag,scalemag = 1,mag,mag; maskfunc(draftfolder,chipmap=True)
        dev,periodmag,scalemag = 1,  1,  1; maskfunc(draftfolder,chipmap=True)
        dev,periodmag,scalemag = 0,  1,  1; maskfunc(draftfolder)
        dev,periodmag,scalemag = 1,mag,  1; maskfunc(draftfolder)
    return wrapper
def standardonelayerlithomask(draftfolder,chipnum=None,chipmap=False):
    maskname = 'Dec99A'
    maskinfo = [maskname,'LN waveguide litho mask','EM 6200','5" mask','1.0um process']
    waferradius,flatradius,workingradius = 38100,36400,34000 # 34000 o-ring radius, 31000 tfln workingradius
    folder = 'masks 2023/'+maskname+'/'+draftfolder+'/'
    os.makedirs(folder,exist_ok=True)
    mask = Element(maskname,layer='MASK')
    mask.info.taperlength = 1500
    mask.info.font = r'Interstate-Bold.ttf'
    mask.info.minfeature = 1.0
    mask.info.rows = rw = 9
    mask.info.chiplength = dx = 62000
    mask.info.chipwidth = dy = 2000
    print(mask)
    chipname = ['WG' for _ in range(rw)]
    chipid = [chipidtext(i,rw) for i in range(rw)]
    def chipxy(i):
        dys = [dy for i in range(rw)]
        return (-dx/2, sum(dys[:i]) - sum(dys)/2)
    for i in range(rw):
        chip = Chip(parent=mask,id=chipid[i])
        if isinstance(chipmap,list) and not i in chipmap:
            continue
        if chipname[i]=='BLOCK':
            chip.addmetriconblockedchip() # leave 4 chip gap at bottom of wafer for metricon witness
        elif chipname[i]=='WG':
            chip.addwaveguidechip()
        chip.translate(*chipxy(i))
        if chipmap:
            chip.addnotescircle(waferradius,bb=chip.boundingbox()).addnotescircle(workingradius,bb=chip.boundingbox())
            chip.addnotesframe(margin=(1000,100))
            chipfolder = folder+'chipmaps/' if scalemag>1 else folder+'chips/'
            os.makedirs(chipfolder,exist_ok=True)
            chip.savemask(svg=True,txt=True,filename=f'{chipfolder}{maskname}-{chipid[i]}',png=savepng)
    if chipmap:
        return
    mask.addstanfordfiducialchip(x=0,y=+workingradius-2500,rotation=0)
    mask.addstanfordfiducialchip(x=0,y=-workingradius+2500,rotation=0)
    for layer in ['MASK']:
        e = Element('window',parent=mask,layer=layer)
        e.addrect(-waferradius,-9000,1000,2*9000).addrect(waferradius-1000,-9000,1000,2*9000)
    mask.addnotescircle(waferradius,bb=mask.boundingbox()).addnotescircle(workingradius,bb=mask.boundingbox()) # actual diameter = 76.2mm, old working diameter = 70mm, poling diameter = 68mm
    mask.roundoff(res=0.1,decimal=False)
    # mask.checkpolygonvalidity()
    for i in range(rw):
        def vstr(v): return ' '.join(map(str,v)) if isinstance(v,(list, tuple)) else str(v)
        line = ','.join([','.join([str(k),vstr(v[i])]) for k,v in locals().items() if k.startswith('chip') and k not in ['chip','chipmap','chipnum','chipxy','chipenable','chipdx']])
        # print(line)
        print(line, file=open(f'{folder}{maskname}.csv','a' if i else 'w'))
    maskfiles = mask.savemask(maskname,layers=['MASK'],layernames=[maskname],
        svg=True,txt=True,png=savepng,folder=folder)
    print('maskfiles',maskfiles)
    for line in ['']+maskinfo+maskfiles+['']:
        print(line)
    return mask

if __name__ == '__main__':
    # standardonelayerlithomask('draft 1')
    # standardonelayerlithomask('draft 1',chipmap=[5])
    # print(standardonelayerlithomask('draft 1')==standardonelayerlithomask('draft 2'))
    # gen(standardonelayerlithomask)('draft 1')
    standardtwolayerlithomask('draft 1')
    # standardtwolayerlithomask('draft 1',chipmap=True)
    # gen(standardtwolayerlithomask)('draft 2')


    import subprocess
    subprocess.Popen(['C:/Program Files/Common Files/eDrawings2020/eDrawings.exe', 'C:/py/mask/mask.dxf'])
    
