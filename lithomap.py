#!/usr/bin/env python
#  -*- coding: utf-8 -*-
from __future__ import print_function,division,unicode_literals
import numpy as np
from numpy import pi,sqrt,sin,cos,tan,arcsin,arccos,arctan,arctan2,log,exp,floor,ceil
import os,pickle
from decimal import Decimal
from geometry import chipidtext,rotatecurve,curvesboundingbox,rectsboundingbox,cropcurves,autocadcolor,compresscurve,plotcurves
from geometry import textcurves,scaletext
from geometry import Bunch,PPfloat,OrderedDictFromCSV

dev = True          # development version in which elements can be modified for the purpose of creating mask maps and chip maps
periodmag =  20     # period magnification for dev
scalemag = 1        # chip aspect ratio for dev
savepng = False     # enabling png file creation is slow

class Note:
    def __init__(self,x,y,dy=25,**kwargs):
        self.x,self.y,self.dy = x,y,dy
        self.note = {k:kwargs[k] for k in kwargs}
    def __eq__(self,other):
        if isinstance(other,Note):
            return tuple(self)==tuple(other)
        return False
    def __lt__(self, other):
        return tuple(self) < tuple(other)
    def __iter__(self):# defined so that tuple(self) can be called, so attributes can be sorted, so Notes can be compared (i.e n1==n2)
        for a in ['x','y','dy']:
            yield getattr(self,a)
        yield sorted(tuple(self.note.items()))
    def text(self):
        return ' '.join([f'{k}={v}' for k,v in self.note.items()])
    def __str__(self):
        return f"({self.x},{self.y}) {self.text()}"
    def __repr__(self):
        return repr(tuple(self))
class Element:
    def __init__(self,name='elem',layer=None,parent=None,x=None,y=None):
        self.info,self.rects,self.polys,self.elems,self.notes = Bunch(),[],[],[],[]
        self.name,self.parent = name,None
        self.x,self.y = 0,0
        if parent:
            self.x,self.y = parent.x,parent.y
            self.layer = layer if layer is not None else parent.layer
            parent.addelem(self)
        else:
            self.info.maskname = name
            assert layer is not None
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
            return tuple(self)==tuple(other)
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
    def addelem(self,e,guidespacing=None):
        e.parent = self
        name,e.name = e.name,self.nextname(e.name) # change name to a unique name # print(e.name)
        self.elems += [e]
        if name.endswith('chip') and not isinstance(e,Chip):
            print("warning, Element('chip') is deprecated")
            n = self.info.chipcount = e.info.chipnumber = getattr(self.info,'chipcount',0) + 1
            nx,ny = 0,n-1
            if hasattr(self.info,'rows'):
                nx,ny = (n-1)//self.info.rows,(n-1)%self.info.rows
                self.info.columns = max(getattr(self.info,'columns',0),nx+1)
            e.info.chipid = str(ny+1)+'ABCDEFGHIJ'[nx]
        if name.endswith('group'):
            e.info.groupnumber = self.info.groupcount = getattr(self.info,'groupcount',0) + 1
            if self.info.groupcount>1: self.y += self.info.groupspacing
            # if 2==e.info.groupnumber: # show group spacing note between g1.last and g2.1
            #     self.addnote(self.x+1000,self.y-self.info.groupspacing/2,separation=self.info.groupspacing)
            e.x,e.y = self.x,self.y
        if name.endswith('guide'):
            self.info.guidecount = getattr(self.info,'guidecount',0) + 1
            guidespacing = guidespacing if guidespacing is not None else self.finddefaultinfo('guidespacing') # self.parent.info.guidespacing
            if self.info.guidecount>1:
                self.y += guidespacing
                self.parent.y = self.y
                # both group and chip are updated with y position of current waveguide
                # chip.y should probably be updated when adding wide guides (spltter,mz) but isn't yet
            if 2==self.info.guidecount and 1==self.info.groupnumber: # show guide spacing note between g1.1 and g1.2
                self.addnote(self.x+1000,self.y-guidespacing/2,separation=guidespacing)
            e.info.guidenumber = str(self.info.groupnumber) + '.' + str(self.info.guidecount)
            e.x,e.y = self.x,self.y
        if hasattr(self,'width'): e.width = self.width  # for keeping track of current waveguide width
        return self
    def subrects(self):
        return self.rects + [r for e in self.elems for r in e.subrects()]
    def subpolys(self):
        return self.polys + [c for e in self.elems for c in e.subpolys()]
    def subelems(self):
        return self.elems + [ee for e in self.elems for ee in e.subelems()]
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
    def addnote(self,x=None,y=None,dy=25,**kwargs):
        e = Element(self.name+'-note',layer='NOTES',parent=self)
        e.notes += [Note(x if x is not None else self.x, y if y is not None else self.y, dy, **kwargs)]
        return self
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
        self.notes = [Note(xscale*note.x,yscale*note.y,yscale*note.dy,**note.note) for note in self.notes]
        self.elems = [e.scale(xscale,yscale) for e in self.elems]
        return self
    def translate(self,x0=0,y0=0):
        self.polys = [[(x0+x,y0+y) for x,y in c] for c in self.polys]
        self.rects = [(x0+x,y0+y,dx,dy) for x,y,dx,dy in self.rects]
        self.notes = [Note(x0+note.x,y0+note.y,note.dy,**note.note) for note in self.notes]
        self.elems = [e.translate(x0,y0) for e in self.elems]
        return self
    def rotate(self,angle,x0=0,y0=0):
        def recttopoly(x,y,dx,dy): return [(x,y),(x+dx,y),(x+dx,y+dy),(x,y+dy),(x,y)]
        self.polys = [rotatecurve(c,angle,x0,y0) for c in self.polys] + [rotatecurve(recttopoly(*r),angle,x0,y0) for r in self.rects]
        self.rects = []
        self.notes = [] # rotated notes not implemented
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
    def addoversizemetrics(self,width,xmetric,ymetric,i0=11,dx=10,nx=8,ny=6):
        for i,w in enumerate(np.linspace(width-1,width+1,i0)):
            self.addoversizemetric(width=width,oversize=w,x=xmetric+(i-i0/2)*2*dx*nx,y=ymetric,dx=dx,nx=nx,ny=ny)
            self.addtext(f"{w:.1f}",xmetric+(i-i0/2)*2*dx*nx+dx*nx+10,ymetric,scale=0.2)
    def addoversizemetric(self,width,oversize,x,y,dx=10,nx=8,ny=6):
        dy,ds = width,2*width-oversize
        for i in range(nx):
            for j in range(ny):
                if 0==(i+j)%2:
                    self.addrect(x+i*dx,y-j*dy-dy-0.5*(ds-dy),dx,ds)
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
    def addinsettext(self,pcc,s,x0=0,y0=0,fitx=0,fity=0,margin=0,scale=1,fitcenter=True,vertical=False,scaleifdev=True):
        f = Font(self.finddefaultinfo('font'),size=128,screencoordinates=True)
        cc = f.textcurves(s,verticallayout=vertical)
        cc = [upsamplecurve(c) for c in cc]
        cc = scaletext(cc,x0,y0,fitx,fity,margin,scale,fitcenter,dev=(dev and scaleifdev),scalemag=sqrt(scalemag))
        self.addpolys(eliminateholes(pcc+cc))
        return self
    def boundingbox(self): # curvesboundingbox
        pbb = curvesboundingbox(self.subpolys())
        rbb = rectsboundingbox(self.subrects())
        return rectsboundingbox([pbb,rbb])
    def polingarea(self,overpole,maxarea=None):
        self.info.polingarea = self.area(layer='POLING')
        self.info.rectpolingareawithoutoverpole = self.rectarea(layer='POLING',maxarea=maxarea)
        self.info.overpole = overpole
        self.info.rectpolingareawithoverpole = self.rectarea(layer='POLING',maxarea=maxarea,overpole=self.info.overpole)
        print('self.info.polingarea',self.info.polingarea)
        print('self.info.rectpolingareawithoutoverpole',self.info.rectpolingareawithoutoverpole)
        print('self.info.overpole',self.info.overpole)
        print('self.info.rectpolingareawithoverpole',self.info.rectpolingareawithoverpole)
    def checkpolygonvalidity(self):
        print('checking polygon validity...')
        def check(c):
            if not issimplepolygon(c):
                plotcurvelist([c])
                return False
            return True
        bs = [check(c) for c in self.subpolys()]
        assert all(bs), 'invalid polygon found'
        print('all polygons valid')
    def plot(self,screencoordinates=True,scale=0.5):
        # plotcurves(cc,screencoordinates=True)
        import matplotlib
        import matplotlib.pyplot as plt
        def recttopoly(x,y,dx,dy):
            return [(x,y),(x+dx,y),(x+dx,y+dy),(x,y+dy),(x,y)]
        def curvestocurve(pss):
            return [p for ps in pss for p in list(ps)+[(float('nan'),float('nan'))] ]
        pss = self.subpolys() + [recttopoly(*r) for r in self.subrects()]
        xs,ys = np.array(list(zip(*curvestocurve(pss))))
        plt.plot(xs,ys,'darkred'); plt.ylabel('y (µm)'); plt.xlabel('x (µm)')
        if screencoordinates: plt.gca().invert_yaxis()
        plt.gca().set_aspect(1)
        fig = plt.gcf()
        sx,sy = scale if hasattr(scale,'__len__') else (scale,scale)
        fig.set_size_inches((fig.get_size_inches()[0]*sx, fig.get_size_inches()[1]*sy))
        for note in self.subnotes():
            plt.text(note.x,note.y,note.text())
        plt.show()
        return self
    def savemask(self,filename,txt=False,layers=[],layernames=[],svg=True,png=True,verbose=True,folder=None):
        import os,shutil,glob
        if folder:
            folder += '' if '/'==folder[-1] else '/'
            os.makedirs(folder,exist_ok=True)
            filename = folder+filename
            layernames = [folder+name for name in layernames]
        if verbose: print('  saving.')
        if dev and not 1==scalemag: self.scale(1,scalemag)
        self.savedxfwithlayers(filename,verbose=verbose)
        # os.popen('copy "'+filename+'.dxf" mask.dxf')
        shutil.copy(filename+'.dxf','mask.dxf')
        if svg:
            self.savedxfwithlayers(filename,svg=1,verbose=verbose)
            # os.popen('copy "'+filename+'.svg" mask.svg')
            # shutil.copy(filename+'.svg','mask.svg')
        if png and svg:
            if verbose: print('  saving "'+filename+'.png"...')
            # os.popen('"C:\\Program Files\\Inkscape\\inkscape.com" '+filename+'.svg --export-png='+filename+'.png')
            svg2png(filename+'.svg')
        if dev and not 1==scalemag: self.scale(1,1./scalemag)
        if txt:
            with open(filename+'.txt','w',encoding='utf-8') as file: file.write(str(self))
            if verbose: print('  "'+filename+'.txt" saved.')
        if layers and not dev:
            self.pickle(filename)
            for layer,name in zip(layers,layernames):
                self.savedxfwithlayers(filename=name+layer,singlelayertosave=layer,verbose=verbose)
                # "\\ADVRSVR2\AdvR Lab Shared Space\015 Litho and Electrode Masks\2022\Feb22C\draft1\Feb22CMASK.dxf"
        if folder and not dev:
            maskfolder = '/'.join(filename.split('/')[:-2])
            for f in glob.glob('*.py'):
                shutil.copy(f,maskfolder)
        return [f'"//ADVRSVR2/AdvR Lab Shared Space/015 Litho and Electrode Masks/2022/{name+layer}.dxf"'.replace('/','\\') for layer,name in zip(layers,layernames)]
    def savedxfwithlayers(self,filename='',singlelayertosave='',svg=False,svgdebug=False,verbose=False,nomodify=True):
        if not filename: filename = 'mask'
        if svg:
            filename = filename.replace('.svg','')+'.svg'
            svgscale = 0.05 # 0.05 is decent for converting svg to png at 96dpi
            svgcompressangle = 179 # 170 = ~2x, 135 = ~3x compression
            svggrid = 0.1 # um (only for x if not scalemag==1)
            svgdigits = max(0,1+int(-np.log10(svggrid*svgscale))) # eg. svgdigits=3 for svggrid*svgscale=0.005
            if nomodify:
                def scale(x):
                    return PPfloat(svgscale*x,svggrid)
                    # return svgscale*x
                bx,by,bdx,bdy = bb = [scale(b) for b in self.boundingbox()]
            else:
                def scale(x):
                    return PPfloat(x,svggrid)
                self.scale(svgscale)
                bx,by,bdx,bdy = bb = self.boundingbox()
            sbb = ' '.join([str(int(b)) for b in bb]) # eg. sbb = '-500 -2000 1000 4000'
            style = 'font-family:Arial;font-weight:bold;' # style='font-size:30;font-family:Comic Sans MS, Arial;font-weight:bold;font-style:oblique;stroke:black;stroke-width:.1;fill:none;text-shadow: 2px 2px;'
            import svgwrite
            dwg = svgwrite.Drawing(filename, viewBox=sbb, text_anchor='middle', style=style, debug=svgdebug)#, profile='tiny')
            dwg.add(dwg.rect(insert=(bx,by), size=(bdx,bdy), fill='white')) # white background
            if svgdebug:
                gm = dwg.add(dwg.g(id=self.name))
                gm.add(dwg.text(self.name,insert=(0,-1000*svgscale*scalemag),font_size=100*svgscale*scalemag,fill='darkred'))
                gm.add(dwg.text('xxx',insert=(0,350*svgscale*scalemag),font_size=100*svgscale*scalemag,text_anchor='end',fill='darkred'))
                gm.add(dwg.text('(0,-2000)',insert=(0,-2000*svgscale*scalemag),font_size=str(100*svgscale*scalemag)+'px',fill='black'))
                gm.add(dwg.rect(insert=(bx,by), size=(0-bx,bdy), opacity=0.1, stroke='darkred', stroke_width=30*svgscale))
                gm.add(dwg.line((bx,by), (bx+bdx,by+bdy), stroke=svgwrite.rgb(20,0,0,'%'), stroke_width=30*svgscale))
        else:
            filename = filename.rstrip('.dxf')+'.dxf'
            import ezdxf
            drawing = ezdxf.new('AC1015')
            space = drawing.modelspace()
            drawing.styles.new('arial', dxfattribs={'font':'arial.ttf'})
        # drawing.styles.new('custom1', dxfattribs={'font':'times.ttf'})
        # space.add_text("Text Style Example: Times New Roman", dxfattribs={'style':'custom1','height':1500,'color':7}).set_pos((-22000,22000), align='LEFT')
        # pink 11, grey pink 13, yellow-grey 53, blue 142, dark magenta 242, light grey 254, red 1, yellow 2, green 3, lt blue 4, blue 5, magenta 6, white 7, dark grey 8, grey 9, darker grey 250 # gold 42, dark yellow 52, green 62
        # dark blue 146, purple grey 207, dark red 16, orange 32, gold 42, dark tan 37, tan 33
        # http://sub-atomic.com/~moses/acadcolors.html (gone) # try http://gohtx.com/acadcolors.php (or https://stackoverflow.com/questions/43876109/  or  https://www.google.com/search?q=http://sub-atomic.com/~moses/acadcolors.html&hl=en&source=lnms&tbm=isch)
        # https://gist.github.com/kuka/f2a5a77b715531c7ae9beae8d5cbfded # acadcolors = [{"aci":0,"hex":"#000000","rgb":"rgb(0,0,0)"}, {"aci":1,"hex":"#FF0000","rgb":"rgb(255,0,0)"}, {"aci":2,"hex":"#FFFF00","rgb":"rgb(255,255,0)"}, {"aci":3,"hex":"#00FF00","rgb":"rgb(0,255,0)"}, {"aci":4,"hex":"#00FFFF","rgb":"rgb(0,255,255)"}, {"aci":5,"hex":"#0000FF","rgb":"rgb(0,0,255)"}, {"aci":6,"hex":"#FF00FF","rgb":"rgb(255,0,255)"}, {"aci":7,"hex":"#FFFFFF","rgb":"rgb(255,255,255)"}, {"aci":8,"hex":"#414141","rgb":"rgb(65,65,65)"}, {"aci":9,"hex":"#808080","rgb":"rgb(128,128,128)"}, {"aci":10,"hex":"#FF0000","rgb":"rgb(255,0,0)"}, {"aci":11,"hex":"#FFAAAA","rgb":"rgb(255,170,170)"}, {"aci":12,"hex":"#BD0000","rgb":"rgb(189,0,0)"}, {"aci":13,"hex":"#BD7E7E","rgb":"rgb(189,126,126)"}, {"aci":14,"hex":"#810000","rgb":"rgb(129,0,0)"}, {"aci":15,"hex":"#815656","rgb":"rgb(129,86,86)"}, {"aci":16,"hex":"#680000","rgb":"rgb(104,0,0)"}, {"aci":17,"hex":"#684545","rgb":"rgb(104,69,69)"}, {"aci":18,"hex":"#4F0000","rgb":"rgb(79,0,0)"}, {"aci":19,"hex":"#4F3535","rgb":"rgb(79,53,53)"}, {"aci":20,"hex":"#FF3F00","rgb":"rgb(255,63,0)"}, {"aci":21,"hex":"#FFBFAA","rgb":"rgb(255,191,170)"}, {"aci":22,"hex":"#BD2E00","rgb":"rgb(189,46,0)"}, {"aci":23,"hex":"#BD8D7E","rgb":"rgb(189,141,126)"}, {"aci":24,"hex":"#811F00","rgb":"rgb(129,31,0)"}, {"aci":25,"hex":"#816056","rgb":"rgb(129,96,86)"}, {"aci":26,"hex":"#681900","rgb":"rgb(104,25,0)"}, {"aci":27,"hex":"#684E45","rgb":"rgb(104,78,69)"}, {"aci":28,"hex":"#4F1300","rgb":"rgb(79,19,0)"}, {"aci":29,"hex":"#4F3B35","rgb":"rgb(79,59,53)"}, {"aci":30,"hex":"#FF7F00","rgb":"rgb(255,127,0)"}, {"aci":31,"hex":"#FFD4AA","rgb":"rgb(255,212,170)"}, {"aci":32,"hex":"#BD5E00","rgb":"rgb(189,94,0)"}, {"aci":33,"hex":"#BD9D7E","rgb":"rgb(189,157,126)"}, {"aci":34,"hex":"#814000","rgb":"rgb(129,64,0)"}, {"aci":35,"hex":"#816B56","rgb":"rgb(129,107,86)"}, {"aci":36,"hex":"#683400","rgb":"rgb(104,52,0)"}, {"aci":37,"hex":"#685645","rgb":"rgb(104,86,69)"}, {"aci":38,"hex":"#4F2700","rgb":"rgb(79,39,0)"}, {"aci":39,"hex":"#4F4235","rgb":"rgb(79,66,53)"}, {"aci":40,"hex":"#FFBF00","rgb":"rgb(255,191,0)"}, {"aci":41,"hex":"#FFEAAA","rgb":"rgb(255,234,170)"}, {"aci":42,"hex":"#BD8D00","rgb":"rgb(189,141,0)"}, {"aci":43,"hex":"#BDAD7E","rgb":"rgb(189,173,126)"}, {"aci":44,"hex":"#816000","rgb":"rgb(129,96,0)"}, {"aci":45,"hex":"#817656","rgb":"rgb(129,118,86)"}, {"aci":46,"hex":"#684E00","rgb":"rgb(104,78,0)"}, {"aci":47,"hex":"#685F45","rgb":"rgb(104,95,69)"}, {"aci":48,"hex":"#4F3B00","rgb":"rgb(79,59,0)"}, {"aci":49,"hex":"#4F4935","rgb":"rgb(79,73,53)"}, {"aci":50,"hex":"#FFFF00","rgb":"rgb(255,255,0)"}, {"aci":51,"hex":"#FFFFAA","rgb":"rgb(255,255,170)"}, {"aci":52,"hex":"#BDBD00","rgb":"rgb(189,189,0)"}, {"aci":53,"hex":"#BDBD7E","rgb":"rgb(189,189,126)"}, {"aci":54,"hex":"#818100","rgb":"rgb(129,129,0)"}, {"aci":55,"hex":"#818156","rgb":"rgb(129,129,86)"}, {"aci":56,"hex":"#686800","rgb":"rgb(104,104,0)"}, {"aci":57,"hex":"#686845","rgb":"rgb(104,104,69)"}, {"aci":58,"hex":"#4F4F00","rgb":"rgb(79,79,0)"}, {"aci":59,"hex":"#4F4F35","rgb":"rgb(79,79,53)"}, {"aci":60,"hex":"#BFFF00","rgb":"rgb(191,255,0)"}, {"aci":61,"hex":"#EAFFAA","rgb":"rgb(234,255,170)"}, {"aci":62,"hex":"#8DBD00","rgb":"rgb(141,189,0)"}, {"aci":63,"hex":"#ADBD7E","rgb":"rgb(173,189,126)"}, {"aci":64,"hex":"#608100","rgb":"rgb(96,129,0)"}, {"aci":65,"hex":"#768156","rgb":"rgb(118,129,86)"}, {"aci":66,"hex":"#4E6800","rgb":"rgb(78,104,0)"}, {"aci":67,"hex":"#5F6845","rgb":"rgb(95,104,69)"}, {"aci":68,"hex":"#3B4F00","rgb":"rgb(59,79,0)"}, {"aci":69,"hex":"#494F35","rgb":"rgb(73,79,53)"}, {"aci":70,"hex":"#7FFF00","rgb":"rgb(127,255,0)"}, {"aci":71,"hex":"#D4FFAA","rgb":"rgb(212,255,170)"}, {"aci":72,"hex":"#5EBD00","rgb":"rgb(94,189,0)"}, {"aci":73,"hex":"#9DBD7E","rgb":"rgb(157,189,126)"}, {"aci":74,"hex":"#408100","rgb":"rgb(64,129,0)"}, {"aci":75,"hex":"#6B8156","rgb":"rgb(107,129,86)"}, {"aci":76,"hex":"#346800","rgb":"rgb(52,104,0)"}, {"aci":77,"hex":"#566845","rgb":"rgb(86,104,69)"}, {"aci":78,"hex":"#274F00","rgb":"rgb(39,79,0)"}, {"aci":79,"hex":"#424F35","rgb":"rgb(66,79,53)"}, {"aci":80,"hex":"#3FFF00","rgb":"rgb(63,255,0)"}, {"aci":81,"hex":"#BFFFAA","rgb":"rgb(191,255,170)"}, {"aci":82,"hex":"#2EBD00","rgb":"rgb(46,189,0)"}, {"aci":83,"hex":"#8DBD7E","rgb":"rgb(141,189,126)"}, {"aci":84,"hex":"#1F8100","rgb":"rgb(31,129,0)"}, {"aci":85,"hex":"#608156","rgb":"rgb(96,129,86)"}, {"aci":86,"hex":"#196800","rgb":"rgb(25,104,0)"}, {"aci":87,"hex":"#4E6845","rgb":"rgb(78,104,69)"}, {"aci":88,"hex":"#134F00","rgb":"rgb(19,79,0)"}, {"aci":89,"hex":"#3B4F35","rgb":"rgb(59,79,53)"}, {"aci":90,"hex":"#00FF00","rgb":"rgb(0,255,0)"}, {"aci":91,"hex":"#AAFFAA","rgb":"rgb(170,255,170)"}, {"aci":92,"hex":"#00BD00","rgb":"rgb(0,189,0)"}, {"aci":93,"hex":"#7EBD7E","rgb":"rgb(126,189,126)"}, {"aci":94,"hex":"#008100","rgb":"rgb(0,129,0)"}, {"aci":95,"hex":"#568156","rgb":"rgb(86,129,86)"}, {"aci":96,"hex":"#006800","rgb":"rgb(0,104,0)"}, {"aci":97,"hex":"#456845","rgb":"rgb(69,104,69)"}, {"aci":98,"hex":"#004F00","rgb":"rgb(0,79,0)"}, {"aci":99,"hex":"#354F35","rgb":"rgb(53,79,53)"}, {"aci":100,"hex":"#00FF3F","rgb":"rgb(0,255,63)"}, {"aci":101,"hex":"#AAFFBF","rgb":"rgb(170,255,191)"}, {"aci":102,"hex":"#00BD2E","rgb":"rgb(0,189,46)"}, {"aci":103,"hex":"#7EBD8D","rgb":"rgb(126,189,141)"}, {"aci":104,"hex":"#00811F","rgb":"rgb(0,129,31)"}, {"aci":105,"hex":"#568160","rgb":"rgb(86,129,96)"}, {"aci":106,"hex":"#006819","rgb":"rgb(0,104,25)"}, {"aci":107,"hex":"#45684E","rgb":"rgb(69,104,78)"}, {"aci":108,"hex":"#004F13","rgb":"rgb(0,79,19)"}, {"aci":109,"hex":"#354F3B","rgb":"rgb(53,79,59)"}, {"aci":110,"hex":"#00FF7F","rgb":"rgb(0,255,127)"}, {"aci":111,"hex":"#AAFFD4","rgb":"rgb(170,255,212)"}, {"aci":112,"hex":"#00BD5E","rgb":"rgb(0,189,94)"}, {"aci":113,"hex":"#7EBD9D","rgb":"rgb(126,189,157)"}, {"aci":114,"hex":"#008140","rgb":"rgb(0,129,64)"}, {"aci":115,"hex":"#56816B","rgb":"rgb(86,129,107)"}, {"aci":116,"hex":"#006834","rgb":"rgb(0,104,52)"}, {"aci":117,"hex":"#456856","rgb":"rgb(69,104,86)"}, {"aci":118,"hex":"#004F27","rgb":"rgb(0,79,39)"}, {"aci":119,"hex":"#354F42","rgb":"rgb(53,79,66)"}, {"aci":120,"hex":"#00FFBF","rgb":"rgb(0,255,191)"}, {"aci":121,"hex":"#AAFFEA","rgb":"rgb(170,255,234)"}, {"aci":122,"hex":"#00BD8D","rgb":"rgb(0,189,141)"}, {"aci":123,"hex":"#7EBDAD","rgb":"rgb(126,189,173)"}, {"aci":124,"hex":"#008160","rgb":"rgb(0,129,96)"}, {"aci":125,"hex":"#568176","rgb":"rgb(86,129,118)"}, {"aci":126,"hex":"#00684E","rgb":"rgb(0,104,78)"}, {"aci":127,"hex":"#45685F","rgb":"rgb(69,104,95)"}, {"aci":128,"hex":"#004F3B","rgb":"rgb(0,79,59)"}, {"aci":129,"hex":"#354F49","rgb":"rgb(53,79,73)"}, {"aci":130,"hex":"#00FFFF","rgb":"rgb(0,255,255)"}, {"aci":131,"hex":"#AAFFFF","rgb":"rgb(170,255,255)"}, {"aci":132,"hex":"#00BDBD","rgb":"rgb(0,189,189)"}, {"aci":133,"hex":"#7EBDBD","rgb":"rgb(126,189,189)"}, {"aci":134,"hex":"#008181","rgb":"rgb(0,129,129)"}, {"aci":135,"hex":"#568181","rgb":"rgb(86,129,129)"}, {"aci":136,"hex":"#006868","rgb":"rgb(0,104,104)"}, {"aci":137,"hex":"#456868","rgb":"rgb(69,104,104)"}, {"aci":138,"hex":"#004F4F","rgb":"rgb(0,79,79)"}, {"aci":139,"hex":"#354F4F","rgb":"rgb(53,79,79)"}, {"aci":140,"hex":"#00BFFF","rgb":"rgb(0,191,255)"}, {"aci":141,"hex":"#AAEAFF","rgb":"rgb(170,234,255)"}, {"aci":142,"hex":"#008DBD","rgb":"rgb(0,141,189)"}, {"aci":143,"hex":"#7EADBD","rgb":"rgb(126,173,189)"}, {"aci":144,"hex":"#006081","rgb":"rgb(0,96,129)"}, {"aci":145,"hex":"#567681","rgb":"rgb(86,118,129)"}, {"aci":146,"hex":"#004E68","rgb":"rgb(0,78,104)"}, {"aci":147,"hex":"#455F68","rgb":"rgb(69,95,104)"}, {"aci":148,"hex":"#003B4F","rgb":"rgb(0,59,79)"}, {"aci":149,"hex":"#35494F","rgb":"rgb(53,73,79)"}, {"aci":150,"hex":"#007FFF","rgb":"rgb(0,127,255)"}, {"aci":151,"hex":"#AAD4FF","rgb":"rgb(170,212,255)"}, {"aci":152,"hex":"#005EBD","rgb":"rgb(0,94,189)"}, {"aci":153,"hex":"#7E9DBD","rgb":"rgb(126,157,189)"}, {"aci":154,"hex":"#004081","rgb":"rgb(0,64,129)"}, {"aci":155,"hex":"#566B81","rgb":"rgb(86,107,129)"}, {"aci":156,"hex":"#003468","rgb":"rgb(0,52,104)"}, {"aci":157,"hex":"#455668","rgb":"rgb(69,86,104)"}, {"aci":158,"hex":"#00274F","rgb":"rgb(0,39,79)"}, {"aci":159,"hex":"#35424F","rgb":"rgb(53,66,79)"}, {"aci":160,"hex":"#003FFF","rgb":"rgb(0,63,255)"}, {"aci":161,"hex":"#AABFFF","rgb":"rgb(170,191,255)"}, {"aci":162,"hex":"#002EBD","rgb":"rgb(0,46,189)"}, {"aci":163,"hex":"#7E8DBD","rgb":"rgb(126,141,189)"}, {"aci":164,"hex":"#001F81","rgb":"rgb(0,31,129)"}, {"aci":165,"hex":"#566081","rgb":"rgb(86,96,129)"}, {"aci":166,"hex":"#001968","rgb":"rgb(0,25,104)"}, {"aci":167,"hex":"#454E68","rgb":"rgb(69,78,104)"}, {"aci":168,"hex":"#00134F","rgb":"rgb(0,19,79)"}, {"aci":169,"hex":"#353B4F","rgb":"rgb(53,59,79)"}, {"aci":170,"hex":"#0000FF","rgb":"rgb(0,0,255)"}, {"aci":171,"hex":"#AAAAFF","rgb":"rgb(170,170,255)"}, {"aci":172,"hex":"#0000BD","rgb":"rgb(0,0,189)"}, {"aci":173,"hex":"#7E7EBD","rgb":"rgb(126,126,189)"}, {"aci":174,"hex":"#000081","rgb":"rgb(0,0,129)"}, {"aci":175,"hex":"#565681","rgb":"rgb(86,86,129)"}, {"aci":176,"hex":"#000068","rgb":"rgb(0,0,104)"}, {"aci":177,"hex":"#454568","rgb":"rgb(69,69,104)"}, {"aci":178,"hex":"#00004F","rgb":"rgb(0,0,79)"}, {"aci":179,"hex":"#35354F","rgb":"rgb(53,53,79)"}, {"aci":180,"hex":"#3F00FF","rgb":"rgb(63,0,255)"}, {"aci":181,"hex":"#BFAAFF","rgb":"rgb(191,170,255)"}, {"aci":182,"hex":"#2E00BD","rgb":"rgb(46,0,189)"}, {"aci":183,"hex":"#8D7EBD","rgb":"rgb(141,126,189)"}, {"aci":184,"hex":"#1F0081","rgb":"rgb(31,0,129)"}, {"aci":185,"hex":"#605681","rgb":"rgb(96,86,129)"}, {"aci":186,"hex":"#190068","rgb":"rgb(25,0,104)"}, {"aci":187,"hex":"#4E4568","rgb":"rgb(78,69,104)"}, {"aci":188,"hex":"#13004F","rgb":"rgb(19,0,79)"}, {"aci":189,"hex":"#3B354F","rgb":"rgb(59,53,79)"}, {"aci":190,"hex":"#7F00FF","rgb":"rgb(127,0,255)"}, {"aci":191,"hex":"#D4AAFF","rgb":"rgb(212,170,255)"}, {"aci":192,"hex":"#5E00BD","rgb":"rgb(94,0,189)"}, {"aci":193,"hex":"#9D7EBD","rgb":"rgb(157,126,189)"}, {"aci":194,"hex":"#400081","rgb":"rgb(64,0,129)"}, {"aci":195,"hex":"#6B5681","rgb":"rgb(107,86,129)"}, {"aci":196,"hex":"#340068","rgb":"rgb(52,0,104)"}, {"aci":197,"hex":"#564568","rgb":"rgb(86,69,104)"}, {"aci":198,"hex":"#27004F","rgb":"rgb(39,0,79)"}, {"aci":199,"hex":"#42354F","rgb":"rgb(66,53,79)"}, {"aci":200,"hex":"#BF00FF","rgb":"rgb(191,0,255)"}, {"aci":201,"hex":"#EAAAFF","rgb":"rgb(234,170,255)"}, {"aci":202,"hex":"#8D00BD","rgb":"rgb(141,0,189)"}, {"aci":203,"hex":"#AD7EBD","rgb":"rgb(173,126,189)"}, {"aci":204,"hex":"#600081","rgb":"rgb(96,0,129)"}, {"aci":205,"hex":"#765681","rgb":"rgb(118,86,129)"}, {"aci":206,"hex":"#4E0068","rgb":"rgb(78,0,104)"}, {"aci":207,"hex":"#5F4568","rgb":"rgb(95,69,104)"}, {"aci":208,"hex":"#3B004F","rgb":"rgb(59,0,79)"}, {"aci":209,"hex":"#49354F","rgb":"rgb(73,53,79)"}, {"aci":210,"hex":"#FF00FF","rgb":"rgb(255,0,255)"}, {"aci":211,"hex":"#FFAAFF","rgb":"rgb(255,170,255)"}, {"aci":212,"hex":"#BD00BD","rgb":"rgb(189,0,189)"}, {"aci":213,"hex":"#BD7EBD","rgb":"rgb(189,126,189)"}, {"aci":214,"hex":"#810081","rgb":"rgb(129,0,129)"}, {"aci":215,"hex":"#815681","rgb":"rgb(129,86,129)"}, {"aci":216,"hex":"#680068","rgb":"rgb(104,0,104)"}, {"aci":217,"hex":"#684568","rgb":"rgb(104,69,104)"}, {"aci":218,"hex":"#4F004F","rgb":"rgb(79,0,79)"}, {"aci":219,"hex":"#4F354F","rgb":"rgb(79,53,79)"}, {"aci":220,"hex":"#FF00BF","rgb":"rgb(255,0,191)"}, {"aci":221,"hex":"#FFAAEA","rgb":"rgb(255,170,234)"}, {"aci":222,"hex":"#BD008D","rgb":"rgb(189,0,141)"}, {"aci":223,"hex":"#BD7EAD","rgb":"rgb(189,126,173)"}, {"aci":224,"hex":"#810060","rgb":"rgb(129,0,96)"}, {"aci":225,"hex":"#815676","rgb":"rgb(129,86,118)"}, {"aci":226,"hex":"#68004E","rgb":"rgb(104,0,78)"}, {"aci":227,"hex":"#68455F","rgb":"rgb(104,69,95)"}, {"aci":228,"hex":"#4F003B","rgb":"rgb(79,0,59)"}, {"aci":229,"hex":"#4F3549","rgb":"rgb(79,53,73)"}, {"aci":230,"hex":"#FF007F","rgb":"rgb(255,0,127)"}, {"aci":231,"hex":"#FFAAD4","rgb":"rgb(255,170,212)"}, {"aci":232,"hex":"#BD005E","rgb":"rgb(189,0,94)"}, {"aci":233,"hex":"#BD7E9D","rgb":"rgb(189,126,157)"}, {"aci":234,"hex":"#810040","rgb":"rgb(129,0,64)"}, {"aci":235,"hex":"#81566B","rgb":"rgb(129,86,107)"}, {"aci":236,"hex":"#680034","rgb":"rgb(104,0,52)"}, {"aci":237,"hex":"#684556","rgb":"rgb(104,69,86)"}, {"aci":238,"hex":"#4F0027","rgb":"rgb(79,0,39)"}, {"aci":239,"hex":"#4F3542","rgb":"rgb(79,53,66)"}, {"aci":240,"hex":"#FF003F","rgb":"rgb(255,0,63)"}, {"aci":241,"hex":"#FFAABF","rgb":"rgb(255,170,191)"}, {"aci":242,"hex":"#BD002E","rgb":"rgb(189,0,46)"}, {"aci":243,"hex":"#BD7E8D","rgb":"rgb(189,126,141)"}, {"aci":244,"hex":"#81001F","rgb":"rgb(129,0,31)"}, {"aci":245,"hex":"#815660","rgb":"rgb(129,86,96)"}, {"aci":246,"hex":"#680019","rgb":"rgb(104,0,25)"}, {"aci":247,"hex":"#68454E","rgb":"rgb(104,69,78)"}, {"aci":248,"hex":"#4F0013","rgb":"rgb(79,0,19)"}, {"aci":249,"hex":"#4F353B","rgb":"rgb(79,53,59)"}, {"aci":250,"hex":"#333333","rgb":"rgb(51,51,51)"}, {"aci":251,"hex":"#505050","rgb":"rgb(80,80,80)"}, {"aci":252,"hex":"#696969","rgb":"rgb(105,105,105)"}, {"aci":253,"hex":"#828282","rgb":"rgb(130,130,130)"}, {"aci":254,"hex":"#BEBEBE","rgb":"rgb(190,190,190)"}, {"aci":255,"hex":"#FFFFFF","rgb":"rgb(255,255,255)"}]
        from collections import defaultdict
        lcolor = defaultdict(lambda:1)
        lcolor.update({'MASK':42,'ELECTRODE':252,'METAL':252,'POLING':250,'NOTES':7,'IMPLANT':32,'BUFFER':132,'DICE':22,'MOCKUP':8,'WG':42,'APE':32,'BU':22,'EL':250,'1':42,'2':32,'3':22,'4':250})
        layers = self.subelemlayers()
        layers = [layer for layer in layers if not 'NOTES'==layer] + ([] if 'NOTES' not in layers else ['NOTES']) # show notes on top of all other layers
        for layer in layers:
            if svg: glayer = dwg.add(dwg.g(id=layer))
            for e in self.subelems(): # print e.layer,layers,len(layers)
                if e.layer==layer:
                    if svg: g = glayer.add(dwg.g(id=e.name,opacity=0.75,fill='rgb'+str(autocadcolor(lcolor[e.layer]))))
                    if not singlelayertosave or singlelayertosave==e.layer:
                        for c in e.polys:
                            if svg:
                                cc = compresscurve(c,maxangle=svgcompressangle)
                                cc = [(scale(x),scale(y)) for x,y in cc]
                                #fill = 'darkgoldenrod' if e.layer not in lcolor else 'rgb'+str((autocadcolor(lcolor[e.layer])))
                                #g.add(dwg.polygon(cc, fill=fill))
                                #ss = ''.join(['L %s %s ' % (x,y) for x,y in cc]) # use path instead of polygon for shorter file size (eg. prevent -666.9250000000001)
                                #g.add(dwg.path(d='M'+ss[1:]+'Z', fill=fill))
                                if 'NOTES'==layer:
                                    g.add(dwg.polyline(cc,fill='none',stroke='darkblue'))
                                elif 'MASK'==layer:
                                    g.add(dwg.polygon(cc))
                                else:
                                    g.add(dwg.polyline(cc,fill='none',stroke='rgb'+str(autocadcolor(lcolor[e.layer]))))
                                if svgdebug:
                                    for x,y in cc: g.add(dwg.circle((x,y),svgscale*20, fill='black'))
                            else:
                                cc = [(x,-y) for x,y in c]
                                if cc[0]==cc[-1]:
                                    space.add_lwpolyline(cc, dxfattribs={'layer':e.layer,'closed':True,'color':lcolor[e.layer]})
                                else:
                                    space.add_lwpolyline(cc, dxfattribs={'layer':e.layer,'color':lcolor[e.layer]})
                                    assert 'NOTES'==layer, 'poly not closed:'+layer+str(cc)
                        for x,y,dx,dy in e.rects:
                            if svg:
                                if 'MASK'==layer:
                                    g.add(dwg.rect(insert=(scale(x),scale(y)), size=(scale(dx),scale(dy)),fill='darkgreen'))
                                else:
                                    g.add(dwg.rect(insert=(scale(x),scale(y)), size=(scale(dx),scale(dy)),fill='none',stroke='rgb'+str(autocadcolor(lcolor[e.layer]))))
                            else:
                                xx,yy,dxx,dyy = x,-y,dx,-dy
                                cc = [(xx,yy),(xx+dxx,yy),(xx+dxx,yy+dyy),(xx,yy+dyy),(xx,yy)]
                                space.add_lwpolyline(cc, dxfattribs={'layer':e.layer,'closed':True,'color':lcolor[e.layer] if not 'MASK'==e.layer else (42 if singlelayertosave else 75)}) # for singlelayer, layer is single color at Tim's request
                    if not singlelayertosave or singlelayertosave=='NOTES':
                        for note in e.notes:
                            (k,v),dy = [(k,note.note[k]) for k in note.note][0],note.dy
                            ncolor = {'grating':31,'width':16,'taper':33,'length':33,'separation':242,'displacement':242,'roc':142,
                                'ewidth':151,'outwidth':151,'inwidth':151,'egap':223,'outgap':223,'ingap':223,'bragg':32,'mzub':32,
                                'mzul':32,'splitradius':9,'implant':32,'buffer':132}
                            rep = {'outwidth':'%d','outgap':'%d','length':'%d','separation':'↕%g','displacement':'↕%.1f',
                                'roc':'%.1fmm ROC','mzub':'%s','mzul':'%s','splitradius':'%gr'}
                            offset = {'width':(-1,0),'taper':(0,0.75),'length':(0,0.75),'roc':(0,-1),'outwidth':(0,-1),'outgap':(0,-2),
                                'inwidth':(0,-1),'ingap':(0,-2),'ewidth':(-0.75,0),'egap':(-0.75,0),'bragg':(4,-0.5),'mzub':(0,0.75),'mzul':(0,0.75)}
                            color = ncolor[k] if k in ncolor else lcolor['NOTES']
                            s = rep[k]%v if k in rep else ( v if isinstance(v,str) else str(int(v) if int(v)==v else v) )
                            xo,yo = offset[k] if k in offset else (0,0)
                            xx,yy = note.x+xo*dy,(note.y+0.5*dy+yo*dy) # for dxf, middle of digit is at 0.5*height
                            if svg:
                                def darken(rgb):
                                    return tuple(int(0.75*ci) for ci in rgb)
                                svgtextscale,svgcolor = 1.5,'rgb'+str(darken(autocadcolor(color)))
                                g.add(dwg.text(s,insert=(scale(xx),scale(yy-0.5*dy+svgtextscale*.375*dy)),font_size=scale(svgtextscale*dy),fill=svgcolor)) # for svg, middle of digit seems to be at 0.375*height
                            else:
                                # dxfattribs={'color':color,'layer':'NOTES','height':dy}
                                dxfattribs={'color':color,'layer':'NOTES','height':dy,'style':'arial'}
                                space.add_text(s, dxfattribs=dxfattribs).set_pos((xx,-yy), align='CENTER')
                                #space.add_mtext(s) # use mtext for multiple lines?
        #space.add_text('X', dxfattribs={'layer':'NOTES','height':2000,'color':53}).set_pos((0,2000), align='CENTER') # anchor = {'taper':'TOP_CENTER','length':'TOP_CENTER','roc':'TOP_CENTER'} # a = anchor[k] if k in anchor else 'CENTER'
        if svg:
            if not nomodify:
                self.scale(1/svgscale)
            dwg.save()
        else:
            drawing.saveas(filename)
        if verbose: print('  "'+filename+'" saved.')
    # def savedxfwrite(self,filename=''):
    #     from dxfwrite import DXFEngine as dxf # pip install dxfwrite
    #     import dxfwrite.const as const
    #     if not filename: filename = self.name
    #     if not filename: filename = 'mask'
    #     colors = [16,42,75,146,207,254] # rogbvg # 0=black,7=white
    #     drawing = dxf.drawing(filename.rstrip('.dxf')+'.dxf')
    #     #for layer in layers: drawing.layers.new(name=layer)
    #     drawing.add_layer('MASK', color=colors[0])
    #     #print 'const.POLYLINE_3D_POLYLINE',const.POLYLINE_3D_POLYLINE
    #     for x,y,dx,dy in self.subrects():
    #         drawing.add(dxf.rectangle((x,-y-dy),dx,dy, color=colors[4], bgcolor=colors[4], layer='MASK')) # drawing.add(dxf.rectangle((0,0),10,1, color=colors[2], layer='MASK')) # https://pythonhosted.org/dxfwrite/entities/polyline.html # drawing.add(dxf.line((0,0), (10,1), color=7))
    #     for c in self.subpolys():
    #         polyline = dxf.polyline(color=colors[3], layer='MASK', flags=0) # flags=const.POLYLINE_3D_POLYLINE is default # const.POLYLINE_3D_POLYLINE=8 in const.py # https://pythonhosted.org/ezdxf/entities.html#polyline
    #         polyline.add_vertices([(x,-y) for x,y in c]) # polyline.add_vertices( [(0,2.0), (3,2.0), (6,2.3), (9,2.3), (0,2.6), (0,2.0)] )
    #         polyline.close()
    #         drawing.add(polyline)
    #     #drawing.add_layer('TEXT', color=colors[1]); drawing.add(dxf.text('Test note', insert=(0,0), layer='TEXT'))
    #     drawing.save()
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
    def adddiceguides(self,x=None,y=None,chipx=None,chipy=None,s=25,strip=False):
        e = Element('dice',parent=self)
        x,y = getattr(self,'x',0) if x is None else x, getattr(self,'y',0) if y is None else y
        chipx = chipx if chipx is not None else self.finddefaultinfo('chiplength')
        chipy = chipy if chipy is not None else self.finddefaultinfo('chipwidth')
        def repeatx(r,periodx,endx):
            x0,y0,dx,dy = r
            for xx in range(0,endx,periodx):
                # self.addrect(x0+xx+x,y0+y,dx,dy) # won't show up in chip map! why not?
                e.addrect(x0+xx+x,y0+y,dx,dy)
        repeatx((0,0,2*s,s),1000,chipx)
        repeatx((0,s,s,2*s),5000,chipx)
        repeatx((0,chipy-2*s,s,s),2000,chipx)
        repeatx((0,chipy-s,2*s,s),1000,chipx)
        if strip:
            ee = Element('dicestrip',layer='STRIPPING',parent=self)
            sy = 150
            ee.addrect(0,-sy/2,chipx,sy).addrect(0,chipy-sy/2,chipx,sy)
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
        # from wave import Vs; Vs(ps).wave().plot(m=1)
        e.addpoly(ps)
        return self
    def addfiducialwindow(self,x=0,y=0,layer='POLING'):
        e = Element('crossfiducial',layer=layer,parent=self)
        if layer=='STRIPPING':
            e.addcenteredrect(0,0,3820,2020)
        else:
            e.addcenteredrect(0,+800,3000,400).addcenteredrect(+1700,0,400,2000)
            e.addcenteredrect(0,-800,3000,400).addcenteredrect(-1700,0,400,2000)
        e.translate(x,y)
        return self
    def addcrossfiducialchip(self,x=0,y=0,vnum=7,layer='POLING',onlymasklayer=False,rotation=0,partial=None,invertvernier=False):
        smallsize,bigsize,spacing = 10,20,127
        Element('crossfiducial',parent=self).addcrossrepeats(bigsize,spacing,vnum=vnum,partial=None).rotate(rotation).translate(x+1000,y)
        Element('crossfiducial',parent=self).addcrossrepeats(smallsize,spacing,vnum=vnum,partial=None).rotate(rotation).translate(x-1000,y)
        self.addvernierset(layer,x-1000,y,rotation,partial=partial,invert=invertvernier).addvernierset(layer,x+1000,y,rotation,partial=partial,invert=invertvernier)
        if not onlymasklayer:
            Element('crossfiducial',layer=layer,parent=self).addcrossrepeats(smallsize,spacing,vnum=vnum,partial=partial).rotate(rotation).translate(x+1000,y)
            Element('crossfiducial',layer=layer,parent=self).addcrossrepeats(bigsize,spacing,vnum=vnum,partial=partial).rotate(rotation).translate(x-1000,y)
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
    def addhframe(self,dx,dy,dw,layer=None,left=True,right=True):
        e = Element('frame',layer=layer,parent=self)
        if left:  e.addrect(-dx/2,-dy/2,dw,dy)
        if right: e.addrect(dx/2-dw,-dy/2,dw,dy)
        return self
    def addvframe(self,dx,dy,dw,layer=None):
        e = Element('frame',layer=layer,parent=self).addrect(-dx/2,-dy/2,dx,dw).addrect(-dx/2,dy/2-dw,dx,dw)
        return self
    def addonchannel(self,dx,width=None,note=True):
        assert 0<=dx
        x,y = self.x,self.y
        width = width if width is not None else self.width
        self.addrect(x,y-width/2.,dx,width)
        self.x,self.width = self.x+dx,width
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
    def addonsbend(self,dx,dy,width=None,x=None,y=None,note=True):
        assert dx>0
        x = x if x is not None else self.x
        y = y if y is not None else self.y
        width = width if width is not None else self.width
        if 0==dy: return self.addonchannel(dx)
        xs,ys,roc,pathlength = sbend(dx,dy)
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
        if poshalf: self.addnote(x+dx/2,y-dy/2,roc=roc/1000)
        return self
    def addonmodefilter(self,width,dx,inwidth=0,outwidth=0,modefilterx=0,taperx=0):
        if inwidth: self.addonchannel(modefilterx,inwidth).addontaper(taperx,outwidth=width)
        assert 0<dx-(modefilterx+taperx)*(bool(inwidth)+bool(outwidth)), 'chip not long enough for mode filters'
        self.addonchannel(dx-(modefilterx+taperx)*(bool(inwidth)+bool(outwidth)),width)
        if outwidth: self.addontaper(taperx,outwidth=outwidth).addonchannel(modefilterx,outwidth)
        return self
    def addmodefilterspliced(self,width,splicex1,splicex2,dx=None,inwidth=0,outwidth=0,modefilterx=None,taperx=None,swapinandout=False):
        # same as addmodefilter() but with a gap missing in the channel part from splicex1 to splicex2
        if swapinandout: inwidth,outwidth = outwidth,inwidth
        e = Element('guide',parent=self)
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
        e = Element('guide',parent=self,x=x0,y=y0)
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
    def addchannel(self,width,dx=None):
        self.addmodefilter(width,dx,inwidth=0,outwidth=0,modefilterx=0,taperx=0)
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
        e = Element('guide',parent=self)
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
            e.addlayergrating(period,gx=gx,gy=gy,x0=x0+(modefilterx+taperx)*(inwidth!=0),y0=y0,dc=dc)
        else:
            e.addlayergrating(period,gx=gx,gy=gy,x0=x0+modefilterx+taperx,y0=y0,dc=dc)
        return self
    def addmultishgmodefilter(self,width,periods,gxs,dx=None,inwidth=0,outwidth=0,modefilterx=None,taperx=None,x0=None,y0=None,dc=None):
        e = Element('guide',parent=self)
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
    def addinterleavedshgmf(self,width,period1,period2,overpole=0,gx=None,gy=None,dx=None,inwidth=0,outwidth=0,modefilterx=None,taperx=None,x0=None,y0=None,extendpoling=False):
        e = Element('guide',parent=self)
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
        e = Element('guide',parent=self)
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
        e = Element('guide',parent=self)
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
    def addwdm(self,width,couplergap,couplerx,couplerwidth,
                mfsbendin,mfsbendout,mfchannelin,mfchannelout,
                modefilterx=0,taperx=5000,mirror=False,vgroovepitch=None,metrics=True,
                samemfouts=False,wdmatoutput=False):
        sbendlength = self.finddefaultinfo('sbendlength')
        vgroovepitch = vgroovepitch if vgroovepitch is not None else self.finddefaultinfo('vgroovepitch')
        dx = self.finddefaultinfo('chiplength')-self.x
        yb = vgroovepitch-couplergap
        # print('modefilterx,taperx,sbendlength,vgroovepitch',modefilterx,taperx,sbendlength,vgroovepitch)
        # print('modefilters:',mfsbendin,'\_/->',mfsbendout,',',mfchannelin,'-->',mfchannelout,'gap:',couplergap,'L0:',couplerx,'couplerguidewidth',couplerwidth,'criticaldimension:',couplergap-couplerwidth)
        x0,y0 = self.x,self.y
        if metrics:
            # xmetric,ymetric = x0+modefilterx+taperx+sbendlength+couplerx/2,y0+(vgroovepitch+6*width if mirror else 0)
            xmetric,ymetric = x0+modefilterx+taperx+sbendlength+couplerx/2,y0+(vgroovepitch if mirror else width*6)
            self.addoversizemetrics(width,xmetric,ymetric,nx=8,ny=6)
        e = Element('guide',parent=self)
        e.info.inputmodefilter,e.info.outputmodefilter,e.info.guidewidth = mfsbendin,mfsbendout,width
        if mfsbendin==mfchannelin==0:
            e.addonchannel(modefilterx+taperx,width=width)
        else:
            e.addonchannel(modefilterx,width=mfsbendin).addontaper(taperx,outwidth=width)
        e.addonsbend(sbendlength,0 if mirror else +yb).addonchannel(couplerx,note=False)
        e.addnote(e.x-couplerx/2,e.y,gap=float2string(couplergap)+' gap')
        e.addonsbend(sbendlength,0 if mirror else -yb)#.addontaper(taperx,outwidth=width)
        gx0, gx = e.x, dx-(e.x-x0)-taperx-modefilterx
        if mfsbendout:
            e.addonchannel(gx).addontaper(taperx,outwidth=mfsbendout).addonchannel(modefilterx)
        else:
            e.addonchannel(gx+taperx+modefilterx)
        ee = Element('guide',parent=self,guidespacing=vgroovepitch)
        ee.info.inputmodefilter,ee.info.outputmodefilter,ee.info.guidewidth = mfchannelin,mfchannelout,width
        if mfsbendin==mfchannelin==0:
            ee.addonchannel(modefilterx+taperx,width=width)
        else:
            ee.addonchannel(modefilterx,width=mfchannelin).addontaper(taperx,outwidth=width)
        ee.addonsbend(sbendlength,-yb if mirror else 0).addonchannel(couplerx)
        ee.addonsbend(sbendlength,+yb if mirror else 0)#.addontaper(taperx,outwidth=width)
        if mfchannelout:
            ee.addonchannel(gx).addontaper(taperx,outwidth=mfchannelout).addonchannel(modefilterx)
        else:
            ee.addonchannel(gx+taperx+modefilterx)
        f = Element('wdm',parent=ee) # only for informational purpose
        f.info.wdmgap,f.info.wdmlength,f.info.wdmguidewidth = couplergap,couplerx,width
        f.info.guidenumbers = e.info.guidenumber + '&' + ee.info.guidenumber
        return self

    def addqfc(self,mfpump,mfin,mfout,width,couplergap,couplerx,couplerwidth,period,pumponsbend,gratingonsbend,
            dx=None,samemfouts=False,extendpoling=False,grating=1,invertwdm=False,auxperiod=0):
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
        print('modefilters:',mfsbendin,'\_/->',mfsbendout,',',mfchannelin,'-->',mfchannelout,'gap:',couplergap,'L0:',couplerx,'couplerguidewidth',couplerwidth,'period:',period,'gratingonsbend:',gratingonsbend,'pumponsbend:',pumponsbend,'criticaldimension:',couplergap-couplerwidth)

        e = Element('guide',parent=self)
        x0,y0 = e.x,e.y
        e.info.inputmodefilter,e.info.outputmodefilter,e.info.guidewidth = mfsbendin,mfsbendout,width
        e.addonchannel(modefilterx,width=mfsbendin).addontaper(taperx,outwidth=couplerwidth)
        e.addonsbend(sbendlength,vgroovepitch-couplergap).addonchannel(couplerx)
        e.addnote(e.x-couplerx/2,e.y,gap=float2string(couplergap)+' gap')
        if invertwdm:
            e.addonsbend(sbendlength,0).addontaper(taperx,outwidth=width)
        else:
            e.addonsbend(sbendlength,-vgroovepitch+couplergap).addontaper(taperx,outwidth=width)
        gx0, gx = e.x, dx-(e.x-x0)-taperx-modefilterx
        if mfsbendout:
            e.addonchannel(gx).addontaper(taperx,outwidth=mfsbendout).addonchannel(modefilterx)
        else:
            e.addonchannel(gx+taperx+modefilterx)
        assert 2==grating
        # if ((invertwdm and not gratingonsbend) or (not invertwdm and gratingonsbend)) and grating:
        e.addlayergrating(period,gx=gx+(taperx+modefilterx)*extendpoling*(0==mfsbendout),x0=gx0,y0=e.y)

        ee = Element('guide',parent=self,guidespacing=vgroovepitch)
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
        assert 2==grating
        # if not ((invertwdm and not gratingonsbend) or (not invertwdm and gratingonsbend)) and grating:
        ee.addlayergrating(period,gx=gx+(taperx+modefilterx)*extendpoling*(0==mfchannelout),x0=gx0,y0=ee.y)

        if 0<auxperiod:
            e.addlayergrating(period,gx=gx+(taperx+modefilterx)*extendpoling*(0==mfsbendout),x0=gx0,y0=0.5*(e.y+ee.y))
            if 2==auxperiod:
                assert 0, 'auxperiod waveguide not implemented'

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

        e0 = Element('guide',x=x0,y=y0,parent=self) # input ir
        e0.addonchannel(modefilterx,width=mfir).addontaper(taperx,outwidth=couplerwidth)
        e0.addonsbend(sbendlength,vgroovepitch-couplergap).addonchannel(couplerx,note=0)
        e0.addnote(e0.x-couplerx/2,e0.y,gap=float2string(couplergap)+' gap')
        e0.addonsbend(sbendlength,couplergap-vgroovepitch).addonchannel(sx)
        e0.addontaper(taperx,outwidth=mfir).addonchannel(modefilterx,width=mfir)

        e1 = Element('guide',x=x0,y=y0+vgroovepitch,parent=self) # input shg
        e1.addonchannel(modefilterx,width=mfshg).addontaper(taperx,outwidth=couplerwidth)
        e1.addonchannel(sbendlength).addonchannel(couplerx).addontaper(taperx,outwidth=width)
        gx0, gy0 = e1.x, e1.y
        e1.addonchannel(gx)
        e1.addlayergrating(period,gx=gx,x0=gx0,y0=gy0)
        gx1, gy1 = e1.x, e1.y
        e1.addontaper(taperx,outwidth=width).addonchannel(couplerx).addonchannel(sbendlength)
        e1.addontaper(taperx,outwidth=couplerwidth).addonchannel(modefilterx,width=mfshg)

        e2 = Element('guide',x=x0,y=y0+2*vgroovepitch,parent=self) # output ir
        e2.addonchannel(modefilterx,width=mfir).addontaper(taperx,outwidth=couplerwidth)
        e2.addonchannel(sx)
        e2.addonsbend(sbendlength,couplergap-vgroovepitch).addonchannel(couplerx,note=0)
        e2.addnote(e2.x-couplerx/2,e2.y,gap=float2string(couplergap)+' gap')
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

        e1 = Element('guide',x=x0,y=y0+guidespacing,parent=self) # input shg
        e1.addonchannel(modefilterx,width=mfshg).addontaper(taperx,outwidth=couplerwidth)
        e1.addonsbend(sbendlength,vgroovepitch-couplergap).addonchannel(couplerx).addontaper(taperx,outwidth=width)
        e1.addnote(e1.x-couplerx/2,e1.y,gap=float2string(couplergap)+' gap')
        gx0, gy0, gx = e1.x, e1.y, dx-2*modefilterx-4*taperx-2*sbendlength-2*couplerx
        e1.addlayergrating(period,gx=gx,x0=gx0,y0=gy0)
        e1.addonchannel(gx)
        e1.addnote(e1.x+couplerx/2,e1.y,gap=float2string(couplergap)+' gap')
        e1.addontaper(taperx,outwidth=couplerwidth).addonchannel(couplerx).addonsbend(sbendlength,vgroovepitch-couplergap)
        e1.addontaper(taperx,outwidth=mfshg).addonchannel(modefilterx,width=mfshg)
        sb2x = 2*taperx + sbendlength + couplerx + gx
        sb2y = vgroovepitch + guidespacing - 2*couplergap

        e0 = Element('guide',x=x0,y=y0,parent=self) # aux ir (output ir)
        e0.addonchannel(modefilterx,width=mfir).addontaper(taperx,outwidth=couplerwidth)
        e0.addonsbend(sb2x-sb2dx,sb2y-sb2dy).addonsbend(sb2dx,sb2dy).addonchannel(couplerx).addonchannel(sbendlength)
        e0.addontaper(taperx,outwidth=mfir).addonchannel(modefilterx,width=mfir)

        e2 = Element('guide',x=x0,y=y0+guidespacing+vgroovepitch,parent=self) # input ir
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

        e = Element('guide',parent=self)
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

        ee = Element('guide',parent=self,guidespacing=vgroovepitch)
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

        e = Element('guide',parent=self)
        x0,y0 = e.x,e.y
        e.info.inputmodefilter,e.info.outputmodefilter,e.info.guidewidth = mfsbendin,mfsbendout,width
        e.addonchannel(modefilterx,width=mfsbendin).addontaper(taperx,outwidth=width)
        e.addonsbend(sbendlength,(vgroovepitch-precouplergap)).addontaper(taperx,outwidth=couplerwidth,note=0).addonsbend(precouplerx,(precouplergap-couplergap),note=0)
        e.addonchannel(couplerx,note=0)
        e.addnote(e.x-couplerx/2,e.y,gap=str(couplergap)+' gap')
        e.addonsbend(precouplerx,-(precouplergap-couplergap)).addontaper(taperx,outwidth=width,note=0).addonsbend(sbendlength,-(vgroovepitch-precouplergap))
        gx0, gx = e.x, dx-(e.x-x0)-taperx-modefilterx
        if mfsbendout:
            e.addonchannel(gx).addontaper(taperx,outwidth=mfsbendout).addonchannel(modefilterx)
        else:
            e.addonchannel(gx+taperx+modefilterx)
        if gratingonsbend:
            e.addlayergrating(period,gx=gx+(taperx+modefilterx)*extendpoling*(0==mfsbendout),x0=gx0,y0=e.y)

        ee = Element('guide',parent=self,guidespacing=vgroovepitch)
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
    def addonsplittapersbend(self,dx,dy,taperx,sx,sy,inwidth=None,outwidth=None,x=None,y=None,reverse=False,note=True):
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
        e = Element('guide',parent=self)
        x0,y0 = e.x,e.y
        e.info.inputmodefilter,e.info.outputmodefilter,e.info.guidewidth = mfsbendin,mfsbendout,width
        e.addonchannel(modefilterx,width=mfsbendin).addontaper(taperx,outwidth=width)

        # e.addonsbend(sbendlength,(vgroovepitch-precouplergap)).addontaper(taperx,outwidth=couplerwidth,note=0).addonsbend(precouplerx,(precouplergap-couplergap),note=0)
        e.addonsplittapersbend(sbendlength+taperx+precouplerx/2,vgroovepitch-couplergap,taperx,precouplerx,10,inwidth=width,outwidth=couplerwidth,note=True)
        # print('e.info.splittaperdx',e.info.splittaperdx)

        e.addonchannel(couplerx,note=0)
        e.addnote(e.x-couplerx/2,e.y,gap=str(couplergap)+' gap')

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

        ee = Element('guide',parent=self,guidespacing=vgroovepitch)
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
    def addchiplabels(self,chipx=None,text=None,x0=1000,y0=200,extratext=''): # self is the group, self.parent is the chip
        chipx = chipx if chipx is not None else self.finddefaultinfo('chiplength') # self.parent.info.chipx
        text = text if text is not None else self.parent.name+' '+self.finddefaultinfo('chipid')
        e = Element('metric',parent=self)
        for x in range(0,chipx,5000):
            #for i,n in enumerate([4,3,2,1]): e.addmetric(x-75*(i+1),0,25,n,2,6)
            e.addtext(text+extratext,x,0)
        e.translate(x0,y0)
        return self
    def addguidelabels(self,chipx=None,text=None,x0=None,y0=None,extratext='',dy=-25,skip=[],polingfill=0): # self is the group, self.parent is the chip
        chipx = chipx if chipx is not None else self.finddefaultinfo('chiplength') # self.parent.info.chipx
        text = text if text is not None else self.finddefaultinfo('chipid')+'-G'+str(self.finddefaultinfo('groupcount'))+'.1'  # self.parent.info.chipid+'-G'+str(self.parent.info.groupcount)+'.1'#+str(self.info.guidenumber) # G1.1,G2.1
        minfeature = self.finddefaultinfo('minfeature',fail=False) or self.finddefaultinfo('maskres')
        x0 = x0 if x0 is not None else self.x
        y0 = y0 if y0 is not None else self.y
        repx = 5000
        xlast = None
        if polingfill:
            g = Element('metric',parent=self)
        for x in range(repx,int(chipx)-repx+1,repx):
            if x not in skip:
                e = Element('metric',parent=self)
                for i,n in enumerate([4,3,2,1]+[0.6]*(minfeature<=0.6)):
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
        g = Element('grating',layer='POLING',parent=self) # g.addrect( x0,y0-gy/2.,gx,gy )
        g.info.period1,g.info.period2 = period1,period2
        g.info.gratingstart,g.info.gratinglength,g.info.gratingwidth = x0,gx,gy
        self.addnote(x0+gx/4,y0,grating="%s Λ, %s Λ'"%(float2string(period1),float2string(period2)))
        if dev: period1,period2 = periodmag*period1,periodmag*period2
        barstarts,barends = interleavedgrating(period1,period2,padcount=1,gx=gx,overpole=overpole) # grating(period,dc,gx,x0=x0)
        for xi,xj in zip(barstarts,barends):
            g.addrect( x0+xi,y0-gy/2., xj-xi,gy )
        return self
    def addbragggrating(self,width,dx,period=8,dc=1,enddc=None,angle=0.2,x0=0,y0=0):
        if False==self.finddefaultinfo('braggangle'):
            angle = 0
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
    def addbragg(self,width,dc,period=None,dx=None):
        e = Element('guide',parent=self)
        dx = dx if dx is not None else self.finddefaultinfo('chiplength')
        period = period if period is not None else self.finddefaultinfo('braggperiod')
        e.info.guidewidth = width
        if 1==dc:
            e.addonchannel(dx,width)
        else:
            e.addonbragg(dx,width,period,dc)
        return self
    def addonbraggmodefilter(self,width,period,dx,indc=1,outdc=1,modefilterx=None,taperx=None):
        inmf,outmf = bool(indc<1),bool(outdc<1)
        if inmf: self.addonbragg(modefilterx,width,period,dc=indc).addonbragg(taperx,width,period,dc=indc,enddc=1)
        assert 0<dx-(modefilterx+taperx)*(bool(inmf)+bool(outmf)), 'chip not long enough for bragg mode filters'
        self.addonchannel(dx-(modefilterx+taperx)*(bool(inmf)+bool(outmf)),width)
        if outmf: self.addonbragg(taperx,width,period,dc=1,enddc=outdc).addonbragg(modefilterx,width,period,dc=outdc)
        return self
    def addbraggmodefilter(self,width,dx=None,indc=1,outdc=1,period=None,modefilterx=None,taperx=None,swapdcs=False):
        if swapdcs: indc,outdc = outdc,indc
        e = Element('guide',parent=self)
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
    def addtaperedbragggrating(self,inwidth,outwidth,dx,period=8,dc=1,enddc=None,angle=0.2,x0=0,y0=0,note=True):
        x,y = self.x,self.y
        if False==self.finddefaultinfo('braggangle'):
            angle = 0
        if dev: period,angle = periodmag*period,atan(periodmag*tan(angle))
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
    def addtaperedbraggmodefilter(self,mfwidth,width,dx=None,indc=1,outdc=1,braggperiod=None,modefilterx=None,taperx=None,swapdcs=False):
        if swapdcs: indc,outdc = outdc,indc
        e = Element('guide',parent=self)
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
        e = Element('guide',parent=self)
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
    def addelectrodemask(self,chipxys,**kwargs):
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
            electrodes = electrodelist(infile,outfile)
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
                submount.addsubmount(period=e.period,dc=e.dutycycle,padcount=e.padcount,gapx=e.padgap,
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
                # submount.adduagratingsubmount(expectedoverpole=e.expectedoverpole,padcount=e.padcount,gapx=e.padgap,inputconnected=e.inputconnected,outputconnected=e.outputconnected)
                submount.addcustomsubmount(filename=e.filename,label=e.label,expectedoverpole=e.expectedoverpole,padcount=e.padcount,gapx=e.padgap,inputconnected=e.inputconnected,outputconnected=e.outputconnected)
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
                submount.addinterleavedsubmount(period0=e.period0, period1=e.period1, padcount=e.padcount, label0=e.label, gapx=e.padgap, inputconnected=e.inputconnected, outputconnected=e.outputconnected)
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
    def adddicelanes(self,xx,yy,layers=['MASK','DICE'],maxradius=None):
        def addcenteredrect(self,x0,y0,dx,dy,swapxy=False):
            return self.addrect(y0-dy/2,x0-dx/2,dy,dx) if swapxy else self.addrect(x0-dx/2,y0-dy/2,dx,dy)
        ee = [Element('dicinglanes',layer=layer,parent=self) for layer in layers]
        for e in ee:
            for y in yy:
                e.adddashedline(xx[0]/2+xx[-1]//2, y, xx[-1]-xx[0], 100, 10000, 1000, xaxis=1, maxradius=maxradius)
            for x in xx:
                e.adddashedline(x, yy[0]/2+yy[-1]//2, 100, yy[-1]-yy[0], 10000, 1000, xaxis=0, maxradius=maxradius)
    def addsplitter(self,width,pitch,mfin=0,mfout=0,mzlength=None,sbendlength=None,splitradius=0,singlemfout=False,taperx=None,mftaperx=None):
        dx = self.finddefaultinfo('chiplength')
        modefilterx = self.finddefaultinfo('modefilterlength')
        taperx = self.finddefaultinfo('taperlength') if taperx is None else taperx
        mftaperx = self.finddefaultinfo('taperlength') if mftaperx is None else mftaperx
        sbendlength = sbendlength if sbendlength is not None else self.finddefaultinfo('sbendlength')
        e = Element('ysplitter-guide',parent=self)
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
        e = Element('scurve-guide',parent=self)
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
        e = Element('machzehnder-guide',parent=self)
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
    def addmfmachzehnder(self,width,pitch,mzlength=0,mf=0,sbendlength=None,splitradius=0,xoffset=0,y=None,**kwargs):
        # for an unbalanced mz, supply either deltawidth, deltalength, or deltabragg # first arm is normal, second arm is modified
        dx = self.finddefaultinfo('chiplength')
        taperx = self.finddefaultinfo('taperlength')
        sbendlength = sbendlength if sbendlength is not None else self.finddefaultinfo('sbendlength')
        e = Element('machzehnder-guide',parent=self)
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
        return self
    def addmetriconblockedchip(self):
        dx,dy = self.finddefaultinfo('chiplength'),self.finddefaultinfo('chipwidth')
        Element('witness',parent=self).addrect(0,0,dx,dy)
        # self.addnote(dx/2,dy/2,dy=100,note=f'↕ {dy} µm')
        return self
    def addwaferflatwindowchip(self,yflat=700):
        dx,dy = self.finddefaultinfo('chiplength'),self.finddefaultinfo('chipwidth')
        print('wafer flat window size',yflat)
        assert 0<yflat<dy, f'yflat:{yflat}'
        # Element('witness',parent=self).addrect(0,0,dx,0.25*dy).addrect(0,0.75*dy,dx,0.25*dy)
        # for i in range(6): Element('witness',parent=self).addrect(0,200*i,dx,100)
        Element('window',parent=self).addrect(0,0,dx,yflat) # wafer flat at (yflat), lined up to bottom of window
        self.addnote(dx/2,yflat/2,dy=100,note=f'↕ {yflat} µm')
        return self
    def addtestchip(self):
        # label = ' '+self.info.chipid+' '+chiptype+' '+str(targetgap)+'G'+str(targetwidth)+'W'
        # self.addtext(self.finddefaultinfo('maskname')+label,x=2000,y=200)
        # self.addtext(self.finddefaultinfo('maskname')+label,x=self.finddefaultinfo('chiplength')/2,y=200)
        # elabel = self.finddefaultinfo('electrodemaskname')+label
        self.info.guidespacing = 30
        self.info.groupspacing = 60
        # self.adddiceguides()
        self.y += 300
        g1 = Element('group',parent=self).addguidelabels(dy=-50)
        g1.addchannel(5)
        for wi in [5,6,7,8]:
            g1.addchannel(wi)
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
    def addcentertapelectrode(self,gap,label,length=40000,xoffset=0):
        hot,gnd = 50,50
        inputy = 100 # length of straight part before tapering out
        ingap,inhot,ingnd = 50,50,50
        outgap,outhot,outgnd = 500,500,500
        edgey = 200 # gap distance between metal and edge of chip
        dx,dy = self.finddefaultinfo('chiplength'),self.finddefaultinfo('chipwidth')
        g = Element('electrode',layer='ELECTRODE',parent=self)
        x,y = self.x+self.finddefaultinfo('chiplength')/2, self.ey if hasattr(self,'ey') else self.y
        g.addtext(label,x=2000+x+xoffset,y=y+500).addmetric(x=2000+x+xoffset-107,y=y+500,dx=25,dy=gap,nx=2,ny=6).addmetric(x=2000+x+xoffset-107*2,y=y+500,dx=25,dy=self.finddefaultinfo('minfeature'),nx=2,ny=6)
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
        return self
    def addcpwelectrode(self,L,gap,hot,fangap,fanhot,dx,dy,xin=0,xout=0,id=None):
        g = Element('electrode',layer='MASK',parent=self)
        import phidl,phidls
        label = f"{self.info.chipid if id is None else id} {L/1000:g}mm {gap}-{hot}-{gap}"
        D = phidls.uclacpw(L=L,gap=gap,hot=hot,fangap=fangap,fanhot=fanhot,chipx=dx-xin-xout,chipy=dy,label=label,mirror=True)
        D.movex(xin)
        polys = phidls.layerpolygons(D,layers=[0],closed=True)
        g.addpolys(polys)
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
    def addpmelectrode(self,gap=13,hot=250,fgap=50,fhot=400,label='',length=40000,xoffset=0,dicegap=209):
        dx,dy = self.finddefaultinfo('chiplength'),self.finddefaultinfo('chipwidth')
        g = Element('electrode',layer='ELECTRODE',parent=self)
        ex = self.x+self.finddefaultinfo('chiplength')/2 + xoffset
        ey = dy/2
        yin = dy/2-dicegap/2-30
        radius = max(400, hot + gap/2 + 50 )
        assert yin-radius>0, 'not enough room for taper'
        # radius = hot+gap/2+50 if (hot==fhot and gap==fgap) else None
        x0,y0 = ex-length/2,ey
        # win,we,wout = [fhot,fgap,fhot], [hot,gap,hot], [fhot,fgap,fhot]
        # ps = mzsribbons(L=length,r=radius,win=win,we=we,wout=wout,yin=yin,yout=yin,p0=(x0,y0),minangle=0.01)
        # g.addpolys([p.listcurve() for p in ps])
        if hot==fhot and gap==fgap:
            vss = mzsribbonsfixed(L=length,r=radius,hot=hot,gap=gap,yin=yin,yout=yin,p0=(x0,y0),minangle=0.01)
        else:
            vss = mzsribbons2(L=length,r=radius,hot=hot,gap=gap,fhot=fhot,fgap=fgap,yin=yin,yout=yin,p0=(x0,y0),minangle=0.01)
        g.addpolys(vss)
        def centers(w):
            return [sum(w[:n])+w[n]/2-sum(w[:len(w)//2])-w[len(w)//2]/2 for n in range(len(w))]
        def addtnotes(g,ws,x0,y0):
            zs = centers(ws)
            zs = [z-zs[len(zs)//2] for z in zs]
            ys = 5*[0] if 1==scalemag else [0,-50,0,-50,0]
            ss = ['inwidth','ingap','inwidth','ingap','inwidth']
            for z,y,w,s in zip(zs,ys,ws,ss): g.addnote(x0+z,y0+y,**{s:w})
        def addenotes(g,ws,x0,y0):
            zs = centers(ws)
            for z,w in zip(zs[0::2],ws[0::2]): g.addnote(x0+000,y0+z,ewidth=w)
            for z,w in zip(zs[1::2],ws[1::2]): g.addnote(x0+100,y0+z,egap=w)
        win,we,wout = [fhot,fgap,fhot], [hot,gap,hot], [fhot,fgap,fhot]
        addenotes(g, we, x0+(150 if 1==scalemag else 1500), y0)
        addenotes(g, we, x0+length+(-200 if 1==scalemag else -2000), y0)
        addtnotes(g, win, x0-radius-25+6.5, dy-200)
        addtnotes(g, wout, x0+length+radius-25+6.5, 200)
        return self
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
        def addpad(self,padnum,text,startx,endx,padinputy,gx,gy,inputconnected,outputconnected,design=None,mock=False): # shouldn't need gx here
            tabx,spiney = 1000,100
            taby = padinputy-spiney
            if not endx: endx = gx # todo:case for not exactperiod, dicegapx/y
            spineendx = gx if inputconnected else endx
            startx = 0 if inputconnected else startx
            x0,y0,dx,dy = gx-tabx,0,max(tabx-gx+endx,500),taby # top tab is at least 500 wide
            if design is None or mock:
                self.addrect(x0,y0,dx,dy) # top tab
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
            # self.addtext(text, x=startx,y=taby, fitx=gx-tabx-startx,fity=taby, margin=100, scale=5,scaleifdev=False)
            self.addtext(text, x=0,y=taby, fitx=gx-tabx,fity=taby, margin=100, scale=5,scaleifdev=False)
            if not mock: self.addsubmountmetric(startx+12,taby-12)
            if mock: self.addrect(min(x0,startx),taby+spiney, max(spineendx,x0+dx)-min(x0,startx),gy)
            # if 0==padnum: self.addmarksshape().addrachelsshape()
            return self
        for n in range(padcount):
            s = padtext[n%len(padtext)]
            e = Element('pad'+str(n),layer='MASK')
            # design = (design if design is None else (design if 'circle'==design[0] else design))
            # print('padstarts,padends',padstarts,padends)
            mock = Element('pad'+str(n),layer='MOCKUP',parent=e)
            addpad(e,   n,s,padstarts[n]-gx*n,padends[n]-gx*n,padinputy,gx,gy,inputconnected,outputconnected,design=design)
            addpad(mock,n,s,padstarts[n]-gx*n,padends[n]-gx*n,padinputy,gx,gy,inputconnected,outputconnected,design=design,mock=True)
            e.translate(gx*n)
            e.translate(self.x,self.y)
            self.addelem(e)
        self.info.text = getattr(self.info,'text','') + ','.join(padtext)
        return self
    def savegrating(self,barstarts,barends):
        name = self.info.chipid
        file = f'{self.parent.name}gratings.py'
        with open(file,'w' if '01A'==name else 'a') as f:
            f.write(f'barstarts{name},barends{name} = {barstarts},{barends}\n')
    def addsubmount(self,period=333,dc=0.5,phase=0,padcount=10,padinputy=1000,gx=2500,gy=3000,gapx=0,
        inputconnected=False,outputconnected=True,paddesign=True,apodize=None,padtext=None):
        # print('period,dc,gx,padcount,gapx,phase',period,dc,gx,padcount,gapx,phase)
        # add bars for all pads as single grating
        barstarts,barends = grating(period,dc,gx,padcount,gapx,phase,apodize=apodize)
        self.addgrating(barstarts,barends,padinputy,gy)
        self.info.averageperiod = averageperiod(barstarts,barends) # print 'averageperiod',self.info.averageperiod
        # add each pad
        padstarts,padends = findpadboundarys(barstarts,barends,gx,padcount)
        padtext = padtext if padtext is not None else makepadtext(self.parent.name,self.info.chipid,period,dc,padcount)
        if apodize:
            padtext[3] = {'asingauss23':'AGAU','asintriangle':'ATRI','trapezoid':'TRAP'}[apodize.replace('inverse','')] # replace duty cycle with type
        self.addpads(padstarts,padends,padtext,padcount,padinputy,gx,gy,inputconnected,outputconnected,design=['circle',period] if paddesign else None) #None)
        self.savegrating(barstarts,barends)
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
def standardtwolayerlithomask(draftfolder,chipnum=None,chipmap=False):
    maskname,polingmaskname = 'Dec99A','Dec99B'
    maskinfo = [maskname,'LN waveguide and poling litho mask','EM 6200','5" mask','1.0um process']
    waferradius,flatradius,workingradius = 38100,36400,34000 # 34000 o-ring radius, 31000 tfln workingradius
    folder = 'masks 2022/'+maskname+'/'+draftfolder+'/'
    os.makedirs(folder,exist_ok=True)
    mask = Element(maskname,layer='MASK')
    mask.info.font = r'Interstate-Bold.ttf'
    mask.info.minfeature = 1.0
    chipname = ['' for _ in range(19)]
    chiptype = ['BLOCK']*4 + ['WDM']*6
    chipname = [c+cc for (c,cc) in zip(chiptype,chipname)]
    mask.info.rows = rw = len(chipname)
    mask.info.chiplength = dx = 62000
    mask.info.chipwidth = dy = 6000
    chipid = [None for i in range(rw)]
    chipenable = [True for i in range(rw)]
    if chipnum is not None:
        chipenable = [i==chipnum for i in range(rw)]
    if isinstance(chipnum, (list, tuple)):
        chipenable = [True for i in range(rw)] if 0==len(chipnum) else [i in chipnum for i in range(rw)]
    def chipxy(i):
        dys = [dy for i in range(rw)]
        dx = mask.info.chiplength
        return V(-dx/2, sum(dys[:i]) - sum(dys)/2)
    for i in range(rw):
        chip = Chip(parent=mask)
        chipid[i] = chip.info.chipid
        # chip.info.chiplength = dx
        # chip.info.chiplength = dy
        if chipenable[i]:
            if chiptype[i]=='BLOCK':
                chip.addmetriconblockedchip() # leave 4 chip gap at bottom of wafer for metricon witness
            elif chiptype[i]=='WDM':
                # chip.addwdmchip8(chipinteraction=chipname[i])
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
def standardonelayerlithomask(draftfolder,chipnum=None,chipmap=False):
    maskname = 'Dec99A'
    maskinfo = [maskname,'LN waveguide litho mask','EM 6200','5" mask','1.0um process']
    waferradius,flatradius,workingradius = 38100,36400,34000 # 34000 o-ring radius, 31000 tfln workingradius
    folder = 'masks 2022/'+maskname+'/'+draftfolder+'/'
    os.makedirs(folder,exist_ok=True)
    mask = Element(maskname,layer='MASK')
    mask.info.font = r'Interstate-Bold.ttf'
    mask.info.minfeature = 1.0
    mask.info.rows = rw = 9
    mask.info.chiplength = dx = 62000
    mask.info.chipwidth = dy = 6000
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
            # chip.addwdmchip8(chipinteraction=chipname[i])
            chip.addmetriconblockedchip()
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
    for line in ['']+maskinfo+maskfiles+['']:
        print(line)
def gen(maskfunc): # decorator: gen(func)(args) or @gen/ndef func
    assert callable(maskfunc), 'usage: gen(maskfunc)(args)'
    def wrapper(draftfolder, *args, **kwargs):
        global dev,periodmag,scalemag,savepng
        savepng = 1
        dev,periodmag,scalemag = 1,20,10; maskfunc(draftfolder,chipmap=True)
        dev,periodmag,scalemag = 1, 1, 1; maskfunc(draftfolder,chipmap=True)
        dev,periodmag,scalemag = 0, 1, 1; maskfunc(draftfolder)
        dev,periodmag,scalemag = 1,20, 1; maskfunc(draftfolder)
    return wrapper

if __name__ == '__main__':
    standardonelayerlithomask('draft 1')
    # gen(standardonelayerlithomask)('draft 1')
    # standardtwolayerlithomask('draft 1',chipmap=True)
    # gen(standardtwolayerlithomask)('draft 2')

    ## notes ##
    # file size, save time reduction from grid rounding?
    # design a grating that can tell you exactly how much overpoling there is?
    # comment methods that modify self.x,y,width
    # offset guide x to see if it's working right (test nonzero self.x)
    # make test mask for unit testing with one each of all Element methods
    # subclass Guide, Group, Submount, Chip, Electrode
    # make parent,name mandatory in Element
    # test mask.roundoff(decimal=True)
    # move addchips to Chip
    # polys and rects add directly to chip don't show up in chip map, only full mask (see adddiceguides)
    # copy folder to shared, copy xlsx to u:
    #   "\\ADVRSVR2\AdvR Lab Shared Space\015 Litho and Electrode Masks\2021\Oct21A\draft 4\Oct21BELECTRODE.dxf"
    #   "\\ADVRSVR2\tonyr\masks.xlsx"

    # if dev:
    #     try:
    #         subprocess.Popen(['C:\\Program Files\\Common Files\\eDrawings2017\\eDrawings.exe', 'D:\\py\\mask\\mask.dxf'])
    #     except WindowsError:
    #         subprocess.Popen(['C:/Program Files/Common Files/eDrawings2020/eDrawings.exe', 'C:/py/mask/mask.dxf'])
