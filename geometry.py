#!/usr/bin/env python
#  -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import pi,sqrt,sin,cos,tan,arcsin,arccos,arctan,arctan2,log,exp,floor,ceil

import os,sys
# sys.path.insert(1, '../../sell')
# from waves import V,Vs,Wave
from grating import grating

# TODO:
# from collections import OrderedDict
# class DotDict(OrderedDict):
#     # dictionary with the ability to read/write attributes and treat keys as attributes
#     # example: point = DotDict(datum=y, squared=y*y, coord=x); if point.squared > threshold: point.isok = 1
#     def __init__(self, *args, **kwargs): super(DotDict, self).__init__(*args, **kwargs)
#     def __delattr__(self, name): del self[name]
#     def __getattr__(self, k): return self[k]
#     __setattr__ = OrderedDict.__setitem__
#     def __str__(self):              # (use namedtuple instead if immutable) # from collections import namedtuple # Point = namedtuple('Point', 'x y')
#         return ','.join(sorted([k+':'+str(v) for k,v in self.items()]))
class DotDict():                      # javascript-like object # code.activestate.com/recipes/52308/
    def __init__(self, **kwds):     # you can read/write the named attributes you just created, add others, del some, etc
        self.__dict__.update(kwds)  # example: point = Bunch(datum=y, squared=y*y, coord=x); if point.squared > threshold: point.isok = 1
    def __str__(self):              # (use namedtuple instead if immutable) # from collections import namedtuple # Point = namedtuple('Point', 'x y')
        return ','.join(sorted([k+':'+str(v) for k,v in self.__dict__.items()]))
from decimal import Decimal
class PPfloat(float):
    # subclass of float with pretty print (values printed rounded off to 'res')
    # prints precisely rounded floats to .dxf and .svg files
    def __new__(cls, value, res):
        instance = super().__new__(cls, value)
        instance.res = res
        return instance
    ## abs add and div floordiv lshift mod mul neg or pow radd rand rdiv rdivmod rfloordiv rlshift rmod rmul ror rpow rrshift rshift rsub rxor rtruediv sub truediv xor trunc
    ## see https://stackoverflow.com/a/19957897/12322780
    def __neg__(self): return PPfloat(super().__neg__(),self.res)
    def __add__(self,other): return PPfloat(super().__add__(other),self.res)
    def __mul__(self,other): return PPfloat(super().__mul__(other),self.res)
    def __radd__(self,other): return PPfloat(super().__radd__(other),self.res)
    def __rmul__(self,other): return PPfloat(super().__rmul__(other),self.res)
    def __sub__(self,other): return PPfloat(super().__sub__(other),self.res)
    def __div__(self,other): return PPfloat(super().__div__(other),self.res)
    def __rsub__(self,other): return PPfloat(super().__rsub__(other),self.res)
    def __rdiv__(self,other): return PPfloat(super().__rdiv__(other),self.res)
    def __str__(self):
        return str( Decimal(f'{self.res:g}')*Decimal(round(self/self.res)) )
from collections import OrderedDict
class OrderedDictFromCSV(OrderedDict): # https://stackoverflow.com/a/43566576
    def __init__(self, *args, **kwargs):
        if 1==len(args) and isinstance(args[0],str):
            ss = args[0].strip().split(',')
            d = {k.strip():v.strip() for k,v in zip(ss[::2],ss[1::2])} # print('d',d)
            def intorfloat(v): return int(v) if float(v)==round(float(v)) else float(v)
            d = {k:(intorfloat(v) if k not in ['qpmtype','filename','apodize'] else v) for k,v in d.items() if not (''==k and ''==v)}
            args = [d]
        super(OrderedDictFromCSV, self).__init__(*args, **kwargs) # we could just call super(OrderedDotDict, self).__init__(*args, **kwargs) but that won't get us nested dotdict objects
    def __delattr__(self, name): del self[name]
    def __getattr__(self, k): return self[k]
    __setattr__ = OrderedDict.__setitem__
    def __str__(self):
        def stringify(d): return ', '.join([k+':'+str(v) for k,v in d.items() if not ''==k==v]) # all items
        showorder = ['chipid','qpmtype','padcount','padgap','inputconnected','outputconnected']
        d0,d1 = {k:self.get(k,'') for k in showorder},{k:v for k,v in self.items() if k not in showorder}
        # return f'chipid:{self.get("chipid","")}, '+', '.join([k+':'+str(v) for k,v in d.items()]) # show chipid first
        return stringify(d0)+', '+stringify(d1)
def chipidtext(n,rows):
    nx,ny = n//rows,n%rows
    return f'{ny+1:02d}'+'ABCDEFGHIJ'[nx]
def svg2png(file):
    assert '.svg'==file[-4:]
    from time import process_time,sleep
    os.environ['path'] += r';c:\Octave\Octave-4.4.1\bin' # location of libcairo-2.dll, https://stackoverflow.com/a/60220855/12322780
    # doesn't work in 32 bit python, try: https://weasyprint.readthedocs.io/en/latest/install.html#windows
    import cairosvg
    with open(file,'rb') as f:
        # svgtext = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12" y2="16"/></svg>"""
        svgtext = f.read()
    timer = process_time()
    outfile = file[:-4]+'.png'
    cairosvg.svg2png(bytestring=svgtext,write_to=outfile)
    print(f'  svg2png in:"{file}" out:"{outfile}" elapsed:{process_time()-timer:.3f}sec')
def float2string(x,digits=8):
    return ('%.*f' % (digits,x)).rstrip('0').rstrip('.')
def dxftest(filename='dxftest.dxf'):
    drawing = ezdxf.new('AC1015')
    space = drawing.modelspace()
    drawing.styles.new('arial', dxfattribs={'font':'arial.ttf'})
    c0 = [(0,0),(100,0),(100,100),(0,0)]
    c1 = [(100+x,-50-y) for x,y in c0]
    polys = [c0,c1]
    r0,r1 = (0,50,50,150),(-50,-150,150,50)
    rects = [r0,r1]
    notes = [('Text.@#$%(^^)&*!',100,-100),('3µm',-50,-100),('α³√η°½♣→∞',-50,-50)]
    layer = 'MASK'
    for c in polys:
        cc = [(x,-y) for x,y in c]
        space.add_lwpolyline(cc, dxfattribs={'layer':layer,'closed':True,'color':42})
    for x,y,dx,dy in rects:
        xx,yy,dxx,dyy = x,-y,dx,-dy
        cc = [(xx,yy),(xx+dxx,yy),(xx+dxx,yy+dyy),(xx,yy+dyy),(xx,yy)]
        space.add_lwpolyline(cc, dxfattribs={'layer':layer,'closed':True,'color':252 if not 'MASK'==layer else 42})
    for note,x,y in notes:
        # dxfattribs={'color':16,'layer':'NOTES','height':50}
        dxfattribs={'color':16,'layer':'NOTES','height':25,'style':'arial'}
        space.add_text(note, dxfattribs=dxfattribs).set_pos((x,y), align='CENTER')
    drawing.saveas(filename)
def comparedxf(file1,file2,importall=False,finalize=False):
    import ezdxf
    from ezdxf.addons import Importer
    source = ezdxf.readfile(file1)
    target = ezdxf.readfile(file2) # ezdxf.new()
    importer = Importer(source, target)
    # import all entities from source modelspace into modelspace of the target drawing
    importer.import_modelspace()
    if importall:
        # import all paperspace layouts from source drawing
        importer.import_paperspace_layouts()
        # import all CIRCLE and LINE entities from source modelspace into an arbitrary target layout.
        # create target layout
        tblock = target.blocks.new('SOURCE_ENTS')
        # query source entities
        ents = source.modelspace().query('CIRCLE LINE')
        # import source entities into target block
        importer.import_entities(ents, tblock)
    if finalize:
        # This is ALWAYS the last & required step, without finalizing the target drawing is maybe invalid!
        # This step imports all additional required table entries and block definitions.
        importer.finalize()
    target.saveas('merged.dxf')
def defaultpadboundarys(gx,padcount):
    return {i:gx*i for i in range(padcount)},{i:gx*(i+1) for i in range(padcount)}
def oldfindpadboundarys(barstarts,barends,gx):
    n,starts,ends = 0,{},{}
    starts[0] = barstarts[0]
    for i in range(len(barstarts)):
        if barstarts[i]>=gx*(n+1):
            ends[n],starts[n+1],n = barends[i-1],barstarts[i],n+1
    ends[n] = barends[-1]
    return starts,ends
def findpadboundarys(barstarts,barends,gx,padcount):
    starts,ends = defaultpadboundarys(gx,padcount)
    n = int(barstarts[0]//gx)
    for i in range(len(barstarts)):
        if barstarts[i]>=gx*(n+1):
            ends[n],starts[n+1],n = barends[i-1],barstarts[i],n+1
    ends[n] = barends[-1]
    return starts,ends
def averageperiod(barstarts,barends,ignoregaps=True):
    barx = [b0/2.+b1/2. for b0,b1 in zip(barstarts,barends)]
    dx = [b1-b0 for b0,b1 in zip(barx[:-1],barx[1:])]
    if ignoregaps:
        averageperiodwithgaps = sum(dx)/len(dx)
        dx = [b1-b0 for b0,b1 in zip(barx[:-1],barx[1:]) if b1-b0<2*averageperiodwithgaps]
    return sum(dx)/len(dx)
def makepadtext(mask,chipid,period,dc,padcount):
    def roundtondigits(period,n): # has subtleties due to edge cases like 99.949
        np,rp = 5,round(period,6)
        while len(str(rp))>n+1 and np>=0: np,rp = np-1,round(period,np)
        s = str(rp).replace('.','')
        while len(s)<n: s += '0'
        assert len(s)==n
        return s
    tt = ['↓'+s for s in roundtondigits(period,4)] # print ' '.join(tt)
    tperiod,tdc = ('%.4f'%period)[:6],str(int(round(dc*100)))+'%'
    padtext = [mask,chipid+'',tperiod,tdc,tt[0],tt[1],tt[2],tt[3],tperiod,tdc]
    while len(padtext)<padcount: padtext += [mask,chipid+'',tperiod,tdc]
    return padtext[:padcount]
def svgtest():
    import svgwrite
    dwg = svgwrite.Drawing('mask.svg', viewBox='-500 -100 1000 200')#, profile='tiny')
    dwg.add(dwg.line((0, 0), (10, 10), stroke=svgwrite.rgb(10, 10, 50, '%')))
    dwg.add(dwg.text('Test', insert=(0, 10), fill='darkred'))
    print(mask2.rects==mask.rects)
    dwg.save()
def savedxf(filename='',layername='MASK',polys=[[(0,0),(1,0),(1,1)],[(2,0),(2,2),(0,2)]],rects=[(1,3,2,2)],filledpolys=False):
    import ezdxf
    if not filename: filename = 'mask'
    filename = filename.rstrip('.dxf')+'.dxf'
    drawing = ezdxf.new('AC1015')
    space = drawing.modelspace()
    drawing.layers.new(name=layername) # colors = [16,42,75,146,207,254] # rogbvg # 0=black,7=white
    if filledpolys:
        hatch = space.add_hatch(color=42,dxfattribs={'layer':'MASK'})  # by default a solid fill hatch with fill color=7 (white/black)
        with hatch.edit_boundary() as boundary:  # edit boundary path (context manager) # every boundary path is always a 2D element # vertex format for the polyline path is: (x, y[, bulge]) # there are no bulge values in this example  #boundary.add_polyline_path([(0, 0), (0, 5), (5, 10), (10, 10)], is_closed=1)
            for c in polys:
                boundary.add_polyline_path([(x,-y) for x,y in c], is_closed=1)
            for x,y,dx,dy in rects:
                boundary.add_polyline_path([(x,-y),(x+dx,-y),(x+dx,-y-dy),(x,-y-dy),(x,-y)], is_closed=1) #drawing.add(dxf.rectangle((x,-y-dy),dx,dy, color=colors[4], bgcolor=colors[4], layer='MASK')) # drawing.add(dxf.rectangle((0,0),10,1, color=colors[2], layer='MASK')) # https://pythonhosted.org/dxfwrite/entities/polyline.html # drawing.add(dxf.line((0,0), (10,1), color=7)) #points = [(0, 0), (3, 0), (6, 3), (6, 6)]; 
    else:
        for c in polys:
            space.add_lwpolyline([(x,-y) for x,y in c], dxfattribs={'layer':layername,'closed':True,'color':42})
        for x,y,dx,dy in rects:
            space.add_lwpolyline([(x,-y),(x+dx,-y),(x+dx,-y-dy),(x,-y-dy),(x,-y)], dxfattribs={'layer':layername,'closed':True,'color':75})
    drawing.saveas(filename)
def savedxfwrite(filename='',polys=[[(0,2),(3,2),(6,2.3),(9,2.3),(0,2.6),(0,2)]],rects=[(0,0,10,1)]):
    from dxfwrite import DXFEngine as dxf # pip install dxfwrite
    import dxfwrite.const as const
    if not filename: filename = 'mask'
    colors = [16,42,75,146,207,254] # rogbvg # 0=black,7=white
    drawing = dxf.drawing(filename.rstrip('.dxf')+'.dxf')
    drawing.add_layer('MASK', color=colors[0])
    #print 'const.POLYLINE_3D_POLYLINE',const.POLYLINE_3D_POLYLINE
    for x,y,dx,dy in rects:
        drawing.add(dxf.rectangle((x,-y-dy),dx,dy, color=colors[4], bgcolor=colors[4], layer='MASK')) # drawing.add(dxf.rectangle((0,0),10,1, color=colors[2], layer='MASK')) # https://pythonhosted.org/dxfwrite/entities/polyline.html # drawing.add(dxf.line((0,0), (10,1), color=7))
    for c in polys:
        polyline = dxf.polyline(color=colors[3], layer='MASK', flags=0) # flags=const.POLYLINE_3D_POLYLINE is default # const.POLYLINE_3D_POLYLINE=8 in const.py # https://pythonhosted.org/ezdxf/entities.html#polyline
        polyline.add_vertices([(x,-y) for x,y in c]) # polyline.add_vertices( [(0,2.0), (3,2.0), (6,2.3), (9,2.3), (0,2.6), (0,2.0)] )
        polyline.close()
        drawing.add(polyline)
    # drawing.add_layer('TEXT', color=colors[1]); drawing.add(dxf.text('Test note', insert=(0,0), layer='TEXT'))
    # drawing.add(dxf.line((0,0), (10,1), color=7)); drawing.add(dxf.rectangle((0,0),10,1, color=7))
    drawing.save()
def plotcurves(pss,screencoordinates=False,scale=1):
    def curvestocurve(pss):
        return [p for ps in pss for p in list(ps)+[(float('nan'),float('nan'))] ]
    xs,ys = np.array(list(zip(*curvestocurve(pss))))
    plt.plot(xs,ys,'darkred'); plt.ylabel('y (µm)'); plt.xlabel('x (µm)')
    if screencoordinates: plt.gca().invert_yaxis()
    plt.gca().set_aspect(1)
    fig = plt.gcf()
    sx,sy = scale if hasattr(scale,'__len__') else (scale,scale)
    fig.set_size_inches((fig.get_size_inches()[0]*sx, fig.get_size_inches()[1]*sy))
    plt.show()
def letterset():
    return ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
def textcurves(ss,vertical=False,skew=0,screencoordinates=False,cursor=(0,0)):
    if not ss:return []
    if vertical:
        return [[(-y,x) for x,y in c] for c in textcurves(ss,False,skew,screencoordinates,cursor)]
    if screencoordinates:
        return [[(x,-y) for x,y in c] for c in textcurves(ss,vertical,skew,False,cursor)]
    if '\n'==ss[0]:
        cursor = (0,cursor[1]-84-42)
    # default data from autocad font file AU140S00.SHP
    lettercurves = {' ':[],'$':[[(34,0),(34,-8),(22,-8),(22,0),(2,0),(2,42),(34,0)],[(36,0),(43,0),(46,1),(52,4),(54,6),(58,12),(60,18),(60,24),(58,30),(56,33),(54,35),(23,75),(22,76),(21,78),(21,80),(22,82),(24,84),(17,84),(11,82),(9,81),(4,77),(1,71),(0,66),(0,60),(1,57),(4,52),(38,8),(39,7),(39,3),(37,1),(36,1),(36,0)],[(56,46),(56,84),(38,84),(38,92),(26,92),(26,84),(56,46)]],'(':[[(24,-2),(17,5),(11,14),(7,22),(4,31),(2,41),(2,51),(4,60),(7,70),(11,78),(17,87),(23,94),(24,94),(24,-2)]],',':[[(10,0),(8,0),(2,4),(1,6),(0,9),(0,14),(1,18),(3,21),(6,23),(10,24),(15,24),(17,23),(20,21),(22,19),(24,16),(24,9),(23,7),(9,-16),(0,-16),(10,0)]],'0':[[(30,0),(25,1),(17,5),(13,8),(9,12),(6,16),(4,19),(2,26),(0,35),(0,44),(4,60),(6,64),(12,72),(16,75),(20,77),(25,79),(30,80),(30,0)],[(38,0),(43,1),(51,5),(55,8),(59,12),(62,16),(64,19),(66,26),(68,35),(68,44),(64,60),(62,64),(59,68),(52,75),(47,78),(43,79),(38,80),(38,0)]],'4':[[(30,17),(0,17),(30,80),(30,17)],[(36,0),(64,0),(64,17),(72,17),(72,26),(64,26),(64,80),(36,80),(36,0)]],'8':[[(26,42),(19,42),(15,41),(12,40),(9,38),(4,33),(2,30),(1,27),(0,23),(0,17),(2,11),(4,8),(8,4),(11,2),(17,0),(26,0),(26,42)],[(26,44),(19,44),(14,46),(12,47),(6,53),(5,55),(4,58),(4,66),(5,69),(6,71),(12,77),(14,78),(19,80),(26,80),(26,44)],[(34,42),(41,42),(45,41),(48,40),(51,38),(56,33),(58,30),(59,27),(60,23),(60,17),(58,11),(56,8),(52,4),(49,2),(43,0),(34,0),(34,42)],[(34,44),(41,44),(46,46),(48,47),(54,53),(55,55),(56,58),(56,66),(55,69),(54,71),(48,77),(46,78),(41,80),(34,80),(34,44)]],'<':[[(12,14),(0,40),(12,66),(28,66),(18,40),(28,14),(12,14)]],'@':[[(42,24),(32,24),(29,25),(23,28),(20,32),(19,34),(18,39),(18,44),(21,50),(23,52),(29,55),(32,56),(42,56),(42,24)],[(48,24),(48,36),(54,36),(54,45),(52,51),(51,53),(49,56),(45,60),(42,62),(39,63),(33,64),(27,63),(24,62),(18,58),(16,56),(14,53),(12,47),(11,43),(11,36),(13,30),(15,27),(20,22),(27,18),(30,17),(48,17),(48,6),(29,6),(19,9),(11,15),(8,18),(5,22),(3,26),(1,32),(0,37),(0,42),(1,47),(2,51),(4,56),(7,61),(14,68),(18,70),(23,72),(27,73),(32,74),(37,74),(42,73),(47,71),(51,69),(56,66),(59,63),(62,59),(65,54),(68,45),(68,24),(48,24)]],'D':[[(36,0),(45,0),(49,1),(53,3),(60,8),(63,11),(67,19),(70,27),(72,36),(72,45),(71,53),(68,63),(65,70),(60,76),(56,79),(53,81),(49,83),(45,84),(36,84),(36,0)],[(28,0),(28,84),(0,84),(0,0),(28,0)]],'H':[[(0,0),(30,0),(30,42),(38,42),(38,0),(68,0),(68,84),(38,84),(38,50),(30,50),(30,84),(0,84),(0,0)]],'L':[[(28,0),(28,84),(0,84),(0,0),(28,0)],[(32,0),(60,0),(60,40),(32,0)]],'P':[[(28,84),(0,84),(0,0),(28,0),(28,84)],[(36,84),(48,84),(51,83),(58,80),(63,75),(65,72),(67,68),(68,64),(68,54),(67,51),(65,47),(63,44),(57,38),(50,35),(47,34),(36,34),(36,84)]],'T':[[(18,84),(0,84),(0,48),(18,84)],[(50,84),(22,84),(22,0),(50,0),(50,84)],[(54,84),(72,84),(72,48),(54,84)]],'X':[[(26,0),(0,0),(14,34),(26,0)],[(34,84),(2,84),(34,0),(66,0),(34,84)],[(42,84),(68,84),(54,50),(42,84)]],'\\':[[(60,0),(48,0),(0,80),(12,80),(60,0)]],'`':[[(12,80),(10,80),(8,79),(5,77),(3,74),(1,70),(1,64),(3,60),(5,57),(8,55),(10,54),(12,54),(12,80)]],'d':[[(32,0),(32,84),(56,84),(56,0),(32,0)],[(26,60),(23,60),(19,59),(13,57),(10,55),(5,50),(3,47),(1,41),(0,33),(0,25),(2,17),(3,14),(6,8),(9,5),(12,3),(16,1),(19,0),(26,0),(26,60)]],'h':[[(0,0),(0,84),(23,84),(23,0),(0,0)],[(29,0),(52,0),(52,48),(51,51),(50,53),(49,54),(43,58),(29,60),(29,0)]],'l':[[(0,0),(0,84),(24,84),(24,0),(0,0)]],'p':[[(24,60),(24,-24),(0,-24),(0,60),(24,60)],[(30,0),(33,0),(37,1),(43,3),(46,5),(51,10),(53,13),(55,19),(56,27),(56,35),(54,43),(53,46),(50,52),(47,55),(44,57),(40,59),(37,60),(30,60),(30,0)]],'t':[[(8,0),(8,52),(0,52),(32,84),(32,60),(40,60),(40,52),(32,52),(32,0),(8,0)]],'x':[[(28,0),(0,0),(14,26),(28,0)],[(28,60),(60,0),(32,0),(0,60),(28,60)],[(32,60),(60,60),(46,34),(32,60)]],'|':[[(0,38),(16,38),(16,18),(0,18),(0,38)],[(0,50),(0,70),(16,70),(16,50),(0,50)]],'#':[[(8,20),(8,28),(0,28),(0,36),(8,36),(8,44),(0,44),(0,52),(8,52),(8,60),(16,60),(16,52),(24,52),(24,60),(32,60),(32,52),(40,52),(40,44),(32,44),(32,36),(40,36),(40,28),(32,28),(32,20),(24,20),(24,28),(16,28),(16,20),(8,20)],[(16,36),(16,44),(24,44),(24,36),(16,36)]],"'":[[(0,80),(2,80),(4,79),(7,77),(9,74),(11,70),(11,64),(9,60),(7,57),(4,55),(2,54),(0,54),(0,80)]],'+':[[(10,24),(10,34),(0,34),(0,50),(10,50),(10,60),(26,60),(26,50),(36,50),(36,34),(26,34),(26,24),(10,24)]],'/':[[(0,0),(12,0),(60,80),(48,80),(0,0)]],'3':[[(13,0),(11,0),(8,1),(6,2),(4,4),(2,7),(1,9),(1,14),(2,17),(3,19),(4,20),(10,24),(16,24),(22,20),(23,19),(24,17),(25,14),(25,9),(24,7),(22,4),(20,2),(18,1),(15,0),(13,0)],[(14,56),(12,56),(9,57),(7,58),(5,60),(3,63),(2,65),(2,70),(3,73),(4,75),(5,76),(11,80),(17,80),(23,76),(24,75),(25,73),(26,70),(26,65),(25,63),(23,60),(21,58),(19,57),(16,56),(14,56)],[(26,80),(28,79),(32,71),(32,52),(31,50),(27,46),(25,45),(20,45),(20,40),(25,40),(27,39),(31,35),(32,33),(32,6),(31,4),(27,0),(26,0),(43,0),(49,2),(51,3),(56,8),(59,13),(60,16),(60,24),(59,27),(56,32),(51,37),(49,38),(37,42),(32,42),(37,42),(46,45),(51,48),(54,52),(55,54),(56,57),(56,65),(53,72),(51,74),(47,77),(39,79),(34,80),(26,80)]],'7':[[(2,0),(36,0),(48,54),(2,0)],[(0,56),(0,78),(48,78),(48,56),(0,56)]],';':[[(10,0),(8,0),(2,4),(1,6),(0,9),(0,14),(1,18),(3,21),(6,23),(10,24),(15,24),(17,23),(20,21),(22,19),(24,16),(24,9),(23,7),(9,-16),(0,-16),(10,0)],[(0,44),(0,46),(1,49),(2,51),(4,53),(7,55),(9,56),(14,56),(17,55),(19,54),(20,53),(24,47),(24,41),(20,35),(19,34),(17,33),(14,32),(9,32),(7,33),(4,35),(2,37),(1,39),(0,42),(0,44)]],'?':[[(18,12),(18,14),(19,17),(20,19),(22,21),(25,23),(27,24),(32,24),(35,23),(37,22),(38,21),(42,15),(42,9),(38,3),(37,2),(35,1),(32,0),(27,0),(25,1),(22,3),(20,5),(19,7),(18,10),(18,12)],[(28,30),(32,30),(36,31),(40,33),(46,37),(49,40),(51,43),(53,47),(55,55),(55,59),(53,67),(51,71),(49,74),(46,77),(40,81),(36,83),(32,84),(24,84),(28,80),(28,30)],[(13,60),(11,60),(8,61),(6,62),(4,64),(2,67),(1,69),(1,74),(2,77),(3,79),(4,80),(10,84),(16,84),(22,80),(23,79),(24,77),(25,74),(25,69),(24,67),(22,64),(20,62),(18,61),(15,60),(13,60)]],'C':[[(34,0),(30,0),(21,2),(14,6),(11,9),(8,13),(4,21),(2,27),(0,37),(0,46),(2,55),(4,62),(8,71),(14,77),(21,82),(25,83),(30,84),(34,84),(34,0)],[(38,0),(64,32),(64,0),(38,0)],[(38,84),(64,84),(64,52),(38,84)]],'G':[[(36,0),(27,0),(23,1),(19,3),(12,8),(9,11),(5,19),(2,27),(0,36),(0,45),(1,53),(4,63),(7,70),(12,76),(16,79),(19,81),(23,83),(27,84),(36,84),(36,0)],[(44,0),(44,40),(72,40),(72,24),(71,19),(69,15),(67,12),(64,8),(60,5),(57,3),(48,0),(44,0)],[(40,84),(68,84),(68,52),(40,84)]],'K':[[(28,0),(0,0),(0,84),(28,84),(28,0)],[(36,0),(72,0),(38,46),(72,84),(36,84),(36,0)]],'O':[[(32,0),(27,0),(23,1),(19,3),(11,9),(9,13),(4,21),(2,28),(0,37),(0,46),(2,55),(4,62),(8,71),(17,80),(24,83),(28,84),(32,84),(32,0)],[(40,0),(44,0),(48,1),(56,5),(59,7),(62,11),(64,14),(68,21),(71,31),(72,38),(72,47),(70,56),(67,64),(63,72),(57,78),(53,81),(45,83),(40,84),(40,0)]],'S':[[(32,0),(2,0),(2,40),(32,0)],[(36,0),(43,0),(46,1),(52,4),(54,6),(58,12),(60,18),(60,24),(58,30),(56,33),(54,35),(23,75),(22,76),(21,78),(21,80),(22,82),(24,84),(17,84),(11,82),(9,81),(4,77),(1,71),(0,66),(0,60),(1,57),(4,52),(38,8),(39,7),(39,3),(37,1),(36,1),(36,0)],[(56,48),(56,84),(28,84),(56,48)]],'W':[[(72,84),(86,28),(100,84),(72,84)],[(36,84),(58,0),(79,0),(83,18),(66,84),(36,84)],[(30,84),(48,18),(43,0),(22,0),(0,84),(30,84)]],'[':[[(32,0),(32,20),(20,20),(20,64),(32,64),(32,84),(0,84),(0,0),(32,0)]],'_':[[(0,-6),(120,-6),(0,-6)]],'c':[[(28,60),(16,57),(12,54),(6,48),(4,45),(2,41),(0,32),(0,28),(2,19),(3,16),(9,8),(16,3),(28,0),(32,0),(37,1),(41,2),(45,4),(48,6),(54,12),(54,21),(52,18),(49,14),(46,11),(43,9),(36,6),(28,6),(28,60)],[(32,60),(56,60),(56,32),(32,60)]],'g':[[(26,-8),(26,-6),(25,-3),(24,-1),(22,1),(19,3),(17,4),(12,4),(9,3),(7,2),(6,1),(2,-5),(2,-11),(6,-17),(7,-18),(9,-19),(12,-20),(17,-20),(19,-19),(22,-17),(24,-15),(25,-13),(26,-10),(26,-8)],[(32,60),(32,-14),(31,-16),(28,-19),(27,-19),(36,-20),(39,-20),(45,-18),(48,-16),(52,-12),(54,-9),(56,-3),(56,60),(32,60)],[(26,60),(26,8),(19,8),(12,11),(9,13),(6,16),(2,22),(0,29),(0,39),(1,43),(2,46),(6,52),(9,55),(12,57),(19,60),(26,60)]],'k':[[(23,0),(23,84),(0,84),(0,0),(23,0)],[(29,0),(29,60),(56,60),(32,32),(60,0),(29,0)]],'o':[[(27,0),(23,0),(19,1),(16,2),(9,7),(7,9),(5,12),(1,20),(1,23),(0,30),(1,37),(2,41),(3,44),(7,50),(13,56),(20,59),(27,59),(27,0)],[(33,0),(37,0),(41,1),(44,2),(51,7),(53,9),(55,12),(59,20),(59,23),(60,30),(59,37),(58,41),(57,44),(53,50),(47,56),(40,59),(33,59),(33,0)]],'s':[[(26,0),(4,24),(4,0),(26,0)],[(30,0),(32,2),(32,4),(30,6),(6,32),(3,38),(2,41),(2,46),(6,54),(8,56),(14,59),(17,60),(18,60),(17,59),(16,57),(16,56),(17,55),(46,24),(48,22),(50,18),(51,13),(50,9),(49,7),(47,4),(44,2),(42,1),(38,0),(30,0)],[(46,34),(46,60),(22,60),(46,34)]],'w':[[(18,0),(0,60),(25,60),(38,20),(30,0),(18,0)],[(62,0),(50,0),(30,60),(55,60),(69,18),(62,0)],[(72,26),(60,60),(84,60),(72,26)]],'{':[[(40,0),(40,20),(28,20),(28,64),(40,64),(40,84),(21,84),(13,80),(11,77),(8,71),(8,48),(6,44),(2,42),(0,42),(2,42),(6,40),(8,36),(8,13),(12,5),(15,3),(21,0),(40,0)]],'"':[[(12,84),(8,82),(5,80),(2,74),(1,70),(2,66),(5,60),(8,58),(12,56),(12,84)],[(20,84),(24,82),(27,80),(30,74),(31,70),(30,66),(27,60),(24,58),(20,56),(20,84)]],'&':[[(38,2),(35,1),(27,-1),(24,-1),(16,1),(7,7),(3,13),(2,16),(0,24),(0,27),(2,35),(8,44),(38,2)],[(46,0),(80,0),(65,21),(76,28),(76,39),(60,28),(24,76),(18,70),(16,67),(14,63),(13,60),(13,48),(16,40),(46,0)],[(50,50),(52,51),(56,55),(58,59),(60,64),(60,67),(59,72),(58,74),(55,79),(48,83),(43,84),(40,84),(35,83),(33,82),(29,78),(50,50)]],'*':[[(24,26),(24,24),(23,23),(21,22),(19,22),(17,23),(16,24),(16,26),(18,34),(16,36),(8,30),(4,30),(2,32),(2,37),(3,38),(4,38),(14,40),(14,44),(4,46),(3,46),(2,47),(2,52),(4,54),(8,54),(16,46),(18,48),(16,58),(16,60),(17,61),(19,62),(21,62),(23,61),(24,60),(24,58),(22,48),(24,46),(32,54),(36,54),(38,52),(38,47),(37,46),(36,46),(26,44),(26,40),(36,38),(37,38),(38,37),(38,32),(36,30),(32,30),(24,36),(22,34),(24,26)]],'.':[[(0,12),(0,14),(1,17),(2,19),(4,21),(7,23),(9,24),(14,24),(17,23),(19,22),(20,21),(24,15),(24,9),(20,3),(19,2),(17,1),(14,0),(9,0),(7,1),(4,3),(2,5),(1,7),(0,10),(0,12)]],'2':[[(6,0),(6,22),(62,22),(62,0),(6,0)],[(27,56),(26,54),(21,51),(19,50),(10,50),(8,51),(2,57),(1,61),(1,69),(2,71),(8,77),(10,78),(19,78),(23,76),(26,74),(29,68),(29,60),(27,56)],[(26,80),(38,80),(42,79),(46,77),(52,73),(55,70),(57,67),(59,63),(61,55),(61,51),(59,43),(58,40),(55,36),(52,33),(46,29),(42,27),(38,26),(28,26),(29,26),(33,29),(34,32),(34,71),(33,74),(32,76),(28,79),(26,80)]],'6':[[(34,0),(29,1),(23,3),(18,6),(14,9),(10,13),(7,17),(4,23),(2,33),(2,47),(4,57),(7,63),(10,67),(13,70),(22,77),(27,79),(32,80),(38,80),(35,76),(34,72),(34,0)],[(52,56),(50,56),(47,57),(45,58),(43,60),(41,63),(40,65),(40,70),(41,73),(42,75),(43,76),(49,80),(55,80),(61,76),(62,75),(63,73),(64,70),(64,65),(63,63),(61,60),(59,58),(57,57),(54,56),(52,56)],[(40,52),(44,52),(48,51),(51,50),(58,45),(60,43),(63,38),(65,32),(66,28),(66,24),(65,20),(63,13),(60,9),(58,7),(51,2),(48,1),(44,0),(40,0),(40,52)]],':':[[(0,12),(0,14),(1,17),(2,19),(4,21),(7,23),(9,24),(14,24),(17,23),(19,22),(20,21),(24,15),(24,9),(20,3),(19,2),(17,1),(14,0),(9,0),(7,1),(4,3),(2,5),(1,7),(0,10),(0,12)],[(0,44),(0,46),(1,49),(2,51),(4,53),(7,55),(9,56),(14,56),(17,55),(19,54),(20,53),(24,47),(24,41),(20,35),(19,34),(17,33),(14,32),(9,32),(7,33),(4,35),(2,37),(1,39),(0,42),(0,44)]],'>':[[(16,14),(28,40),(16,66),(0,66),(10,40),(0,14),(16,14)]],'B':[[(28,0),(28,84),(0,84),(0,0),(28,0)],[(36,0),(36,84),(47,84),(50,83),(55,81),(59,77),(62,72),(63,69),(63,61),(62,58),(61,56),(59,53),(57,51),(52,48),(50,47),(47,46),(44,46),(53,43),(56,41),(58,39),(62,33),(63,27),(63,24),(62,18),(62,15),(61,12),(59,9),(54,4),(52,3),(43,0),(36,0)]],'F':[[(32,0),(32,84),(0,84),(0,0),(32,0)],[(36,44),(52,28),(52,58),(36,44)],[(36,84),(64,84),(64,50),(36,84)]],'J':[[(0,15),(0,13),(1,9),(3,6),(8,3),(10,2),(16,2),(20,4),(23,6),(25,9),(26,13),(26,20),(25,22),(23,25),(21,27),(18,29),(16,30),(10,30),(8,29),(5,27),(3,25),(0,20),(0,15)],[(24,0),(26,1),(27,2),(29,5),(30,7),(30,84),(60,84),(60,22),(59,18),(57,14),(55,11),(52,7),(46,3),(42,1),(38,0),(24,0)]],'N':[[(26,0),(0,64),(0,0),(26,0)],[(64,0),(64,8),(30,84),(0,84),(0,76),(34,0),(64,0)],[(64,20),(38,84),(64,84),(64,20)]],'R':[[(28,0),(28,84),(0,84),(0,0),(28,0)],[(36,0),(36,84),(47,84),(56,81),(59,79),(61,77),(63,74),(64,72),(66,66),(66,59),(64,53),(60,48),(58,46),(52,43),(49,42),(39,42),(72,0),(36,0)]],'V':[[(46,0),(50,13),(30,84),(0,84),(24,0),(46,0)],[(54,24),(36,84),(72,84),(54,24)]],'Z':[[(0,0),(46,84),(80,84),(34,0),(0,0)],[(8,30),(8,84),(38,84),(8,30)],[(43,0),(72,0),(72,55),(43,0)]],'^':[[(0,24),(0,44),(20,52),(40,44),(40,24),(20,32),(0,24)]],'b':[[(24,0),(24,84),(0,84),(0,0),(24,0)],[(30,60),(33,60),(37,59),(43,57),(46,55),(51,50),(53,47),(55,41),(56,33),(56,25),(54,17),(53,14),(50,8),(47,5),(44,3),(40,1),(37,0),(30,0),(30,60)]],'f':[[(6,0),(30,0),(30,52),(40,52),(40,59),(24,59),(22,60),(16,64),(15,66),(14,70),(14,75),(17,81),(18,82),(20,83),(20,84),(17,83),(15,82),(9,76),(8,74),(6,68),(6,59),(0,59),(0,52),(6,52),(6,0)],[(40,74),(40,71),(38,68),(35,65),(33,64),(28,64),(25,65),(21,69),(20,72),(20,76),(21,79),(25,83),(28,84),(33,84),(35,83),(38,80),(40,77),(40,74)]],'j':[[(24,78),(26,76),(26,69),(25,67),(23,65),(19,63),(14,62),(12,62),(7,64),(5,65),(3,67),(2,69),(2,75),(3,77),(5,79),(8,81),(10,82),(16,82),(20,81),(22,80),(24,78)],[(26,56),(26,0),(25,-3),(23,-7),(19,-13),(16,-16),(9,-19),(6,-20),(2,-20),(2,56),(26,56)]],'n':[[(23,0),(23,60),(0,60),(0,0),(23,0)],[(29,0),(52,0),(52,50),(51,53),(50,55),(47,58),(45,59),(42,60),(29,60),(29,0)]],'r':[[(22,60),(0,60),(0,0),(22,0),(22,60)],[(30,56),(29,54),(28,51),(28,49),(29,46),(31,43),(34,40),(36,39),(39,38),(41,38),(44,39),(46,40),(50,43),(51,46),(52,50),(52,51),(51,55),(49,58),(48,59),(45,61),(42,62),(38,62),(35,61),(33,60),(32,59),(30,56)]],'v':[[(24,0),(0,60),(26,60),(42,16),(36,0),(24,0)],[(46,24),(32,60),(60,60),(46,24)]],'z':[[(6,20),(6,60),(30,60),(6,20)],[(28,0),(0,0),(36,60),(64,60),(28,0)],[(34,0),(58,40),(58,0),(34,0)]],'~':[[(0,50),(4,53),(8,55),(12,56),(17,57),(26,57),(31,55),(39,51),(42,48),(48,44),(52,42),(56,41),(64,41),(71,43),(77,47),(80,50),(80,34),(75,31),(64,28),(59,28),(53,29),(48,30),(38,35),(33,39),(32,41),(26,45),(21,46),(18,46),(12,45),(6,41),(2,37),(0,34),(0,50)]],'!':[[(2,12),(2,14),(3,17),(4,19),(6,21),(9,23),(11,24),(16,24),(19,23),(21,22),(22,21),(26,15),(26,9),(22,3),(21,2),(19,1),(16,0),(11,0),(9,1),(6,3),(4,5),(3,7),(2,10),(2,12)],[(14,26),(0,84),(28,84),(14,26)]],'%':[[(22,32),(19,32),(13,34),(10,36),(6,40),(4,43),(1,52),(1,61),(2,65),(4,69),(6,72),(10,76),(13,78),(19,80),(22,80),(22,32)],[(28,32),(31,32),(37,34),(40,36),(44,40),(46,43),(49,52),(49,61),(48,65),(46,69),(44,72),(40,76),(37,78),(31,80),(28,80),(28,32)],[(24,0),(36,0),(88,80),(76,80),(24,0)],[(84,0),(81,0),(75,2),(72,4),(68,8),(66,11),(63,20),(63,29),(64,33),(66,37),(68,40),(72,44),(75,46),(81,48),(84,48),(84,0)],[(90,0),(93,0),(99,2),(102,4),(106,8),(108,11),(111,20),(111,29),(110,33),(108,37),(106,40),(102,44),(99,46),(93,48),(90,48),(90,0)]],')':[[(0,-2),(7,5),(13,14),(17,22),(20,31),(22,41),(22,51),(20,60),(17,70),(13,78),(7,87),(1,94),(0,94),(0,-2)]],'-':[[(0,36),(0,48),(28,48),(28,36),(0,36)]],'1':[[(10,0),(10,58),(0,52),(0,60),(40,80),(40,0),(10,0)]],'5':[[(12,0),(10,0),(7,1),(5,2),(3,4),(1,7),(0,9),(0,14),(1,17),(2,19),(3,20),(9,24),(15,24),(21,20),(22,19),(23,17),(24,14),(24,9),(23,7),(21,4),(19,2),(17,1),(14,0),(12,0)],[(27,0),(38,0),(42,1),(45,2),(52,7),(54,9),(57,14),(59,20),(60,24),(60,28),(59,32),(57,39),(54,43),(52,45),(45,50),(42,51),(38,52),(21,52),(21,46),(26,46),(30,44),(32,40),(32,6),(31,4),(27,0)],[(14,46),(14,58),(56,58),(56,78),(6,78),(6,46),(14,46)]],'9':[[(34,80),(39,79),(45,77),(50,74),(54,71),(58,67),(61,63),(64,57),(66,47),(66,33),(64,23),(61,17),(58,13),(55,10),(46,3),(41,1),(36,0),(30,0),(33,4),(34,8),(34,80)],[(16,24),(18,24),(21,23),(23,22),(25,20),(27,17),(28,15),(28,10),(27,7),(26,5),(25,4),(19,0),(13,0),(7,4),(6,5),(5,7),(4,10),(4,15),(5,17),(7,20),(9,22),(11,23),(14,24),(16,24)],[(28,28),(24,28),(20,29),(17,30),(10,35),(8,37),(5,42),(3,48),(2,52),(2,56),(3,60),(5,67),(8,71),(10,73),(17,78),(20,79),(24,80),(28,80),(28,28)]],'=':[[(0,46),(0,56),(28,56),(28,46),(0,46)],[(0,38),(0,28),(28,28),(28,38),(0,38)]],'A':[[(0,0),(34,0),(28,28),(36,28),(42,0),(72,0),(54,84),(22,84),(34,36),(26,36),(19,70),(0,0)]],'E':[[(32,0),(32,84),(0,84),(0,0),(32,0)],[(36,0),(64,0),(64,36),(36,0)],[(36,44),(52,28),(52,58),(36,44)],[(36,84),(62,84),(62,52),(36,84)]],'I':[[(0,0),(32,0),(32,84),(0,84),(0,0)]],'M':[[(28,0),(14,56),(0,0),(28,0)],[(64,0),(42,84),(21,84),(17,66),(34,0),(64,0)],[(70,0),(52,66),(57,84),(78,84),(100,0),(70,0)]],'Q':[[(32,0),(27,0),(23,1),(19,3),(11,9),(9,13),(4,21),(2,28),(0,37),(0,46),(2,55),(4,62),(8,71),(17,80),(24,83),(28,84),(32,84),(32,0)],[(40,-20),(40,84),(44,84),(48,83),(55,80),(59,77),(61,75),(64,71),(68,63),(70,56),(72,46),(72,37),(70,29),(67,19),(63,13),(60,9),(57,6),(53,3),(48,1),(44,1),(62,-20),(40,-20)]],'U':[[(28,0),(20,0),(17,1),(13,3),(7,7),(5,9),(3,12),(1,17),(0,20),(0,84),(28,84),(28,0)],[(36,0),(44,0),(47,1),(51,3),(57,7),(59,9),(61,12),(63,17),(64,20),(64,84),(36,84),(36,0)]],'Y':[[(52,0),(52,37),(32,84),(0,84),(22,32),(22,0),(52,0)],[(54,46),(38,84),(72,84),(54,46)]],']':[[(0,0),(0,20),(12,20),(12,64),(0,64),(0,84),(32,84),(32,0),(0,0)]],'a':[[(16,36),(13,36),(10,38),(7,41),(6,43),(6,48),(7,51),(11,55),(14,56),(18,56),(21,55),(25,51),(26,48),(26,43),(25,41),(22,38),(19,36),(16,36)],[(26,0),(14,0),(11,1),(5,4),(2,8),(1,10),(0,15),(0,20),(3,26),(5,28),(11,31),(14,32),(26,32),(26,0)],[(32,0),(56,0),(56,43),(54,49),(52,52),(48,56),(45,58),(39,60),(24,60),(28,58),(32,50),(32,0)]],'e':[[(28,60),(16,57),(12,54),(6,48),(4,45),(2,41),(0,32),(0,28),(2,19),(3,16),(9,8),(16,3),(28,0),(32,0),(37,1),(41,2),(45,4),(48,6),(54,12),(54,21),(52,18),(49,14),(46,11),(43,9),(36,6),(28,6),(28,60)],[(34,60),(42,58),(48,54),(51,50),(53,47),(56,40),(56,30),(34,30),(34,60)]],'i':[[(24,78),(25,77),(26,75),(26,70),(25,67),(24,66),(23,66),(19,64),(16,63),(9,63),(5,65),(3,67),(2,69),(2,74),(3,77),(5,79),(11,82),(17,82),(21,80),(24,78)],[(26,56),(26,0),(2,0),(2,56),(26,56)]],'m':[[(24,0),(24,60),(0,60),(0,0),(24,0)],[(54,0),(54,60),(30,60),(30,0),(54,0)],[(60,0),(84,0),(84,50),(83,53),(82,55),(79,58),(77,59),(74,60),(60,60),(60,0)]],'q':[[(32,60),(32,-24),(56,-24),(56,60),(32,60)],[(26,0),(23,0),(19,1),(13,3),(10,5),(5,10),(3,13),(1,19),(0,27),(0,35),(2,43),(3,46),(6,52),(9,55),(12,57),(16,59),(19,60),(26,60),(26,0)]],'u':[[(23,60),(0,60),(0,14),(1,10),(4,6),(12,2),(18,0),(23,0),(23,60)],[(29,60),(52,60),(52,14),(51,10),(48,6),(44,3),(39,1),(34,0),(29,0),(29,60)]],'y':[[(26,60),(44,18),(44,-20),(20,-20),(20,12),(0,60),(26,60)],[(32,60),(68,60),(48,22),(32,60)]],'}':[[(0,0),(0,20),(12,20),(12,64),(0,64),(0,84),(19,84),(27,80),(29,77),(32,71),(32,48),(34,44),(38,42),(40,42),(38,42),(34,40),(32,36),(32,13),(28,5),(25,3),(19,0),(0,0)]]}
    letterwidth = {' ':57,'$':77,'(':41,',':41,'0':85,'4':89,'8':77,'<':45,'@':85,'D':89,'H':85,'L':77,'P':85,'T':89,'X':85,'\\':77,'`':29,'d':73,'h':69,'l':41,'p':73,'t':57,'x':77,'|':33,'#':57,"'":29,'+':53,'/':77,'3':77,'7':65,';':41,'?':73,'C':81,'G':89,'K':89,'O':89,'S':77,'W':117,'[':49,'_':137,'c':73,'g':73,'k':77,'o':77,'s':69,'w':101,'{':57,'"':49,'&':97,'*':57,'.':41,'2':81,'6':85,':':41,'>':45,'B':81,'F':81,'J':77,'N':81,'R':89,'V':89,'Z':97,'^':57,'b':73,'f':57,'j':45,'n':69,'r':69,'v':77,'z':81,'~':97,'!':45,'%':129,')':41,'-':45,'1':57,'5':77,'9':85,'=':45,'A':89,'E':81,'I':49,'M':117,'Q':89,'U':81,'Y':89,']':49,'a':73,'e':73,'i':45,'m':101,'q':73,'u':69,'y':85,'}':57}
    # x height = 60, A height = 84, y descent = 20, vertical advance = 84+42
    def skewcurve(ps):
        return [(x+skew*y,y) for x,y in ps]
    def addpointtocurve(p,qs):
        return [tuple([pi+qi for pi,qi in zip(p,q)]) for q in qs] # return [tuple(v(p)+v(q)) for q in qs]
    curves = []
    if ss[0] in lettercurves:
        curves += [addpointtocurve(cursor,skewcurve(curve)) for curve in lettercurves[ss[0]]] # curves += [v(cursor)+v(skewcurve(curve)) for curve in lettercurves[ss[0]]]
        cursor = (cursor[0]+letterwidth[ss[0]],cursor[1])
    return curves+textcurves(ss[1:],False,skew,screencoordinates,cursor)
def curveboundingbox(curve):
    def depth(g):  # print depth(textcurves('abc')[0][0]),textcurves('abc')[0][0],depth(textcurves('abc')[0][0][0]),textcurves('abc')[0][0][0]
        import collections
        return 1 + max(depth(item) for item in g) if isinstance(g,collections.Iterable) else 0
    curve = [p for p in curve if p is not None]
    if not curve: return None
    if 3==depth(curve): # if curve is actually list of curves, flatten first
        return curveboundingbox([p for c in curve for p in c])
    xs,ys = list(zip(*curve)) # assert 2==depth(curve)
    x0,y0,x1,y1 = min(xs),min(ys),max(xs),max(ys)
    return (x0,y0,x1-x0,y1-y0)
def rectsboundingbox(rs):
    rs = [r for r in rs if r is not None]
    if not rs: return None
    # x0s,y0s,dxs,dys = list(zip(*rs))
    x0s,y0s,dxs,dys = zip(*rs)
    x1s,y1s = [x+dx for x,dx in zip(x0s,dxs)], [y+dy for y,dy in zip(y0s,dys)]
    x0,y0,x1,y1 = min(x0s),min(y0s),max(x1s),max(y1s)
    return (x0,y0,x1-x0,y1-y0)
#print rectsboundingbox([(0,1,6,6),(-2,-2,3,1)]) # (-2, -2, 8, 9)
def curvesboundingbox(cc):
    if not cc: return None
    return rectsboundingbox([curveboundingbox(c) for c in cc])
def iscurve(c):
    return isinstance(c,List) and isinstance(c[0],Tuple) and 2==len(c[0])
def iscurves(cc):
    return isinstance(cc,List) and isinstance(cc[0],List) and isinstance(cc[0][0],Tuple) and 2==len(cc[0][0])
def scaletext(cc,x=0,y=0,fitx=0,fity=0,margin=0,scale=0,center=True,dev=0,scalemag=1):
    return scalecurves(cc,x,y,fitx,fity,margin,scale,center,istext=True,dev=dev,scalemag=scalemag)
def scalecurves(cc,x=0,y=0,fitx=0,fity=0,margin=0,scale=0,center=True,istext=False,scalex=0,scaley=0,dev=0,scalemag=1):  # (F,T)[test==True] # {True:T,False:F}[test==True] # gy = (1,-1)[istext==True] # -1 if istext else +1
    x,y,fitx,fity = x+margin, y+(-1 if istext else +1)*margin, (fitx-2*margin if fitx else fitx), (fity-2*margin if fity else fity)
    if (fitx or fity) and cc:
        u0,v0,du,dv = curveboundingbox(cc)
        scalex,scaley = 1.*fitx/du,1.*fity/dv
        scale = min(scale,scalex,scaley) if scale else min(scalex,scaley)
        if 0==fitx or 0==fity: scale = max(scalex,scaley)
        if center and scalex>scale: x += fitx*(1-scale/scalex)/2
        if center and scaley>scale: y += fity*(1-scale/scaley)/2 * (-1 if istext else +1)
        return [[(x+scale*(u-u0)*(1 if not dev else scalemag),y+scale*(v-v0-dv*istext)) for u,v in c] for c in cc]#,scale
    if not scale: scale = 1
    return [[(x+scale*u*(1 if not dev else scalemag),y+scale*v) for u,v in c] for c in cc]#,scale
def rotatecurve(c,angle=0,x0=0,y0=0):
    def rotatepoint(x,y):
        return x0 + (x-x0)*cos(angle) - (y-y0)*sin(angle), y0 + (x-x0)*sin(angle) + (y-y0)*cos(angle)
    return [rotatepoint(x,y) for x,y in c]
def rotatecurves(cc,angle=0,x0=0,y0=0):
    return [rotatecurve(c,angle,x0,y0) for c in cc]
def turncurvesupsidedown(cc,frame=None): # rectangle defined by frame will occupy exact same position as before but things inside will now be rotated 180° 
    if not frame: frame = curvesboundingbox(cc)
    x0,y0,dx0,dy0 = frame
    return [[(2*x0+dx0-x,2*y0+dy0-y) for x,y in c] for c in cc]
def turnrectsupsidedown(rs,frame=None):
    if not frame: frame = rectsboundingbox(rs)
    x0,y0,dx0,dy0 = frame
    return [(2*x0+dx0-x-dx,2*y0+dy0-y-dy,dx,dy) for x,y,dx,dy in rs]
def rectlistarea(rs):
    return sum(dx*dy for x0,y0,dx,dy in rs)
def polyarea(c):
    return np.abs(signedpolyarea(c))
def signedpolyarea(c):
    xs,ys = zip(*c)
    xs,ys = np.array(xs),np.array(ys)
    return 0.5*(np.dot(xs,np.roll(ys,1))-np.dot(ys,np.roll(xs,1))) # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
def compresscurve(c,maxangle):  # maxangle in degrees # removes all points in a curve if the angle at that point exceeds maxangle
    c0 = c                      # for a better algorithm see https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
    for angle in np.linspace(180,maxangle,10): # remove largest angles first
        def exceedsangle(p0,p1,p2):
            a,b = np.array(p0)-np.array(p1),np.array(p2)-np.array(p1)
            cosθ = np.dot(a/np.linalg.norm(a),b/np.linalg.norm(b))
            return cosθ<cosφ
        n,cosφ = 1,np.cos(angle/180*np.pi)
        while len(c)-1<=n:
            if c[n-1]==c[n] or c[n]==c[n+1] or exceedsangle(c[n-1],c[n],c[n+1]):
                c = c[:n]+c[n+1:]
    if len(c)==len(c0): return c
    return compresscurve(c,maxangle)
def isinsideboundingbox(x,y,bb):
    return bb[0]<x<bb[0]+bb[2] and bb[1]<y<bb[1]+bb[3]
def clipped(x,y,bb):
    # return x if bb[0]<x<bb[0]+bb[2] else np.nan, y if bb[1]<y<bb[1]+bb[3] else np.nan
    return np.clip(x,bb[0],bb[0]+bb[2]), np.clip(y,bb[1],bb[1]+bb[3])
def deleteduplicatepoints(c):
    return [p for i,p in enumerate(c) if 0==i or not c[i]==c[i-1]]
def clipcurve(c,bb):
    c = [clipped(*p,bb) for p in c]
    return deleteduplicatepoints(c)
def upsamplecurve(c0,doublings=1):
    c1 = []
    for (x0,y0),(x1,y1) in zip(c0[:-1],c0[1:]):
        c1 += [[x0,y0],[x0/2+x1/2,y0/2+y1/2]]
    c1 += c0[-1:]
    if doublings>1: return upsamplecurve(c1,doublings-1)
    return c1
def cropped(c,bb):
    cs,ci = [],[]
    for p in c:
        if isinsideboundingbox(*p,bb):
            ci.append(p)
        else:
            if len(ci):
                cs.append(ci)
            ci = []
    if len(ci):
        cs.append(ci)    
    return cs
def cropcurves(cs,bb):
    return [ci for c in cs for ci in cropped(c,bb)]
def curvetowave(c):
    xs,ys = zip(*c)
    return Wave(ys,xs)
def plotcurvelist(cc,**kwargs):
    if cc:
        ws = [curvetowave(c) for c in cc]
        xmin = min([x for w in ws for x in w.x])
        xmax = max([x for w in ws for x in w.x])
        Wave.plots(*ws,xlim=(xmin,xmax),**kwargs)
def autocadcolor(n):
    autocadcolors = ((0,0,0,0),(1,255,0,0),(2,255,255,0),(3,0,255,0),(4,0,255,255),(5,0,0,255),(6,255,0,255),(7,255,255,255),(8,65,65,65),(9,128,128,128),(10,255,0,0),(11,255,170,170),(12,189,0,0),(13,189,126,126),(14,129,0,0),(15,129,86,86),(16,104,0,0),(17,104,69,69),(18,79,0,0),(19,79,53,53),(20,255,63,0),(21,255,191,170),(22,189,46,0),(23,189,141,126),(24,129,31,0),(25,129,96,86),(26,104,25,0),(27,104,78,69),(28,79,19,0),(29,79,59,53),(30,255,127,0),(31,255,212,170),(32,189,94,0),(33,189,157,126),(34,129,64,0),(35,129,107,86),(36,104,52,0),(37,104,86,69),(38,79,39,0),(39,79,66,53),(40,255,191,0),(41,255,234,170),(42,189,141,0),(43,189,173,126),(44,129,96,0),(45,129,118,86),(46,104,78,0),(47,104,95,69),(48,79,59,0),(49,79,73,53),(50,255,255,0),(51,255,255,170),(52,189,189,0),(53,189,189,126),(54,129,129,0),(55,129,129,86),(56,104,104,0),(57,104,104,69),(58,79,79,0),(59,79,79,53),(60,191,255,0),(61,234,255,170),(62,141,189,0),(63,173,189,126),(64,96,129,0),(65,118,129,86),(66,78,104,0),(67,95,104,69),(68,59,79,0),(69,73,79,53),(70,127,255,0),(71,212,255,170),(72,94,189,0),(73,157,189,126),(74,64,129,0),(75,107,129,86),(76,52,104,0),(77,86,104,69),(78,39,79,0),(79,66,79,53),(80,63,255,0),(81,191,255,170),(82,46,189,0),(83,141,189,126),(84,31,129,0),(85,96,129,86),(86,25,104,0),(87,78,104,69),(88,19,79,0),(89,59,79,53),(90,0,255,0),(91,170,255,170),(92,0,189,0),(93,126,189,126),(94,0,129,0),(95,86,129,86),(96,0,104,0),(97,69,104,69),(98,0,79,0),(99,53,79,53),(100,0,255,63),(101,170,255,191),(102,0,189,46),(103,126,189,141),(104,0,129,31),(105,86,129,96),(106,0,104,25),(107,69,104,78),(108,0,79,19),(109,53,79,59),(110,0,255,127),(111,170,255,212),(112,0,189,94),(113,126,189,157),(114,0,129,64),(115,86,129,107),(116,0,104,52),(117,69,104,86),(118,0,79,39),(119,53,79,66),(120,0,255,191),(121,170,255,234),(122,0,189,141),(123,126,189,173),(124,0,129,96),(125,86,129,118),(126,0,104,78),(127,69,104,95),(128,0,79,59),(129,53,79,73),(130,0,255,255),(131,170,255,255),(132,0,189,189),(133,126,189,189),(134,0,129,129),(135,86,129,129),(136,0,104,104),(137,69,104,104),(138,0,79,79),(139,53,79,79),(140,0,191,255),(141,170,234,255),(142,0,141,189),(143,126,173,189),(144,0,96,129),(145,86,118,129),(146,0,78,104),(147,69,95,104),(148,0,59,79),(149,53,73,79),(150,0,127,255),(151,170,212,255),(152,0,94,189),(153,126,157,189),(154,0,64,129),(155,86,107,129),(156,0,52,104),(157,69,86,104),(158,0,39,79),(159,53,66,79),(160,0,63,255),(161,170,191,255),(162,0,46,189),(163,126,141,189),(164,0,31,129),(165,86,96,129),(166,0,25,104),(167,69,78,104),(168,0,19,79),(169,53,59,79),(170,0,0,255),(171,170,170,255),(172,0,0,189),(173,126,126,189),(174,0,0,129),(175,86,86,129),(176,0,0,104),(177,69,69,104),(178,0,0,79),(179,53,53,79),(180,63,0,255),(181,191,170,255),(182,46,0,189),(183,141,126,189),(184,31,0,129),(185,96,86,129),(186,25,0,104),(187,78,69,104),(188,19,0,79),(189,59,53,79),(190,127,0,255),(191,212,170,255),(192,94,0,189),(193,157,126,189),(194,64,0,129),(195,107,86,129),(196,52,0,104),(197,86,69,104),(198,39,0,79),(199,66,53,79),(200,191,0,255),(201,234,170,255),(202,141,0,189),(203,173,126,189),(204,96,0,129),(205,118,86,129),(206,78,0,104),(207,95,69,104),(208,59,0,79),(209,73,53,79),(210,255,0,255),(211,255,170,255),(212,189,0,189),(213,189,126,189),(214,129,0,129),(215,129,86,129),(216,104,0,104),(217,104,69,104),(218,79,0,79),(219,79,53,79),(220,255,0,191),(221,255,170,234),(222,189,0,141),(223,189,126,173),(224,129,0,96),(225,129,86,118),(226,104,0,78),(227,104,69,95),(228,79,0,59),(229,79,53,73),(230,255,0,127),(231,255,170,212),(232,189,0,94),(233,189,126,157),(234,129,0,64),(235,129,86,107),(236,104,0,52),(237,104,69,86),(238,79,0,39),(239,79,53,66),(240,255,0,63),(241,255,170,191),(242,189,0,46),(243,189,126,141),(244,129,0,31),(245,129,86,96),(246,104,0,25),(247,104,69,78),(248,79,0,19),(249,79,53,59),(250,51,51,51),(251,80,80,80),(252,105,105,105),(253,130,130,130),(254,190,190,190),(255,255,255,255))
    i,r,g,b = autocadcolors[n]
    assert n==i
    return (r,g,b)
def curvelength(xs,ys):
    dxs,dys = np.diff(xs),np.diff(ys)
    return sum(np.sqrt(dxs**2+dys**2))
def sbend(dx,dy):
    k,a = np.pi/dx,dy/2
    rocinum = 1/k**2/a #print('bendx:',dx,'bendy:',dy,'roc(um):', 1/k**2/a,'roc(cm):', 1/k**2/a/10000,'bendx = sqrt( pi^2/2 * bendy * roc )',sqrt( pi**2/2 * bendy * roc))
    xs = np.linspace(0,dx,101)
    ys = a*(1-np.cos(k*xs))
    return xs,ys,rocinum,curvelength(xs,ys)
def splittapersbend(dx,dy,taperx,sx,sy):
    # taperx = length of the linear part
    # sx = length of second bend, if it was a full sbend
    # sy = displacement of second bend, if it was a full sbend 
    xx,yy = sx/2,sy/2 # length and displacement of second bend, exactly half of an sbend
    kk,aa = np.pi/2/xx,yy
    rr = 1/kk**2/aa
    xxs = np.linspace(-xx,0,101) + dx
    yys = dy + aa*(np.cos(kk*(xxs-dx))-1)
    slope = (yys[1]-yys[0])/(xxs[1]-xxs[0])
    # print((yys[1]-yys[0]),(xxs[1]-xxs[0]),slope)
    x0,y0 = xxs[0]-taperx,yys[0]-taperx*slope
    # print('x0,y0',x0,y0)
    # return [x0]+list(xxs),[y0]+list(yys)
    # need to solve these equations for a and k: f(x0) = a*(1-cos(k*x0)) = y0, f'(x0) = a*k*sin(k*x0) = slope (also want k*x0>pi/2?)
    ks = np.linspace(np.pi/2/dx,np.pi/y0,1001) # print(ks[:10])
    gs = 1 - np.cos(ks*x0) - (y0/slope)*ks*np.sin(ks*x0) # print(gs[:10])
    zs = np.diff(np.sign(gs)) # diff(sign(w)) will be non-zero at cross-overs and will have the sign of the crossing: # https://stackoverflow.com/a/25091643
    zerocrossings = np.where(np.logical_or(-2==zs,2==zs))[0] # eg. zs = [0 0 2 0 0 0 -2 0 0 0] # print('zerocrossings',zerocrossings)
    p = zerocrossings[0]
    kg = Wave(ks,gs)[p:p+2]
    k0 = kg(0) # get interpolated zero crossing
    a0 = y0/(1-np.cos(k0*x0))
    xs = np.linspace(0,x0,101)
    ys = a0*(1-np.cos(k0*xs))
    assert a0<dy and np.pi/k0<2*dx, str(dx)+' '+str(dy)+' '+str(np.pi/k0)+' '+str(a0)
    # xs,ys = list(xs)+[x0]+list(xxs), list(ys)+[y0]+list(yys)
    roc1,roc2 = 1/k0**2/a0, 1/kk**2/aa
    # print('  x0,y0',xs[-1],ys[-1])
    # print('  x1,y1',xxs[0],yys[0])
    print('  roc1,roc2',roc1,roc2)
    return xs,ys,xxs,yys,roc1,roc2,curvelength(list(xs)+list(xxs),list(ys)+list(yys))
def testsbend():
    xs,ys,roc,ds = sbend(1000,100)
    print(roc,ds)
    Wave(ys,xs).plot()
    x1s,y1s,x2s,y2s,roc1,roc2,ds = splittapersbend(11000,127,2500,3400,10)
    print('roc1,roc2,ds',roc1,roc2,ds)
    xs,ys = np.array(list(x1s)+list(x2s)),np.array(list(y1s)+list(y2s))
    Wave(ys,xs).plot()
    dydx = (ys[1:]-ys[:-1])/(xs[1:]-xs[:-1])
    Wave(dydx,xs[:-1]).plot()
def closest(a,b): # return index of closest pair of points in curves a and b
    def dd(p0,p1):
        (x0,y0),(x1,y1) = p0,p1
        return (x1-x0)**2 + (y1-y0)**2
    d0,m0,n0 = np.inf,None,None
    for m,p in enumerate(a):
        for n,q in enumerate(b):
            if dd(p,q)<d0:
                d0,m0,n0 = dd(p,q),m,n
    # print(m0,n0,a[m0],b[n0],dd(a[m0],b[n0]))
    return m0,n0
def acontainsb(a,b): # True is all points of b are inside and not touching a
    return all(matplotlib.path.Path(a).contains_points(b))
def eliminateholes(cc,showgap=False):
    # reduce outer and innerpolygons to single outer polygon
    # assumes inner and outer polygons have opposite signed area
    def acontainsb(an,bn):
        if an==bn: return False
        path = matplotlib.path.Path(cc[an])
        return all(path.contains_points(cc[bn]))
    # find an index that has no duplicates (it is first level nested)
    def findhole():
        holes = False
        for i in range(len(cc)):
            ns = [n for n in range(len(cc)) if acontainsb(n,i)]
            holes = holes or bool(len(ns))
            if 1==len(ns): # found single hole (inside polygon with only one outside polygon)
                return ns[0],i
        assert not holes, 'if no first level nested holes, there should be none at all'
    h = findhole()
    if h is None: # end recursion if no holes found
        return cc
    an,bn = h
    assert -1==np.sign(signedpolyarea(cc[an])*signedpolyarea(cc[bn])), 'inner and outer curves must be opposite cw/ccw to combine'
    def cutdonut(a,b): # asumes b is a hole inside a (creating donut), returns a replacement 'cut' donut 
        m,n = closest(a,b) # index of closest pair of points in curves a and b
        return list(a[:m+(0 if showgap else 1)]) + list(b[n:]) + list(b[:n+(0 if showgap else 1)]) + list(a[m:])
    cc[an] = cutdonut(cc[an],cc[bn])
    return eliminateholes([c for c in cc[:bn]+cc[bn+1:]],showgap)

class Path():
    def __init__(self,xys,width,normal=None,cw=False): # width=3 or width=[2,3,4] or width = lambda u:1+u
        self.width = width if callable(width) else (np.array(width if hasattr(width, '__len__') else [width]*len(xys)).reshape(-1,1))
        self.xys,self.normal,self.cw = np.array(xys),normal,cw
    def segmentdistances(self):
        d0,d1 = np.vstack(( self.xys[:1],self.xys[:-1] )),self.xys[:]
        def dist(v): return np.sqrt((v**2).sum(axis=-1,keepdims=True))
        return dist(d1-d0)
    def travelfraction(self):
        sd = self.segmentdistances()
        tf = np.cumsum(sd,axis=0) # print('sd.shape',sd.shape,'tf.shape',tf.shape)
        return tf/tf[-1]
    def rightperps(self): # right perpendiculars (unit length direction vectors)
        def dist(v): return np.sqrt((v**2).sum(axis=-1,keepdims=True))
        def norm(v): return v/dist(v)
        def extend(vs): return np.vstack((vs[:1],vs,vs[-1:])) # double the first and last points
        def rightperp(vs): return np.hstack((vs[:,1:],-vs[:,:1]))
        def leftperp(vs): return -rightperp(vs)
        segs = extend(norm(self.xys[1:]-self.xys[:-1])) # direction vector of each segment, segs[n] is segment before xys[n], segs[n+1] is dir of segment after xys[n]
        fperps = rightperp(segs[1:])  # fperps[n] = direction vector ⊥ to segment forward of xys[n]
        bperps = rightperp(segs[:-1]) # rperps[n] = direction vector ⊥ to segment backward of xys[n]
        return norm(fperps/2+bperps/2) # make length proportional to 1/cos(angle) to keep width constant?
    def leftperps(self):
        return -self.rightperps()
    def rightwidth(self):
        # print(self.width(self.travelfraction())/2 if callable(self.width) else self.width/2)
        return self.width(self.travelfraction())/2 if callable(self.width) else self.width/2
    def leftwidth(self):
        return -self.rightwidth()
    def spine(self):
        return self.xys.copy()
    def curve(self):
        if self.normal is None:
            r = self.xys + self.rightwidth() * self.rightperps()
            l = self.xys - self.rightwidth() * self.rightperps()
        else:
            def dist(v): return np.sqrt((v**2).sum(axis=-1,keepdims=True))
            def norm(v): return v/dist(v)
            def rightperp(vs): return np.hstack((vs[:,1:],-vs[:,:1]))
            fd = np.array(self.normal).reshape(1,-1)
            r = self.xys + self.rightwidth() * rightperp(norm(fd))
            l = self.xys - self.rightwidth() * rightperp(norm(fd))
        # print('c',c.shape,c[0].shape,c[0][0].shape)
        # print('xys',self.xys.shape,self.xys[0].shape,self.xys[0][0].shape)
        # print('rw',self.rightwidth().shape,self.rightwidth()[0].shape,self.rightwidth()[0][0].shape)
        return np.vstack((l,r[::-1],l[:1])) if self.cw else np.vstack((r,l[::-1 ],r[:1]))
    def listcurve(self):
        return [list(v) for v in self.curve()]
    def wave(self):
        c = self.curve()
        return Wave(c[:,1],c[:,0])
    def plot(self,**kwargs):
        self.wave().plot(aspect=True,**kwargs)
    def copy(self):
        return Path(self.xys.copy(),width=self.width,cw=self.cw)
    def __add__(self, p0):
        p = self.copy()
        p.xys = p.xys + p0
        return p
    def __str__(self):
        return f"width:{self.width} shape:{self.xys.shape} xys:{''.join([str(list(p)) for p in self.xys[:2]])}..{''.join([str(list(p)) for p in self.xys[-2:]])}"
class Arch(Path):
    def __init__(self,r,φ0,φ1,width,p0=[0,0],minangle=0.5,cw=False):
        def arc(r,φ0,φ1,minangle): # segments of arc will form a circumscribing regular polygon of the radius r circle
            def xy(r,φ): return np.array([r*cos(φ),r*sin(φ)])
            n = int(abs(φ1-φ0)/minangle)+1 # there will be n+1 triangle in circumscribing polygon, n-1 isosceles triangles plus two half-isosceles at start and end
            α = (φ1-φ0)/n/2.
            q = r/cos(α) #*sqrt(1+sin(α)**2) # radius of circle that circumscribes the polygon
            a = np.array([xy(r,φ0)] + [xy(q,m*α+φ0) for m in range(1,2*n,2)] + [xy(r,2*n*α+φ0)]) # print(n,np.array([0]+list(range(1,2*n,2))+[2*n]))
            return np.array(p0) + a - a[0]
        super().__init__(xys = arc(r,φ0,φ1,minangle),width=width,cw=cw)
        self.r,self.φ0,self.φ1 = r,φ0,φ1
    def copy(self):
        p = Path(self.xys.copy(),width=self.width,cw=self.cw)
        p.r,p.φ0,p.φ1 = self.r,self.φ0,self.φ1
        return p
def flattoparch(p0,L,r,spacings,minangle=0.1,upsidedown=1): # spacings = list of [width,gap,width,gap,..,width] from bottom to top, r and p0 are relative to middle element of spacings
    n,n0,p0,spacings = len(spacings),len(spacings)//2,V(p0),np.array(spacings)
    assert 1==n%2, "spacings can't be even"
    ys = np.cumsum(spacings)-spacings/2
    ys,ws = ys[::2] - ys[n0],spacings[::2]
    a = np.array([p0,p0+np.array([L,0])]) # xys of center flat
    flats = [Path(a,w)+V(0,y) for y,w in zip(ys,ws)]
    def arcs(left=True):
        if upsidedown:
            v0,φ0,φ1 = (V(0,0),pi*3/2,pi) if left else (V(L,0),-pi/2,0)
        else:
            v0,φ0,φ1 = (V(0,0),pi/2,pi) if left else (V(L,0),pi/2,0)
        return [Arch(r+(-y if upsidedown else y),φ0,φ1,w,p0=p0+V(0,y)+v0,minangle=minangle) for y,w in zip(ys,ws)]
    leftarcs = arcs(left=1)
    rightarcs = arcs(left=0)
    return flats+leftarcs+rightarcs
def ribbonarcs(r,spacings,φ0=pi/2,φ1=0,p0=(0,0),minangle=0.5,cw=False):
    n,n0,p0,spacings = len(spacings),len(spacings)//2,V(p0),np.array(spacings)
    assert 1==n%2, "spacings can't be even"
    def offsets(spacings):
        ys = np.cumsum(spacings)-spacings/2
        return ys[::2] - ys[n0]
    y0s,w0s = offsets(spacings),spacings[::2]
    p0s = [p0 + V(cos(φ0),sin(φ0))*y0 for y0 in y0s]
    print(y0s,w0s,p0s)
    xys = [Arch(r+y,φ0=φ0,φ1=φ1,width=w,p0=p,minangle=minangle,cw=cw) for y,w,p in zip(y0s,w0s,p0s)]
    return xys
def ribbons(L,spacings,endspacings=None,p0=(0,0),horizontal=True): # use endspacings to make linearly tapered ribbons 
    n,n0,p0,spacings = len(spacings),len(spacings)//2,V(p0),np.array(spacings)
    endspacings = np.array(endspacings) if endspacings is not None else spacings
    assert 1==n%2, "spacings can't be even"
    def offsets(spacings):
        # print('spacings',spacings)
        ys = np.cumsum(spacings)-spacings/2
        return ys[::2] - ys[n0]
    y0s,y1s = offsets(spacings),offsets(endspacings)
    w0s,w1s = spacings[::2],endspacings[::2]
    a = np.array([p0,p0+(V(L,0) if horizontal else V(0,L))]) # xys of center ribbon
    if not horizontal:
        return [Path(a+np.array([[x0,0],[x1,0]]),[w0,w1],normal=(0,1)) for x0,x1,w0,w1 in zip(y0s,y1s,w0s,w1s)]
    else:
        return [Path(a+np.array([[0,y0],[0,y1]]),[w0,w1],normal=(1,0)) for y0,y1,w0,w1 in zip(y0s,y1s,w0s,w1s)]
def taperedribbons(xys,widthfuncs,p0=(0,0),gaptest=False):
    n,n0,p0 = len(widthfuncs),len(widthfuncs)//2,np.array(p0)
    assert 1==n%2, "widthfuncs can't be even"
    def xoffset(i,x):
        def xi(i,x):
            # print(i,widthfuncs[:i])
            fs = [f for f in widthfuncs[:i]]
            # print(sum([f(x) for f in widthfuncs[:i]]) , widthfuncs[i](x)/2)
            return sum([f(x) for f in widthfuncs[:i+1]]) - widthfuncs[i](x)/2
        return xi(i,x) - xi(n0,x)
    def segmentdistances(xys):
        d0,d1 = np.vstack(( xys[:1],xys[:-1] )),xys[:]
        def dist(v): return np.sqrt((v**2).sum(axis=-1,keepdims=True))
        return dist(d1-d0)
    def travelfraction(xys):
        sd = segmentdistances(np.array(xys))
        tf = np.cumsum(sd,axis=0)
        return tf/tf[-1]
    def xysi(i):
        tf = travelfraction(xys) # print('tf',tf)
        xs = [xoffset(i,t) for t in tf.flatten()] # print('xs',xs)
        return np.array(xys)+np.array([[x,0] for x in xs])
    def widthi(i):
        tf = travelfraction(xys)
        return [widthfuncs[i](t) for t in tf]
    # print('xysi',xysi(0).shape,xysi(0)[0].shape,xysi(0)[0][0].shape)
    ribbons = [Path(xysi(i),widthi(i),normal=(0,1)) for i in range(1 if gaptest else 0,n,2)]
    return ribbons
def mzsribbons2(L,r,hot,gap,fhot,fgap,yin,yout=None,p0=(0,0),minangle=0.1):
    yout = yin if yout is None else yout
    r = yin if r is None else r
    a0,b0,c0,c1,b1,a1 = V(-r,yin),V(-r,r),V(0,0),V(L,0),V(L+r,-r),V(L+r,-yout)
    ### linear scaling
    # ai = Vs.arc(r,pi,1.5*pi,minangle=0.01).scale([1,yin/r]) + a0
    # a = Vs([c0,c1])
    # ao = Vs.arc(r,0.5*pi,0,minangle=0.01).scale([1,yout/r]) + c1
    ### quad scaling
    def fo(y): # y:0→r,fo:0→yin
        return -r*( abs(y/r) + (yin/r-1)*(y/r)**2 )
    def fi(y):
        return -yin+r*( abs(1+y/r) + (yin/r-1)*(1+y/r)**2 )
    ai = Vs.arc(r,pi,1.5*pi,minangle=0.01).map(lambda x:x,fi) + a0
    a = Vs([c0,c1])
    ao = Vs.arc(r,0.5*pi,0,minangle=0.01).map(lambda x:x,fo) + c1
    # [ 12.09385244  13.88770351  15.58834625  17.21270263  43.44403209  71.08188564  92.43532345 108.01408927 118.92169182 126.2266635 130.65358594 132.88075357] x:[  6   8  10  12  50 100 150 200 250 300 350 400]
    # gapvswidth50Ω = Wave([12.09385244,13.88770351,15.58834625,17.21270263,43.44403209,71.08188564,92.43532345,108.01408927,118.92169182,126.2266635,130.65358594,132.88075357],[6,8,10,12,50,100,150,200,250,300,350,400])
    # gapvswidth50Ω = Wave([12.415,14.165,15.934,17.549,19.859,23.671,27.247,30.830,34.345,37.808,41.273,44.676,48.005,51.272,54.590,57.830,60.991,64.223,67.369,70.451,73.588,76.626,88.658,100.197,111.269,121.853,131.988,141.575,150.737,159.382,167.549,175.209,182.487,189.242,195.638,201.565,207.112,212.379,217.209,221.714,225.874,229.709],[6,8,10,12,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500])
    # gapvswidth50Ω = Wave([10.073,11.808,13.560,15.229,17.521,21.278,24.882,28.328,31.685,35.061,38.267,41.446,44.605,47.563,50.591,53.474,56.320,59.153,61.790,64.518,67.028,69.622,79.054,87.405,94.919,101.434,107.168,112.145,116.443,120.157,123.269,125.905,128.238,130.040,131.011,131.692,132.061],[6,8,10,12,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400]) # ww = Wave([elvsg(hot,buffer=0.1,ohms=50,quick=1) for hot in hots],hots,"50Ω")
    # gapvswidth50Ω1000grid100nmbuffer
    gapvswidth50Ω = Wave([10.205,12.012,13.688,15.363,17.697,21.501,25.018,28.597,32.063,35.481,38.935,42.302,45.686,48.910,52.226,55.439,58.631,61.865,64.993,68.112,71.254,74.362,86.318,97.873,109.048,119.694,129.960,139.550,148.767,157.490,165.681,173.445,180.778,187.578,193.959,199.987,205.609],[6,8,10,12,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400])
    widthvsgap50Ω = gapvswidth50Ω.swapxy()
    def easefunc(x):
        # return np.sin(x*np.pi/2)**2
        return 1-(1-np.sin(x*np.pi/2)**2)**2 # wider at fanout
    def hotwidth(f):
        # return hot + (fhot-hot)*easefunc(f)
        # print('gap',gapwidth(f),'hot',widthvsgap50Ω(gapwidth(f),extrapolate='log'))
        # return widthvsgap50Ω(gapwidth(f),extrapolate='log')
        def lerp(x,x0,x1,y0,y1):
            return y0 + (y1-y0)*(x-x0)/(x1-x0)
        h0,h,h1 = [widthvsgap50Ω(g,extrapolate='log') for g in [gap,gapwidth(f),fgap]]
        return lerp(h,h0,h1,hot,fhot)
    def gapwidth(f):
        return gap + (fgap-gap)*easefunc(f)
    vi = ai[::-1].thickribbons([hotwidth,gapwidth,hotwidth])
    v = a.thickribbons([hot,gap,hot])
    vo = ao.thickribbons([hotwidth,gapwidth,hotwidth])
    vss = vi + v + vo
    # Vs.list2chain(vss).plot(aspect=1,m=' ',xlim=(-1000,500))
    # Vs.list2chain(vss).plot(aspect=1,m=' ',xlim=(45000,47000))
    # exit()
    return [vs + p0 for vs in vss]
def mzsribbonsfixed(L,r,hot,gap,yin,yout=None,p0=(0,0),minangle=0.1):
    yout = yin if yout is None else yout
    r = yin if r is None else r
    a0,b0,c0,c1,b1,a1 = V(-r,yin),V(-r,r),V(0,0),V(L,0),V(L+r,-r),V(L+r,-yout)
    ai = Vs([a0,b0]).upsample(400)[:-1] ++ (Vs.arc(r,pi,1.5*pi,width=None,p0=(0,0),minangle=0.01) + b0)
    a = Vs([c0,c1])
    ao = (Vs.arc(r,0.5*pi,0,width=None,p0=(0,0),minangle=0.01) + c1) ++ Vs([b1,a1]).upsample(400)[1:]
    ws = [hot,gap,hot]
    vi = ai[::-1].thickribbons(ws)
    v = a.thickribbons(ws)
    vo = ao.thickribbons(ws)
    vss = vi + v + vo
    return [vs + p0 for vs in vss]

def mzsribbons(L,r,win,we,wout,yin,yout=None,p0=(0,0),minangle=0.1):
    yout = yin if yout is None else yout
    cs = ( ribbons(yin-r,we,win,p0+V(-r,r),horizontal=0)
         + ribbonarcs(r,we,pi,3/2*pi,p0+V(-r,r),minangle)
         + ribbons(L,we,we,p0)
         + ribbonarcs(r,we,pi/2,0,p0+V(L,0),minangle)
         + ribbons(yout-r,wout,we,p0+V(L+r,-yout),horizontal=0)  )
    return cs
def mzrftest(L,r,taperx,win,we,wout,yin,yout=None,p0=(0,0),minangle=0.1,onlystraight=False):
    yout = yin if yout is None else yout
    cs = ( (ribbons(yin-r,win,win,V(p0)+V(-r,r),horizontal=0) if r else [])
         + (ribbonarcs(r,win,pi,3/2*pi,V(p0)+V(-r,r),minangle) if r else [])
         + (ribbons(taperx,win,we,V(p0)) if not onlystraight else [])
         + ribbons(L,we,we,V(p0)+V(taperx,0))
         + (ribbons(taperx,we,wout,V(p0)+V(L+taperx,0)) if not onlystraight else [])
         + (ribbonarcs(r,wout,pi/2,0,V(p0)+V(L+2*taperx,0),minangle) if r else [])
         + (ribbons(yout-r,wout,wout,V(p0)+V(L+2*taperx+r,-yout),horizontal=0) if r else [])  )
    return cs
def mzribbons(L=5000,p0=(0,0),ty=1000,r=200,w0=[100,47,8,47,100],w1=[300,125,30,125,300],ftaper=None,upsidedown=False,tnum=101,minangle=0.01):
    ftaper = ftaper if ftaper is not None else lambda v:v**2
    cs = flattoparch(p0,L,r,w0,minangle=minangle,upsidedown=upsidedown)
    fs = [lambda u,wi=wi,wj=wj:wi+ftaper(u)*(wj-wi) for wi,wj in zip(w0,w1)]
    xys = np.array([[0,y] for y in np.linspace(0,ty if upsidedown else -ty,tnum)])+p0
    cs += taperedribbons(xys + ((-r,r) if upsidedown else (-r,-r)),fs)
    cs += taperedribbons(xys + ((L+r,+r) if upsidedown else (L+r,-r)),fs)
    return cs

def intersect(p1,q1,p2,q2): # https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    def onSegment(p, q, r):
        (px,py),(qx,qy),(rx,ry) = p,q,r
        if ( (qx <= max(px, rx)) and (qx >= min(px, rx)) and 
               (qy <= max(py, ry)) and (qy >= min(py, ry))): 
            return True
        return False
    def orientation(p, q, r):
        (px,py),(qx,qy),(rx,ry) = p,q,r
        val = (float(qy - py) * (rx - qx)) - (float(qx - px) * (ry - qy)) 
        return 1 if val>0 else (2 if val<0 else 0)
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
    if ((o1 != o2) and (o3 != o4)):  # General case 
        return True
    if ((o1 == 0) and onSegment(p1, p2, q1)): # p1 , q1 and p2 are collinear and p2 lies on segment p1q1 
        return True
    if ((o2 == 0) and onSegment(p1, q2, q1)): # p1 , q1 and q2 are collinear and q2 lies on segment p1q1 
        return True
    if ((o3 == 0) and onSegment(p2, p1, q2)): # p2 , q2 and p1 are collinear and p1 lies on segment p2q2 
        return True
    if ((o4 == 0) and onSegment(p2, q1, q2)): # p2 , q2 and q1 are collinear and q1 lies on segment p2q2 
        return True
    return False
def collinear(p0,p1,q0,q1):
    (p0x,p0y),(p1x,p1y),(q0x,q0y),(q1x,q1y) = p0,p1,q0,q1
    px,py,qx,qy = p1x-p0x,p1y-p0y,q1x-q0x,q1y-q0y
    return np.abs(px*qy - qx*py) < 1e-10
def crossing(p0,p1,q0,q1):
    if intersect(p0,p1,q0,q1):
        return p0!=q0 and p0!=q1 and p1!=q0 and p1!=q1
    return False
def issimplepolygon(c):
    segs = [(p0,p1) for p0,p1 in zip(c[:-1],c[1:])]
    for n,seg0 in enumerate(segs):
        for seg1 in segs[n+1:]:
            (p0,p1),(q0,q1) = seg0,seg1
            if crossing(p0,p1,q0,q1):
                # from waves import Vs
                # Vs.plots(Vs([p0,p1]),Vs([q0,q1]))
                # print('collinear(p0,p1,q0,q1)',collinear(p0,p1,q0,q1))
                return False
    return True
def rectanglepackertest(chiplengthlist,rotation=True):
    from rectpack import newPacker
    rectangles = [(100, 30), (40, 6000), (30, 30),(70, 70), (1000, 50), (30, 30)][::-1]*20
    rectangles = [(x+2000,2000) for x in chiplengthlist*3]
    bins = [(30000,30000),(30000,30000),]
    # bins = [(300, 450), (1500, 7000), (200, 150)]
    packer = newPacker(rotation=rotation)
    # Add the rectangles to packing queue
    for r in rectangles:
        packer.add_rect(*r)
    # Add the bins where the rectangles will be placed
    for b in bins:
        packer.add_bin(*b)
    # Start packing
    packer.pack()
    return packer

if __name__ == '__main__':
    def validpolytest():
        c = [(0,0),(1,0),(1,1),(0,1),(0,0)] # square
        print(issimplepolygon(c),True)
        c = [(0,0),(1,1),(1,0),(0,1),(0,0)] # bowtie
        print(issimplepolygon(c),False)
        c = [(0,0),(1,0),(1,1),(0,0.5),(0,1),(0,0)] # touching V
        print(issimplepolygon(c),False)
    def intersecttest(f):
        print([
            f((0,1), (0,2), (2,10), (1,9)),  # non intersect
            f((0,1), (0,2), (1,10), (1,9)),  # non intersect parallel  lines
            f((0,1), (0,2), (1,10), (2,10)), # non intersect vertical and horizontal lines
            f((0,1), (1,2), (0,10), (1,9)),  # non intersect
            f((0,1), (0,2), (1,1), (1,3)),  # non intersect
            f((0,1), (0,2), (1,3), (1,1)),  # non intersect
            f((0,1), (0,3), (0,2), (0,4)),  # overlapping parallel
            f((0,1), (0,3), (0,3), (1,4)),  # touching  lines
            f((1,1), (3,1), (2.5,2), (2.5,0)),  # +
            f((7,7), (-6,-6), (-9,9), (6,-6)),  # x at origin
            f((7,7), (-6,-8), (-7,9), (9,-6)),  # x
            f((0,0), (0,3), (-1,3), (1,3)),  # T
            ])
        print([bool(i) for i in [0,0,0,0,0,0,1,1,1,1,1,1]])
    def pathtest():
        xys = np.array([[0,0],[1,1],[2,1],[2,2]])
        ws = [.2,.3,.4,.2]
        p,pp = Path(xys,0.2),Path(xys,2*np.array(ws),cw=1)
        Wave.plots(p.wave(),pp.wave(),x='x',y='y',aspect=True,pause=pause)
        q = Arch(5,pi/2,pi,1) # print(q) # w,ww = Wave(q.xys[:,1],q.xys[:,0]),Wave(q.xys[:2,1],q.xys[:2,0])
        q.plot(m=1,pause=pause)
        class Path2(Path):
            def __init__(self): super().__init__(xys=[[0,0],[1,1],[1,0]],width=0.1)
        Path2().plot(m=1,pause=pause)
    def factettest():
        xys = np.array([[0,0],[1,1],[2,1],[2,2]])
        ws = [.2,.3,.4,.2]
        p = Path(xys,ws,normal=(10,10))
        Wave.plots(p.wave(),m=1,x='x',y='y',aspect=True,pause=pause)
    def ribbontest():
        wws = [p.wave() for p in ribbons(-1000,[300,125,30,125,300])]
        ws = [p.wave() for p in ribbons(1000,[1000,60,90,60,1000],[1000,250,270,250,1000])]
        Wave.plots(*ws,*wws,m=0,seed=2,markersize=2,aspect=1,pause=pause)
        wws = [p.wave() for p in ribbons(-1000,[300,125,30,125,300],horizontal=0)]
        ws = [p.wave() for p in ribbons(1000,[1000,60,90,60,1000],[1000,250,270,250,1000],horizontal=0)]
        Wave.plots(*ws,*wws,m=0,seed=2,markersize=2,aspect=1,pause=pause)
    def arctest():
        ws = [p.wave() for p in [Arch(5,0,pi*3/2,1,V(-10,-10)),Arch(5,pi/2,pi,1,V(0,-10)),Arch(5,0,pi/2,1,V(10,-10))]] # ccw arcs, φ0<φ1
        wws = [p.wave() for p in [Arch(5,0,-pi*3/2,2,V(-10,10)),Arch(5,-pi/2,-pi,2,V(0,10)),Arch(5,0,-pi/2,2,V(10,10))]] # cw arcs, φ1<φ0
        p = Arch(5,0,2*pi,1,[20,0]) # full circle
        assert np.allclose(p.xys[0],p.xys[-1]),str(list(p.xys[0]))+str(list(p.xys[-1]))
        ww = [p.wave() for p in flattoparch([-15,30],30,20,[3,1,1,1,2])]
        Wave.plots(*ws,*wws,p.wave(),*ww,m=1,seed=3,markersize=2,aspect=1,pause=pause)
    def ribbonarctest():
        ps = ribbonarcs(10,[1,2,3,2,1]) + ribbonarcs(10,[1,2,3,2,1],φ0=pi/2,φ1=2*pi)
        ws = [p.wave() for p in ps]
        Wave.plots(*ws,m=1,seed=2,markersize=2,aspect=1,pause=pause)
    def pathfunctiontest(r=5,w=9):
        θ = np.linspace(0,pi,11).reshape(-1,1)
        xys = np.hstack((r*cos(θ),r*sin(θ)))
        xys = np.vstack(( xys[:1]+V(0,-r),xys,xys[-1:]+V(0,-r) ))
        p = Path(xys,width=w)
        pp = Path(xys,width=(lambda u:1+(2*w-2)*abs(u-0.5)))
        Wave.plots(p.wave(),pp.wave(),m=1,x='x',y='y',aspect=True,pause=pause)
    def taperedribbontest():
        ws = taperedribbons([[0,i] for i in np.linspace(0,8,17)],[lambda u:u**2+3, lambda u:u**2+2, lambda u:u**2+1, lambda u:u**2+2, lambda u:u**2+4])
        wws = taperedribbons([[0,i] for i in np.linspace(0,8,3)],[lambda u:u**2+3, lambda u:u**2+2, lambda u:u**2+1, lambda u:u**2+2, lambda u:u**2+4])
        wwws = taperedribbons([[0,-i] for i in np.linspace(0,8,17)],[lambda u:u**2+3, lambda u:u**2+2, lambda u:u**2+1, lambda u:u**2+2, lambda u:u**2+4],gaptest=1)
        Wave.plots(*[w.wave() for w in ws+wws+wwws],m=1,seed=2,markersize=2,aspect=1,pause=pause)
    def mzribbonstest():
        # Wave.plots(*[w.wave() for w in mzribbons(ftaper=lambda v:v)],m=0,seed=2,markersize=2,aspect=1,pause=pause)
        Wave.plots(*[w.wave() for w in mzribbons(ftaper=None)],m=0,seed=2,markersize=2,aspect=1,pause=pause)
        Wave.plots(*[w.wave() for w in mzrftest(L=5000,r=500,taperx=1000,win=[300,125,30,125,300],we=[100,47,8,47,100],wout=[300/2,125,30,125,300/2],yin=1000)],m=0,seed=2,markersize=2,aspect=1,pause=pause)
        Wave.plots(*[w.wave() for w in mzsribbons(L=5000,r=200,win=[300,125,30,125,300],we=[100,47,8,47,100],wout=[300/2,125,30,125,300/2],yin=1000,yout=2000)],m=1,seed=2,markersize=2,aspect=1,pause=pause)
    def mztapertest():
        L,p0,ty,r,upsidedown,ftaper = 5000,(0,0),1000,200,False,lambda v:v**2
        # chip is 100-47-8-47-100 to 300-125-30-125-300
        w0,w1 = [100,47,8,47,100],[300,125,30,125,300]
        cs = flattoparch(p0,L,r,w0,minangle=0.01,upsidedown=upsidedown)
        fs = [lambda u,wi=wi,wj=wj:wi+ftaper(u)*(wj-wi) for wi,wj in zip(w0,w1)]
        ps = np.array([[0,y] for y in np.linspace(0,ty if upsidedown else -ty,101)])+p0
        cs += taperedribbons(ps + ((-r,r) if upsidedown else (-r,-r)),fs)
        cs += taperedribbons(ps + ((L+r,+r) if upsidedown else (L+r,-r)),fs)
        # interconnect is 500-60-90-60-500 to 1000-250-260-250-1000
        ww0,ww1 = [500,60,90,60,500],[1000,250,260,250,1000]
        ffs = [lambda u,wi=wi,wj=wj:wi+(1-ftaper(1-u))*(wj-wi) for wi,wj in zip(ww0,ww1)]
        pps = np.array([[0,y] for y in np.linspace(0,2*ty if upsidedown else -2*ty,101)])+p0
        cs += taperedribbons(pps + ((-r,r+ty) if upsidedown else (-r,-r-ty)),ffs)
        cs += taperedribbons(pps + ((L+r,+r+ty) if upsidedown else (L+r,-r-ty)),ffs)
        Wave.plots(*[w.wave() for w in cs],x='x (µm)',y='y (µm)',xlim=(-2000,3000),m=0,seed=0,markersize=2,aspect=1,pause=pause)
    pause = 1
    pathtest()
    factettest()
    arctest()
    ribbonarctest()
    ribbontest()
    pathfunctiontest()
    taperedribbontest()
    mzribbonstest()
    mztapertest()
    intersecttest(intersect)
    validpolytest()
    savedxf()
    testsbend()

