#!/usr/bin/env python
#  -*- coding: utf-8 -*-
from freetype import *
import numpy as np
from geometry import eliminateholes

class Font(object):
    import freetype
    def __init__(self, filename=r'C:\Windows\Fonts\arial.ttf', size=128, screencoordinates=False):
        self.face,self.size,self.screencoordinates = freetype.Face(filename),size,screencoordinates
        self.face.set_char_size(size) # FT_Set_Char_Size https://www.freetype.org/freetype2/docs/tutorial/step1.html
    def holelesscharcurves(self,c,holeless=True,showgap=False):
        # TODO: intersects_path(self, other, filled=True) from matplotlib.path to check for curves enclosing curves, then reduce to single curve
        # print(c)
        return eliminateholes(self.charcurves(c),showgap=showgap) if holeless else self.charcurves(c) 
    def textcurves(self,ss,verticallayout=False,verticalinvert=None,holeless=True,cursor=(0,0)):
        import sys
        assert len(ss)<sys.getrecursionlimit(), f'text too long:{str(len(ss))} "{ss[:40]}..."' # import sys; sys.setrecursionlimit(3000)
        verticalinvert = self.screencoordinates if verticalinvert is None else verticalinvert
        if verticallayout:
            return [[(-y,x) for x,y in c] for c in self.textcurves(ss,False,verticalinvert,holeless,cursor)]
        if verticalinvert:
            cc = self.textcurves(ss,verticallayout,False,holeless,cursor)
            return [[(x,-y) for x,y in c] for c in cc]
        if not ss: return []
        def addpoints(p,q):
            return tuple([pi+qi for pi,qi in zip(p,q)])
        def addpointtocurve(p,qs):
            return [tuple([pi+qi for pi,qi in zip(p,q)]) for q in qs] # return [tuple(v(p)+v(q)) for q in qs]
        h,v = self.horizontaladvance(ss[0]),self.verticaladvance(ss[0])
        if '\n'==ss[0]:
            return self.textcurves(ss[1:],verticallayout,verticalinvert,holeless,addpoints((0,cursor[1]), (0,-v)))
        return [addpointtocurve(cursor,c) for c in self.holelesscharcurves(ss[0],holeless=holeless)] + self.textcurves(ss[1:],verticallayout,verticalinvert,holeless,addpoints(cursor, (h,0)))
    def charcurves(self,c):
        # print('     *',c,ord(c))
        g = self.glyphpath(c)
        return g.to_polygons() if g else []
    def plot(self,c,**kwargs):
        return plotcurvelist(self.holelesscharcurves(c,holeless=kwargs.pop('holeless',True)),**kwargs)
    def charbounds(self,c):
        cc = self.charcurves(c)
        xs = [pj[0] for c in cc for pj in c]
        ys = [pj[1] for c in cc for pj in c]
        return (min(xs),min(ys),max(xs),max(ys))
    def xheight(self):
        return self.charbounds('x')[3]
    def ydescent(self):
        return self.charbounds('y')[1]
    def Aheight(self):
        return self.charbounds('A')[3]
    def horizontaladvance(self,c):
        self.face.load_char(c)
        # print(c,self.face.glyph.metrics.horiAdvance,self.face.glyph.advance.x)
        # return self.face.glyph.metrics.horiAdvance
        # assert 2048==self.face.units_per_EM
        # print(c,self.face.units_per_EM)
        return self.face.glyph.linearHoriAdvance/64*self.size/2048
    def verticaladvance(self,c):
        self.face.load_char(c)
        return self.face.glyph.metrics.vertAdvance
    def charwidth(self,c):
        self.face.load_char(c)
        return self.face.glyph.advance.x # same as self.face.glyph.metrics.horiAdvance
    def has_kerning(self):
        return self.face.has_kerning
    def actually_has_kerning(self):
        # for m in range(ord('A'),ord('z')+1):
        #     print(chr(m),end='')
        #     for n in range(ord('A'),ord('z')+1):
        #         print(Font().kerning_offset(chr(m),chr(n)),end='')
        #     print()
        return any([Font().kerning_offset(chr(m),chr(n)) for m in range(ord('A'),ord('z')+1) for n in range(ord('A'),ord('z')+1)])
    def kerning_offset(self, previous_char, char):
        # https://dbader.org/blog/monochrome-font-rendering-with-freetype-and-python
        # print(previous_char, self.face.get_char_index(previous_char), char, self.face.get_char_index(char))
        kerning = self.face.get_kerning(self.face.get_char_index(previous_char), self.face.get_char_index(char))
        return kerning.x # / 64
    def boundstest(self):
        print('x',self.charbounds('x'))
        print('A',self.charbounds('A'))
        print('y',self.charbounds('y'))
        print(self.xheight(),self.ydescent(),self.Aheight())
    def glyphmetrics(self,char='x'):
        self.face.load_char(char)
        m = self.face.glyph.metrics
        print('self.face.glyph.advance',self.face.glyph.advance.x,self.face.glyph.advance.y)
        print('self.face.glyph.linearHoriAdvance',self.face.glyph.linearHoriAdvance/64*128/2048)
        print('self.face.get_advance()',self.face.get_advance(self.face.get_char_index(char),freetype.FT_LOAD_RENDER)/64)
        print(char,(self.face.glyph.advance.x,self.charwidth(char)),(self.verticaladvance(char),),m.horiAdvance,m.vertAdvance,m.width,m.height)
    def fontmetrics(self):
        f = self.face
        print(f.descender,f.underline_position,f.ascender,f.max_advance_width,f.height,(f.bbox.xMin,f.bbox.yMin),(f.bbox.xMax,f.bbox.yMax),'(prescaled)')
        # self.glyphmetrics('A')
        # self.glyphmetrics('m')
        # self.glyphmetrics('W')
        # self.glyphmetrics('y')
        # self.glyphmetrics('x')
        print(self.charwidth('x'),[len(c) for c in self.charcurves('x')],[p for p in self.charcurves('x')[0][:4]])
    def plottest(self):
        import matplotlib.pyplot as plt
        figure = plt.figure(figsize=(8,10))
        axis = figure.add_subplot(111)
        cc = self.charcurves('g')
        print([len(c) for c in cc])
        for c in cc:
            axis.scatter([pj[0] for pj in c], [pj[1] for pj in c])
        plt.show()
    def holelesscurvetest(self,text):
        print(-1,signedpolyarea([(0,0),(1,0),(1,1),(0,1),(0,0)])) # -1==ccw traditional coordinates==cw screen coordinates
        print(+1,signedpolyarea([(0,0),(0,1),(1,1),(1,0),(0,0)])) # +1==cw traditional coordinates==ccw screen coordinates
        print([signedpolyarea(c) for c in self.charcurves('%')])
        # for s in 'BAR8akgoq0@#&': plotcurvelist(eliminateholes(self.charcurves(s),0))
        for s in text: plotcurvelist(eliminateholes(self.charcurves(s),0),pause=0)
        # plotcurvelist(eliminateholes(self.charcurves('%'),0)) # fails
    def glyphpath(self,char,simplify=False):
        from matplotlib.path import Path
        self.face.load_char(char)
        outline = self.face.glyph.outline
        points = np.array(outline.points, dtype=[('x',float), ('y',float)])
        start, end = 0, 0
        VERTS, CODES = [], []
        # Iterate over each contour
        for i in range(len(outline.contours)):
            end    = outline.contours[i]
            points = outline.points[start:end+1]
            points.append(points[0])
            tags   = outline.tags[start:end+1]
            tags.append(tags[0])
            segments = [ [points[0],], ]
            for j in range(1, len(points) ):
                segments[-1].append(points[j])
                if tags[j] & (1 << 0) and j < (len(points)-1):
                    segments.append( [points[j],] )
            verts = [points[0], ]
            codes = [Path.MOVETO,]
            for segment in segments:
                if len(segment) == 2:
                    verts.extend(segment[1:])
                    codes.extend([Path.LINETO])
                elif len(segment) == 3:
                    verts.extend(segment[1:])
                    codes.extend([Path.CURVE3, Path.CURVE3])
                else:
                    verts.append(segment[1])
                    codes.append(Path.CURVE3)
                    for i in range(1,len(segment)-2):
                        A,B = segment[i], segment[i+1]
                        C = ((A[0]+B[0])/2.0, (A[1]+B[1])/2.0)
                        verts.extend([ C, B ])
                        codes.extend([ Path.CURVE3, Path.CURVE3])
                    verts.append(segment[-1])
                    codes.append(Path.CURVE3)
            VERTS.extend(verts)
            CODES.extend(codes)
            start = end+1
        return Path(VERTS, CODES).cleaned(simplify=simplify) if VERTS else []
    def allchars(self,removeblank=True):
        a = [chr(c) for c,i in self.face.get_chars()]
        # for ai in a: print(ai,len(self.charcurves(ai)))
        if removeblank:
            a = [ai for ai in a if len(self.charcurves(ai))]
        return ''.join(a)
    def plotall(self,holeless=True):
        text = self.allchars()
        rows = int(sqrt(len(text)/1.6))
        rowlen = [len(text)//rows + (i<len(text)%rows) for i in range(rows)] # print(rowlen,sum(rowlen),len(text)) # print([[sum(rowlen[0:i]),sum(rowlen[0:i+1])] for i in range(rows)])
        # words = '\n'.join([text[sum(rowlen[0:i]):sum(rowlen[0:i+1])] for i in range(rows)])
        words = ['\n'*i + text[sum(rowlen[0:i]):sum(rowlen[0:i+1])] for i in range(rows)]
        plotcurvelist([c for word in words for c in self.textcurves(word,holeless=holeless)])
def fonttests():
    Font().boundstest()
    Font().fontmetrics()
    Font().glyphmetrics('\n')
    Font().glyphmetrics('B')
    # Font(r'C:\Windows\Fonts\Caslon Stencil W01 D.ttf').fontmetrics()
    # Font().plottest()
def kerningtest():
    print(Font().has_kerning())
    print(Font().actually_has_kerning())
    # print(Font(r'C:\Windows\Fonts\Caslon Stencil W01 D.ttf').has_kerning())
    # print(Font(r'C:\Windows\Fonts\helvetica.ttf').has_kerning())
    print(Font().kerning_offset('A','W'),Font().kerning_offset('L','T'),Font().kerning_offset('T','A'),Font().kerning_offset('T','a'),Font().kerning_offset('Y','a'),Font().kerning_offset('W','o'))
def fontplots(font=r'C:\Windows\Fonts\Caslon Stencil W01 D.ttf',size=128):
    # Font().holelesscurvetest('BAR8akgoq02#@&$')
    Font(font,size).holelesscurvetest('BAR8akgoq0@#')
    # Font().plot('&',pause=0)
    Font(font,size).plot('&',m=1,pause=0)
    # Font(font,size).plot('%',m=1,pause=0)
    # Font(font,size).plot('‰',m=1,pause=0)
    # Font().plot('‰',m=1) # eliminateholes fails
    # plotcurvelist( Font().textcurves('Book\nsquiggly'), pause=0 )
    plotcurvelist( Font(font,size).textcurves('Book\nsquiggly'), pause=0 )

# \\advrpc36\c$\Windows\Fonts - list font filenames

if __name__ == '__main__':

    ## Font tests
    fonttests()
    kerningtest()
    fontplots()
    fontplots(r'C:\Windows\Fonts\Interstate-Bold.ttf')
    print( Font(r'C:\Windows\Fonts\Caslon Stencil W01 D.ttf').allchars().strip() )
    Font(r'C:\Windows\Fonts\Interstate-Bold.ttf').plotall(holeless=1)
    Font(r'C:\Windows\Fonts\Caslon Stencil W01 D.ttf').plotall()
    Font(r'C:\Windows\Fonts\arial.ttf').plotall()
    Font(r'C:\Windows\Fonts\verdana.ttf').plotall()
    Font(r'C:\Windows\Fonts\times.ttf').plotall()
    Font(r'C:\Windows\Fonts\akzidenz.ttf').plotall()
    Font(r'C:\Windows\Fonts\wingding.ttf').plotall() # no chr found to plot
    plotcurvelist( Font(r'C:\Windows\Fonts\Interstate-Bold.ttf').holelesscharcurves('®') )
    plotcurvelist( rotatecurves(Font(r'C:\Windows\Fonts\Interstate-Bold.ttf').holelesscharcurves('®'),angle=pi,x0=50,y0=50) )
