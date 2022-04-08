from lithomap import *

class WaveguideChip(Chip):
    def addwaveguidechip(self):
        label = ' '+self.info.chipid
        self.addtext(self.finddefaultinfo('maskname')+label,x=2000,y=200)
        self.addtext(self.finddefaultinfo('maskname')+label,x=self.finddefaultinfo('chiplength')/2,y=200)
        self.info.guidespacing = 100
        self.info.groupspacing = 200
        self.adddiceguides()
        self.y += 300
        g1 = Element('group',parent=self).addguidelabels(dy=-50)
        for _ in range(7):
            g1.addchannel(3)
        g2 = Element('group',parent=self).addguidelabels(dy=-50)
        for _ in range(7):
            g2.addchannel(3)
        return self

def waveguidemaskApr22C(draftfolder,chipnum=None,chipmap=False):
    maskname = 'Apr22C'
    maskinfo = [maskname,'KTP waveguide litho mask','AF 7115','5" mask','1.0um process']
    waferradius,flatradius,workingradius = 38100,36400,34000
    folder = 'masks 2022/'+maskname+'/'+draftfolder+'/'
    os.makedirs(folder,exist_ok=True)
    mask = Element(maskname,layer='MASK')
    mask.info.font = r'Interstate-Bold.ttf'
    mask.info.minfeature = 1.0
    mask.info.rows = rw = 30
    mask.info.cols = cl = 2
    mask.info.chiplength = dx = 30000
    mask.info.chipwidth = dy = 2000
    chipid = [chipidtext(i,rw) for i in range(rw*cl)]
    def chipxy(i):
        ix,iy = i//rw,i%rw
        dys = [dy for i in range(rw)]
        return (dx*(ix-1), sum(dys[:iy]) - sum(dys)/2)
    for i in range(rw*cl):
        chip = WaveguideChip(parent=mask,id=chipid[i])
        if isinstance(chipmap,list) and not i in chipmap:
            continue
        chip.addwaveguidechip()
        chip.translate(*chipxy(i))
        if chipmap:
            chip.addnotesframe(margin=(1000,100))
            chipfolder = folder+'chipmaps/' if scalemag>1 else folder+'chips/'
            os.makedirs(chipfolder,exist_ok=True)
            chip.savemask(svg=True,txt=True,filename=f'{chipfolder}{maskname}-{chipid[i]}',png=savepng)
    if chipmap:
        return
    mask.roundoff(res=0.1,decimal=False)
    # mask.checkpolygonvalidity()
    for i in range(rw*cl):
        def vstr(v): return ' '.join(map(str,v)) if isinstance(v,(list, tuple)) else str(v)
        line = ','.join([','.join([str(k),vstr(v[i])]) for k,v in locals().items() if k.startswith('chip') and k not in ['chip','chipmap','chipnum','chipxy','chipenable','chipdx']])
        # print(line)
        print(line, file=open(f'{folder}{maskname}.csv','a' if i else 'w'))
    maskfiles = mask.savemask(maskname,layers=['MASK'],layernames=[maskname],
        svg=True,txt=True,png=savepng,folder=folder)
    for line in ['']+maskinfo+maskfiles+['']:
        print(line)

if __name__ == '__main__':
    waveguidemaskApr22C('draft 1')

    # import subprocess
    # subprocess.Popen(['C:/Program Files/Common Files/eDrawings2020/eDrawings.exe', 'C:/py/mask/mask.dxf'])
