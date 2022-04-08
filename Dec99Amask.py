from lithomap import *
import numpy as np
import os

def standardonelayerlithomask(draftfolder,chipnum=None,chipmap=False):
    maskname = 'Dec99A'
    maskinfo = [maskname,'LN waveguide litho mask','EM 6200','5" mask','1.0um process']
    waferradius,flatradius,workingradius = 38100,36400,34000 # 34000 o-ring radius, 31000 tfln workingradius
    folder = 'masks 2022/'+maskname+'/'+draftfolder+'/'
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

if __name__ == '__main__':
    standardonelayerlithomask('draft 1')

    # import subprocess
    # subprocess.Popen(['C:/Program Files/Common Files/eDrawings2020/eDrawings.exe', 'C:/py/mask/mask.dxf'])
