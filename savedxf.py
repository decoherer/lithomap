import os,shutil,glob
import numpy as np
import svgwrite
import ezdxf
from geometry import PPfloat,autocadcolor,compresscurve,pdf2png,svg2pdf
from collections import defaultdict

def savemask(self,filename,txt=False,layers=[],layernames=[],svg=True,png=True,gds=True,pdf=True,verbose=True,folder=None,dev=0,scalemag=1,pickle=True):
    if folder:
        folder += '' if '/'==folder[-1] else '/'
        os.makedirs(folder,exist_ok=True)
        filename = folder+filename
        layernames = [folder+layernames for _ in layers] if isinstance(layernames,str) else [folder+name for name in layernames]
    if verbose: print('  saving.')
    if dev and not 1==scalemag: self.scale(1,scalemag)
    self.savedxfwithlayers(filename,verbose=verbose)
    # os.popen('copy "'+filename+'.dxf" mask.dxf')
    shutil.copy(filename+'.dxf','mask.dxf')
    if gds and layers and not dev:
        for layer,name in zip(layers,layernames):
            self.savegds(filename=name+layer,layer=layer)
    if svg:
        self.savedxfwithlayers(filename,svg=1,verbose=verbose)
        # os.popen('copy "'+filename+'.svg" mask.svg')
        # shutil.copy(filename+'.svg','mask.svg')
    if pdf and svg:
        if verbose: print('  saving "'+filename+'.pdf"...')
        # os.popen('"C:\\Program Files\\Inkscape\\inkscape.com" '+filename+'.svg --export-png='+filename+'.png')
        svg2pdf(filename+'.svg')
    if png:
        assert svg and pdf
        if verbose: print('  saving"'+filename+'.png"...')
        # os.popen('"C:\\Program Files\\Inkscape\\inkscape.com" '+filename+'.svg --export-png='+filename+'.png')
        # svg2png(filename+'.svg')
        pdf2png(filename+'.pdf')
    if dev and not 1==scalemag: self.scale(1,1./scalemag)
    if txt:
        with open(filename+'.txt','w',encoding='utf-8') as file: file.write(str(self))
        if verbose: print('  "'+filename+'.txt" saved.')
    if layers and not dev:
        if pickle:
            self.pickle(filename)
        for layer,name in zip(layers,layernames):
            self.savedxfwithlayers(filename=name+layer,singlelayertosave=layer,verbose=verbose)
            # "\\ADVRSVR2\AdvR Lab Shared Space\015 Litho and Electrode Masks\2022\Feb22C\draft1\Feb22CMASK.dxf"
    if folder and not dev:
        # maskfolder = '/'.join(filename.split('/')[:-2])
        maskfolder = '/'.join(filename.split('/')[:-1])
        for f in glob.glob('*.py'):
            shutil.copy(f,maskfolder)
        for f in glob.glob('*.csv'):
            shutil.copy(f,maskfolder)
    return [f'"P:/ADVRLabSharedSpace/015 Litho and Electrode Masks/2024/{name+layer}.dxf"'.replace('masks 2024/','').replace('/','\\') for layer,name in zip(layers,layernames)]

def savedxfwithlayers(self,filename='',singlelayertosave='',svg=False,svgdebug=False,verbose=False,nomodify=True,scalemag=1):
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
        drawing = ezdxf.new('AC1015')
        space = drawing.modelspace()
        drawing.styles.new('arial', dxfattribs={'font':'arial.ttf'})
    # drawing.styles.new('custom1', dxfattribs={'font':'times.ttf'})
    # space.add_text("Text Style Example: Times New Roman", dxfattribs={'style':'custom1','height':1500,'color':7}).set_pos((-22000,22000), align='LEFT')
    # pink 11, grey pink 13, yellow-grey 53, blue 142, dark magenta 242, light grey 254, red 1, yellow 2, green 3, lt blue 4, blue 5, magenta 6, white 7, dark grey 8, grey 9, darker grey 250 # gold 42, dark yellow 52, green 62
    # dark blue 146, purple grey 207, dark red 16, orange 32, gold 42, dark tan 37, tan 33
    # http://sub-atomic.com/~moses/acadcolors.html (gone) # try http://gohtx.com/acadcolors.php (or https://stackoverflow.com/questions/43876109/  or  https://www.google.com/search?q=http://sub-atomic.com/~moses/acadcolors.html&hl=en&source=lnms&tbm=isch)
    # https://gist.github.com/kuka/f2a5a77b715531c7ae9beae8d5cbfded # acadcolors = [{"aci":0,"hex":"#000000","rgb":"rgb(0,0,0)"}, {"aci":1,"hex":"#FF0000","rgb":"rgb(255,0,0)"}, {"aci":2,"hex":"#FFFF00","rgb":"rgb(255,255,0)"}, {"aci":3,"hex":"#00FF00","rgb":"rgb(0,255,0)"}, {"aci":4,"hex":"#00FFFF","rgb":"rgb(0,255,255)"}, {"aci":5,"hex":"#0000FF","rgb":"rgb(0,0,255)"}, {"aci":6,"hex":"#FF00FF","rgb":"rgb(255,0,255)"}, {"aci":7,"hex":"#FFFFFF","rgb":"rgb(255,255,255)"}, {"aci":8,"hex":"#414141","rgb":"rgb(65,65,65)"}, {"aci":9,"hex":"#808080","rgb":"rgb(128,128,128)"}, {"aci":10,"hex":"#FF0000","rgb":"rgb(255,0,0)"}, {"aci":11,"hex":"#FFAAAA","rgb":"rgb(255,170,170)"}, {"aci":12,"hex":"#BD0000","rgb":"rgb(189,0,0)"}, {"aci":13,"hex":"#BD7E7E","rgb":"rgb(189,126,126)"}, {"aci":14,"hex":"#810000","rgb":"rgb(129,0,0)"}, {"aci":15,"hex":"#815656","rgb":"rgb(129,86,86)"}, {"aci":16,"hex":"#680000","rgb":"rgb(104,0,0)"}, {"aci":17,"hex":"#684545","rgb":"rgb(104,69,69)"}, {"aci":18,"hex":"#4F0000","rgb":"rgb(79,0,0)"}, {"aci":19,"hex":"#4F3535","rgb":"rgb(79,53,53)"}, {"aci":20,"hex":"#FF3F00","rgb":"rgb(255,63,0)"}, {"aci":21,"hex":"#FFBFAA","rgb":"rgb(255,191,170)"}, {"aci":22,"hex":"#BD2E00","rgb":"rgb(189,46,0)"}, {"aci":23,"hex":"#BD8D7E","rgb":"rgb(189,141,126)"}, {"aci":24,"hex":"#811F00","rgb":"rgb(129,31,0)"}, {"aci":25,"hex":"#816056","rgb":"rgb(129,96,86)"}, {"aci":26,"hex":"#681900","rgb":"rgb(104,25,0)"}, {"aci":27,"hex":"#684E45","rgb":"rgb(104,78,69)"}, {"aci":28,"hex":"#4F1300","rgb":"rgb(79,19,0)"}, {"aci":29,"hex":"#4F3B35","rgb":"rgb(79,59,53)"}, {"aci":30,"hex":"#FF7F00","rgb":"rgb(255,127,0)"}, {"aci":31,"hex":"#FFD4AA","rgb":"rgb(255,212,170)"}, {"aci":32,"hex":"#BD5E00","rgb":"rgb(189,94,0)"}, {"aci":33,"hex":"#BD9D7E","rgb":"rgb(189,157,126)"}, {"aci":34,"hex":"#814000","rgb":"rgb(129,64,0)"}, {"aci":35,"hex":"#816B56","rgb":"rgb(129,107,86)"}, {"aci":36,"hex":"#683400","rgb":"rgb(104,52,0)"}, {"aci":37,"hex":"#685645","rgb":"rgb(104,86,69)"}, {"aci":38,"hex":"#4F2700","rgb":"rgb(79,39,0)"}, {"aci":39,"hex":"#4F4235","rgb":"rgb(79,66,53)"}, {"aci":40,"hex":"#FFBF00","rgb":"rgb(255,191,0)"}, {"aci":41,"hex":"#FFEAAA","rgb":"rgb(255,234,170)"}, {"aci":42,"hex":"#BD8D00","rgb":"rgb(189,141,0)"}, {"aci":43,"hex":"#BDAD7E","rgb":"rgb(189,173,126)"}, {"aci":44,"hex":"#816000","rgb":"rgb(129,96,0)"}, {"aci":45,"hex":"#817656","rgb":"rgb(129,118,86)"}, {"aci":46,"hex":"#684E00","rgb":"rgb(104,78,0)"}, {"aci":47,"hex":"#685F45","rgb":"rgb(104,95,69)"}, {"aci":48,"hex":"#4F3B00","rgb":"rgb(79,59,0)"}, {"aci":49,"hex":"#4F4935","rgb":"rgb(79,73,53)"}, {"aci":50,"hex":"#FFFF00","rgb":"rgb(255,255,0)"}, {"aci":51,"hex":"#FFFFAA","rgb":"rgb(255,255,170)"}, {"aci":52,"hex":"#BDBD00","rgb":"rgb(189,189,0)"}, {"aci":53,"hex":"#BDBD7E","rgb":"rgb(189,189,126)"}, {"aci":54,"hex":"#818100","rgb":"rgb(129,129,0)"}, {"aci":55,"hex":"#818156","rgb":"rgb(129,129,86)"}, {"aci":56,"hex":"#686800","rgb":"rgb(104,104,0)"}, {"aci":57,"hex":"#686845","rgb":"rgb(104,104,69)"}, {"aci":58,"hex":"#4F4F00","rgb":"rgb(79,79,0)"}, {"aci":59,"hex":"#4F4F35","rgb":"rgb(79,79,53)"}, {"aci":60,"hex":"#BFFF00","rgb":"rgb(191,255,0)"}, {"aci":61,"hex":"#EAFFAA","rgb":"rgb(234,255,170)"}, {"aci":62,"hex":"#8DBD00","rgb":"rgb(141,189,0)"}, {"aci":63,"hex":"#ADBD7E","rgb":"rgb(173,189,126)"}, {"aci":64,"hex":"#608100","rgb":"rgb(96,129,0)"}, {"aci":65,"hex":"#768156","rgb":"rgb(118,129,86)"}, {"aci":66,"hex":"#4E6800","rgb":"rgb(78,104,0)"}, {"aci":67,"hex":"#5F6845","rgb":"rgb(95,104,69)"}, {"aci":68,"hex":"#3B4F00","rgb":"rgb(59,79,0)"}, {"aci":69,"hex":"#494F35","rgb":"rgb(73,79,53)"}, {"aci":70,"hex":"#7FFF00","rgb":"rgb(127,255,0)"}, {"aci":71,"hex":"#D4FFAA","rgb":"rgb(212,255,170)"}, {"aci":72,"hex":"#5EBD00","rgb":"rgb(94,189,0)"}, {"aci":73,"hex":"#9DBD7E","rgb":"rgb(157,189,126)"}, {"aci":74,"hex":"#408100","rgb":"rgb(64,129,0)"}, {"aci":75,"hex":"#6B8156","rgb":"rgb(107,129,86)"}, {"aci":76,"hex":"#346800","rgb":"rgb(52,104,0)"}, {"aci":77,"hex":"#566845","rgb":"rgb(86,104,69)"}, {"aci":78,"hex":"#274F00","rgb":"rgb(39,79,0)"}, {"aci":79,"hex":"#424F35","rgb":"rgb(66,79,53)"}, {"aci":80,"hex":"#3FFF00","rgb":"rgb(63,255,0)"}, {"aci":81,"hex":"#BFFFAA","rgb":"rgb(191,255,170)"}, {"aci":82,"hex":"#2EBD00","rgb":"rgb(46,189,0)"}, {"aci":83,"hex":"#8DBD7E","rgb":"rgb(141,189,126)"}, {"aci":84,"hex":"#1F8100","rgb":"rgb(31,129,0)"}, {"aci":85,"hex":"#608156","rgb":"rgb(96,129,86)"}, {"aci":86,"hex":"#196800","rgb":"rgb(25,104,0)"}, {"aci":87,"hex":"#4E6845","rgb":"rgb(78,104,69)"}, {"aci":88,"hex":"#134F00","rgb":"rgb(19,79,0)"}, {"aci":89,"hex":"#3B4F35","rgb":"rgb(59,79,53)"}, {"aci":90,"hex":"#00FF00","rgb":"rgb(0,255,0)"}, {"aci":91,"hex":"#AAFFAA","rgb":"rgb(170,255,170)"}, {"aci":92,"hex":"#00BD00","rgb":"rgb(0,189,0)"}, {"aci":93,"hex":"#7EBD7E","rgb":"rgb(126,189,126)"}, {"aci":94,"hex":"#008100","rgb":"rgb(0,129,0)"}, {"aci":95,"hex":"#568156","rgb":"rgb(86,129,86)"}, {"aci":96,"hex":"#006800","rgb":"rgb(0,104,0)"}, {"aci":97,"hex":"#456845","rgb":"rgb(69,104,69)"}, {"aci":98,"hex":"#004F00","rgb":"rgb(0,79,0)"}, {"aci":99,"hex":"#354F35","rgb":"rgb(53,79,53)"}, {"aci":100,"hex":"#00FF3F","rgb":"rgb(0,255,63)"}, {"aci":101,"hex":"#AAFFBF","rgb":"rgb(170,255,191)"}, {"aci":102,"hex":"#00BD2E","rgb":"rgb(0,189,46)"}, {"aci":103,"hex":"#7EBD8D","rgb":"rgb(126,189,141)"}, {"aci":104,"hex":"#00811F","rgb":"rgb(0,129,31)"}, {"aci":105,"hex":"#568160","rgb":"rgb(86,129,96)"}, {"aci":106,"hex":"#006819","rgb":"rgb(0,104,25)"}, {"aci":107,"hex":"#45684E","rgb":"rgb(69,104,78)"}, {"aci":108,"hex":"#004F13","rgb":"rgb(0,79,19)"}, {"aci":109,"hex":"#354F3B","rgb":"rgb(53,79,59)"}, {"aci":110,"hex":"#00FF7F","rgb":"rgb(0,255,127)"}, {"aci":111,"hex":"#AAFFD4","rgb":"rgb(170,255,212)"}, {"aci":112,"hex":"#00BD5E","rgb":"rgb(0,189,94)"}, {"aci":113,"hex":"#7EBD9D","rgb":"rgb(126,189,157)"}, {"aci":114,"hex":"#008140","rgb":"rgb(0,129,64)"}, {"aci":115,"hex":"#56816B","rgb":"rgb(86,129,107)"}, {"aci":116,"hex":"#006834","rgb":"rgb(0,104,52)"}, {"aci":117,"hex":"#456856","rgb":"rgb(69,104,86)"}, {"aci":118,"hex":"#004F27","rgb":"rgb(0,79,39)"}, {"aci":119,"hex":"#354F42","rgb":"rgb(53,79,66)"}, {"aci":120,"hex":"#00FFBF","rgb":"rgb(0,255,191)"}, {"aci":121,"hex":"#AAFFEA","rgb":"rgb(170,255,234)"}, {"aci":122,"hex":"#00BD8D","rgb":"rgb(0,189,141)"}, {"aci":123,"hex":"#7EBDAD","rgb":"rgb(126,189,173)"}, {"aci":124,"hex":"#008160","rgb":"rgb(0,129,96)"}, {"aci":125,"hex":"#568176","rgb":"rgb(86,129,118)"}, {"aci":126,"hex":"#00684E","rgb":"rgb(0,104,78)"}, {"aci":127,"hex":"#45685F","rgb":"rgb(69,104,95)"}, {"aci":128,"hex":"#004F3B","rgb":"rgb(0,79,59)"}, {"aci":129,"hex":"#354F49","rgb":"rgb(53,79,73)"}, {"aci":130,"hex":"#00FFFF","rgb":"rgb(0,255,255)"}, {"aci":131,"hex":"#AAFFFF","rgb":"rgb(170,255,255)"}, {"aci":132,"hex":"#00BDBD","rgb":"rgb(0,189,189)"}, {"aci":133,"hex":"#7EBDBD","rgb":"rgb(126,189,189)"}, {"aci":134,"hex":"#008181","rgb":"rgb(0,129,129)"}, {"aci":135,"hex":"#568181","rgb":"rgb(86,129,129)"}, {"aci":136,"hex":"#006868","rgb":"rgb(0,104,104)"}, {"aci":137,"hex":"#456868","rgb":"rgb(69,104,104)"}, {"aci":138,"hex":"#004F4F","rgb":"rgb(0,79,79)"}, {"aci":139,"hex":"#354F4F","rgb":"rgb(53,79,79)"}, {"aci":140,"hex":"#00BFFF","rgb":"rgb(0,191,255)"}, {"aci":141,"hex":"#AAEAFF","rgb":"rgb(170,234,255)"}, {"aci":142,"hex":"#008DBD","rgb":"rgb(0,141,189)"}, {"aci":143,"hex":"#7EADBD","rgb":"rgb(126,173,189)"}, {"aci":144,"hex":"#006081","rgb":"rgb(0,96,129)"}, {"aci":145,"hex":"#567681","rgb":"rgb(86,118,129)"}, {"aci":146,"hex":"#004E68","rgb":"rgb(0,78,104)"}, {"aci":147,"hex":"#455F68","rgb":"rgb(69,95,104)"}, {"aci":148,"hex":"#003B4F","rgb":"rgb(0,59,79)"}, {"aci":149,"hex":"#35494F","rgb":"rgb(53,73,79)"}, {"aci":150,"hex":"#007FFF","rgb":"rgb(0,127,255)"}, {"aci":151,"hex":"#AAD4FF","rgb":"rgb(170,212,255)"}, {"aci":152,"hex":"#005EBD","rgb":"rgb(0,94,189)"}, {"aci":153,"hex":"#7E9DBD","rgb":"rgb(126,157,189)"}, {"aci":154,"hex":"#004081","rgb":"rgb(0,64,129)"}, {"aci":155,"hex":"#566B81","rgb":"rgb(86,107,129)"}, {"aci":156,"hex":"#003468","rgb":"rgb(0,52,104)"}, {"aci":157,"hex":"#455668","rgb":"rgb(69,86,104)"}, {"aci":158,"hex":"#00274F","rgb":"rgb(0,39,79)"}, {"aci":159,"hex":"#35424F","rgb":"rgb(53,66,79)"}, {"aci":160,"hex":"#003FFF","rgb":"rgb(0,63,255)"}, {"aci":161,"hex":"#AABFFF","rgb":"rgb(170,191,255)"}, {"aci":162,"hex":"#002EBD","rgb":"rgb(0,46,189)"}, {"aci":163,"hex":"#7E8DBD","rgb":"rgb(126,141,189)"}, {"aci":164,"hex":"#001F81","rgb":"rgb(0,31,129)"}, {"aci":165,"hex":"#566081","rgb":"rgb(86,96,129)"}, {"aci":166,"hex":"#001968","rgb":"rgb(0,25,104)"}, {"aci":167,"hex":"#454E68","rgb":"rgb(69,78,104)"}, {"aci":168,"hex":"#00134F","rgb":"rgb(0,19,79)"}, {"aci":169,"hex":"#353B4F","rgb":"rgb(53,59,79)"}, {"aci":170,"hex":"#0000FF","rgb":"rgb(0,0,255)"}, {"aci":171,"hex":"#AAAAFF","rgb":"rgb(170,170,255)"}, {"aci":172,"hex":"#0000BD","rgb":"rgb(0,0,189)"}, {"aci":173,"hex":"#7E7EBD","rgb":"rgb(126,126,189)"}, {"aci":174,"hex":"#000081","rgb":"rgb(0,0,129)"}, {"aci":175,"hex":"#565681","rgb":"rgb(86,86,129)"}, {"aci":176,"hex":"#000068","rgb":"rgb(0,0,104)"}, {"aci":177,"hex":"#454568","rgb":"rgb(69,69,104)"}, {"aci":178,"hex":"#00004F","rgb":"rgb(0,0,79)"}, {"aci":179,"hex":"#35354F","rgb":"rgb(53,53,79)"}, {"aci":180,"hex":"#3F00FF","rgb":"rgb(63,0,255)"}, {"aci":181,"hex":"#BFAAFF","rgb":"rgb(191,170,255)"}, {"aci":182,"hex":"#2E00BD","rgb":"rgb(46,0,189)"}, {"aci":183,"hex":"#8D7EBD","rgb":"rgb(141,126,189)"}, {"aci":184,"hex":"#1F0081","rgb":"rgb(31,0,129)"}, {"aci":185,"hex":"#605681","rgb":"rgb(96,86,129)"}, {"aci":186,"hex":"#190068","rgb":"rgb(25,0,104)"}, {"aci":187,"hex":"#4E4568","rgb":"rgb(78,69,104)"}, {"aci":188,"hex":"#13004F","rgb":"rgb(19,0,79)"}, {"aci":189,"hex":"#3B354F","rgb":"rgb(59,53,79)"}, {"aci":190,"hex":"#7F00FF","rgb":"rgb(127,0,255)"}, {"aci":191,"hex":"#D4AAFF","rgb":"rgb(212,170,255)"}, {"aci":192,"hex":"#5E00BD","rgb":"rgb(94,0,189)"}, {"aci":193,"hex":"#9D7EBD","rgb":"rgb(157,126,189)"}, {"aci":194,"hex":"#400081","rgb":"rgb(64,0,129)"}, {"aci":195,"hex":"#6B5681","rgb":"rgb(107,86,129)"}, {"aci":196,"hex":"#340068","rgb":"rgb(52,0,104)"}, {"aci":197,"hex":"#564568","rgb":"rgb(86,69,104)"}, {"aci":198,"hex":"#27004F","rgb":"rgb(39,0,79)"}, {"aci":199,"hex":"#42354F","rgb":"rgb(66,53,79)"}, {"aci":200,"hex":"#BF00FF","rgb":"rgb(191,0,255)"}, {"aci":201,"hex":"#EAAAFF","rgb":"rgb(234,170,255)"}, {"aci":202,"hex":"#8D00BD","rgb":"rgb(141,0,189)"}, {"aci":203,"hex":"#AD7EBD","rgb":"rgb(173,126,189)"}, {"aci":204,"hex":"#600081","rgb":"rgb(96,0,129)"}, {"aci":205,"hex":"#765681","rgb":"rgb(118,86,129)"}, {"aci":206,"hex":"#4E0068","rgb":"rgb(78,0,104)"}, {"aci":207,"hex":"#5F4568","rgb":"rgb(95,69,104)"}, {"aci":208,"hex":"#3B004F","rgb":"rgb(59,0,79)"}, {"aci":209,"hex":"#49354F","rgb":"rgb(73,53,79)"}, {"aci":210,"hex":"#FF00FF","rgb":"rgb(255,0,255)"}, {"aci":211,"hex":"#FFAAFF","rgb":"rgb(255,170,255)"}, {"aci":212,"hex":"#BD00BD","rgb":"rgb(189,0,189)"}, {"aci":213,"hex":"#BD7EBD","rgb":"rgb(189,126,189)"}, {"aci":214,"hex":"#810081","rgb":"rgb(129,0,129)"}, {"aci":215,"hex":"#815681","rgb":"rgb(129,86,129)"}, {"aci":216,"hex":"#680068","rgb":"rgb(104,0,104)"}, {"aci":217,"hex":"#684568","rgb":"rgb(104,69,104)"}, {"aci":218,"hex":"#4F004F","rgb":"rgb(79,0,79)"}, {"aci":219,"hex":"#4F354F","rgb":"rgb(79,53,79)"}, {"aci":220,"hex":"#FF00BF","rgb":"rgb(255,0,191)"}, {"aci":221,"hex":"#FFAAEA","rgb":"rgb(255,170,234)"}, {"aci":222,"hex":"#BD008D","rgb":"rgb(189,0,141)"}, {"aci":223,"hex":"#BD7EAD","rgb":"rgb(189,126,173)"}, {"aci":224,"hex":"#810060","rgb":"rgb(129,0,96)"}, {"aci":225,"hex":"#815676","rgb":"rgb(129,86,118)"}, {"aci":226,"hex":"#68004E","rgb":"rgb(104,0,78)"}, {"aci":227,"hex":"#68455F","rgb":"rgb(104,69,95)"}, {"aci":228,"hex":"#4F003B","rgb":"rgb(79,0,59)"}, {"aci":229,"hex":"#4F3549","rgb":"rgb(79,53,73)"}, {"aci":230,"hex":"#FF007F","rgb":"rgb(255,0,127)"}, {"aci":231,"hex":"#FFAAD4","rgb":"rgb(255,170,212)"}, {"aci":232,"hex":"#BD005E","rgb":"rgb(189,0,94)"}, {"aci":233,"hex":"#BD7E9D","rgb":"rgb(189,126,157)"}, {"aci":234,"hex":"#810040","rgb":"rgb(129,0,64)"}, {"aci":235,"hex":"#81566B","rgb":"rgb(129,86,107)"}, {"aci":236,"hex":"#680034","rgb":"rgb(104,0,52)"}, {"aci":237,"hex":"#684556","rgb":"rgb(104,69,86)"}, {"aci":238,"hex":"#4F0027","rgb":"rgb(79,0,39)"}, {"aci":239,"hex":"#4F3542","rgb":"rgb(79,53,66)"}, {"aci":240,"hex":"#FF003F","rgb":"rgb(255,0,63)"}, {"aci":241,"hex":"#FFAABF","rgb":"rgb(255,170,191)"}, {"aci":242,"hex":"#BD002E","rgb":"rgb(189,0,46)"}, {"aci":243,"hex":"#BD7E8D","rgb":"rgb(189,126,141)"}, {"aci":244,"hex":"#81001F","rgb":"rgb(129,0,31)"}, {"aci":245,"hex":"#815660","rgb":"rgb(129,86,96)"}, {"aci":246,"hex":"#680019","rgb":"rgb(104,0,25)"}, {"aci":247,"hex":"#68454E","rgb":"rgb(104,69,78)"}, {"aci":248,"hex":"#4F0013","rgb":"rgb(79,0,19)"}, {"aci":249,"hex":"#4F353B","rgb":"rgb(79,53,59)"}, {"aci":250,"hex":"#333333","rgb":"rgb(51,51,51)"}, {"aci":251,"hex":"#505050","rgb":"rgb(80,80,80)"}, {"aci":252,"hex":"#696969","rgb":"rgb(105,105,105)"}, {"aci":253,"hex":"#828282","rgb":"rgb(130,130,130)"}, {"aci":254,"hex":"#BEBEBE","rgb":"rgb(190,190,190)"}, {"aci":255,"hex":"#FFFFFF","rgb":"rgb(255,255,255)"}]
    lcolor = defaultdict(lambda:1)
    lcolor.update({'MASK':42,'ELECTRODE':252,'METAL':252,'POLING':250,'NOTES':7,'IMPLANT':32,'OXIDE':11,'BUFFER':132,'DICE':22,'MOCKUP':8,'WG':42,'APE':32,'BU':22,'EL':250,'1':42,'2':32,'3':22,'4':250})
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
                        (k,v),dy,draw = [(k,note.note[k]) for k in note.note][0],note.dy,note.draw
                        ncolor = {'grating':31,'width':16,'taper':33,'length':33,'separation':242,'displacement':242,'roc':142,
                            'ewidth0':151,'ewidth':151,'ewidth2':151,'outwidth':151,'inwidth':151,'hot':151,'gap':223,
                            'egap0':223,'egap':223,'egap2':223,'outgap':223,'outgap2':223,'ingap':223,'ingap2':223,
                            'bragg':32,'mzub':32,'mzul':32,'splitradius':9,'implant':32,'buffer':132,'electrode':252}
                        rep = {'outwidth':'%d','outgap':'%d','length':'%d','separation':'↕%g','displacement':'↕%g', # 'displacement':'↕%.1f',
                            'roc':'%.2fmm ROC','mzub':'%s','mzul':'%s','splitradius':'%gr'}
                        offset = {'width':(-1,0),'taper':(0,0.75),'length':(0,0.75),'roc':(0,-1),
                            'outwidth':(0,+1.5),'outgap':(0,+2.5),'outgap2':(0,+0.5),
                            'inwidth':(0,-1.5),'ingap':(0,-2.5),'ingap2':(0,-0.5),
                            'ewidth0':(0,0),'ewidth':(+1,0),'egap0':(0,0),'egap':(+2,0),'ewidth2':(-1,0),'egap2':(-2,0),
                            'bragg':(4,-0.5),'mzub':(0,0.75),'mzul':(0,0.75)}
                        color = ncolor[k] if k in ncolor else lcolor['NOTES']
                        s = rep[k]%v if k in rep else ( v if isinstance(v,str) else str(int(v) if int(v)==v else v) )
                        xo,yo = offset[k] if k in offset else (0,0)
                        xx,yy = note.x+xo*dy,(note.y+0.5*dy+yo*dy) # for dxf, middle of digit is at 0.5*height
                        if svg:
                            def darken(rgb):
                                return tuple(int(0.75*ci) for ci in rgb)
                            svgtextscale,svgcolor = 1.5,'rgb'+str(darken(autocadcolor(color)))
                            g.add(dwg.text(s,insert=(scale(xx),scale(yy-0.5*dy+svgtextscale*.375*dy)),font_size=scale(svgtextscale*dy),fill=svgcolor)) # for svg, middle of digit seems to be at 0.375*height
                            if draw:
                                for seg in draw:
                                    g.add(dwg.polyline([(100*x,100*y) for x,y in seg],fill='none',stroke='darkblue'))
                        else:
                            dxfattribs={'color':color,'layer':'NOTES','height':dy,'style':'arial'}
                            # space.add_text(s, dxfattribs=dxfattribs).set_pos((xx,-yy), align='MIDDLE_CENTER') # pre ezdxf 0.18
                            space.add_text(s, dxfattribs=dxfattribs).set_placement((xx,-yy), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
                            #space.add_mtext(s) # use mtext for multiple lines?
                            if draw:
                                for seg in draw:
                                    space.add_lwpolyline([(x,-y) for x,y in seg], dxfattribs={'color':color,'layer':'NOTES'})

    #space.add_text('X', dxfattribs={'layer':'NOTES','height':2000,'color':53}).set_pos((0,2000), align='CENTER') # anchor = {'taper':'TOP_CENTER','length':'TOP_CENTER','roc':'TOP_CENTER'} # a = anchor[k] if k in anchor else 'CENTER'
    if svg:
        if not nomodify:
            self.scale(1/svgscale)
        dwg.save()
    else:
        drawing.saveas(filename)
    if verbose: print('  "'+filename+'" saved.')
