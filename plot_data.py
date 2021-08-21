from pygnuplot import gnuplot
g = gnuplot.Gnuplot()
g.set('noxlabel', 'noylabel', 'noxrange', 'noyrange')
g.plot('sin(x) with lp')
