#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


## load bokeh packages (for interactive plots)
from bokeh.io import curdoc
from bokeh.models.widgets import Tabs



## ALL BOX PLOTS ***********************************************************************************

from scripts.boxPlot import boxplot_tab
from scripts.barPlot import barplot_tab
tab_box = boxplot_tab()
tab_bar = barplot_tab()

## assign tabs to be displayed
tabs = Tabs(tabs=[tab_box, tab_bar])



## CREATE APPLICATION WITH VARIOUS TABS ----------------------------------------------
## Put the tabs in the current document for display ----------------------------------

curdoc().add_root(tabs)
curdoc().title = "EnergyConsumption"
