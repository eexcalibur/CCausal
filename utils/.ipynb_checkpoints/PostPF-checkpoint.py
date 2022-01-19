import numpy as np
from graph_tool.all import *

def create_graph_by_adjmatrix(adj, pos_name, directed=True, show_weight=True, axes=None):
    g = Graph(directed=directed)
    g.add_edge_list(np.transpose(np.array(adj).nonzero()))
    #pos = arf_layout(g, max_iter=0)

    if pos_name == 'sfdp':
        pos = sfdp_layout(g)

    if pos_name == 'radial_tree':
        max_vertex = np.argmax(g.get_total_degrees(g.get_vertices()))
        pos  = radial_tree_layout(g, g.vertex(max_vertex))

    ##pos = planar_layout(g)

    if pos_name == 'cycle':
        state = minimize_nested_blockmodel_dl(g)
        t = get_hierarchy_tree(state)[0]
        tpos = pos = radial_tree_layout(t, t.vertex(t.num_vertices() - 1), weighted=True)
        pos = g.own_property(tpos)


    v_prop = g.new_vertex_property("string")
    for v,name in zip(g.vertices(),adj.columns):
        v_prop[v] = name
        
    e_prob = g.new_edge_property("string")
    #e_len  = g.new_edge_property("double")
    for e in g.edges():
        start =   int("%d" % (e.source()))
        end = int("%d" % (e.target()))
        e_prob[e] = "%.2f" % (adj.iloc[start,end])
    
    if axes is not None:
        axes.axis('off')
    
    if show_weight == True:
        graph_draw(g, pos=pos, vertex_text=v_prop, edge_text=e_prob, vertex_font_size=18, 
                   edge_font_size=14, edge_pen_width=6, mplfig=axes)
    else:
        graph_draw(g, pos=pos, vertex_text=v_prop, vertex_font_size=18, 
                   edge_pen_width=6, mplfig=axes)


    return g


#def plot_causal_method(result,title,reg_c,nol_c,lat,lon, lat_rgns, lon_rgns, mask=True):
#    fig, axes = plt.subplots(2,1,figsize=(7,5), subplot_kw={'projection': ccrs.PlateCarree(180)}, squeeze=False)
#    
#    if mask == True:
#        contour.plot_2d_contourf_by_array_region(fig, axes[0][0], np.ma.masked_array(result['reg'],mask=ocean_mask), 
#                              lat, lon, lat_rgns, lon_rgns, 'Regression', '', nol_c)
#        contour.plot_corrsig_only(axes[0][0], np.ma.masked_array(result["reg_p"], mask=ocean_mask).filled(fill_value=np.nan), 
#                              lat, lon, lat_rgns, lon_rgns, 0.05)
#        
#        contour.plot_2d_contourf_by_array_region(fig, axes[1][0], np.ma.masked_array(result['notears_linear'],mask=ocean_mask), 
#                                             lat, lon, lat_rgns, lon_rgns, "Notears_linear/Granger", '', nol_c)
#        contour.plot_corrsig_only(axes[0][0], np.ma.masked_array(result["granger"], mask=ocean_mask).filled(fill_value=np.nan), 
#                                              lat, lon, lat_rgns, lon_rgns, 0.05)       
#        
#    if mask == False:
#        contour.plot_2d_contourf_by_array_region(fig, axes[0][0], result['reg'], 
#                                                 lat, lon, lat_rgns, lon_rgns, 'Regression', '', reg_c)
#        contour.plot_corrsig_only(axes[0][0], result["reg_p"], lat, lon, lat_rgns, lon_rgns, 0.05)
#  
#        contour.plot_2d_contourf_by_array_region(fig, axes[1][0], result['notears_linear'], 
#                                                 lat, lon, lat_rgns, lon_rgns, "Notears_linear/Granger", '', nol_c)
#        contour.plot_corrsig_only(axes[1][0], result["granger"], lat, lon, lat_rgns, lon_rgns, 0.05)
#       
#    
#    fig.suptitle(title,fontsize=18)
#    
#    plt.tight_layout()
#
#def plot_causal_overlap(result,title,reg_c,nol_c,lat,lon, lat_rgns, lon_rgns, mask=True):
#    fig, axes = plt.subplots(2,1,figsize=(7,8), subplot_kw={'projection': ccrs.PlateCarree(180)}, squeeze=False)
#    
##     contour.plot_2d_contourf_by_array_region(fig, axes[0][0], np.ma.masked_array(result['TAS']['reg'],mask=ocean_mask), 
##                               lat, lon, lat_rgns, lon_rgns, 'Regression', '', nol_c)
##     contour.plot_corrsig_only(axes[0][0], np.ma.masked_array(result['TAS']["reg_p"], mask=ocean_mask).filled(fill_value=np.nan), 
##                               lat, lon, lat_rgns, lon_rgns, 0.05)
##     contour.plot_2d_contour_by_array_region(fig, axes[0][0], result['HGT']['reg'], 
##                               lat, lon, lat_rgns, lon_rgns, 'Regression')
#        
#    #contour.plot_2d_contourf_by_array_region(fig, axes[0][0], np.ma.masked_array(result['TAS']['notears_linear'],mask=ocean_mask), 
#    #                          lat, lon, lat_rgns, lon_rgns, 'Notears_linear/Granger', '', nol_c)
#    contour.plot_2d_contourf_by_array_region(fig, axes[0][0], result['TAS']['notears_linear'], 
#                              lat, lon, lat_rgns, lon_rgns, 'Notears_linear', '', nol_c)    
#    
#    contour.plot_corrsig_only(axes[1][0], np.ma.masked_array(result['TAS']["granger"], mask=ocean_mask).filled(fill_value=np.nan), 
#                              lat, lon, lat_rgns, lon_rgns, 0.05)
#    contour.plot_2d_contour_by_array_region(fig, axes[0][0], result['HGT']['notears_linear'], 
#                              lat, lon, lat_rgns, lon_rgns, 'Notears_linear')
#    
#    #fig.suptitle(title,fontsize=18)
#    
#    plt.tight_layout()
#
