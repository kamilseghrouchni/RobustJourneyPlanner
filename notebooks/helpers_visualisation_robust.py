import plotly.express as px
import plotly.graph_objects as go
from ipywidgets import widgets
from IPython.display import display, HTML
from contextlib import contextmanager
from helpers_robust import *


"""
Implementations for ploting the journey 
"""
def get_coord_lat(nodes,stop) :
    """
    Returns lat coordinates
    """
    stop_coord = nodes.loc[stop] 
    return stop_coord.lat

def get_coord_lon(nodes_id,stop) : 
    """
    Returns long coordinates
    """
    stop_coord = nodes_id.loc[stop] 
    return stop_coord.lon



def plot_path_robust(graph_nodes, graph_strong_connected, route=[]):
    
    """
    Given a list of latitudes and longitudes, origin 
    and destination point, plots a path on a map
    
    Parameters
    ----------
    lat, long: list of latitudes and longitudes
    origin_point, destination_point: co-ordinates of origin
    and destination
    Returns
    -------
    Nothing. Only shows the map.
    
    
    """   
    
    # instantiate ipywidgets 
    tab = widgets.Tab()    
    out = widgets.Output(layout={'border': '1px solid black'})
    # set upper toggles 
    tab.set_title(0, "Your journey planner")
    tab.set_title(1, "Trip overview")
    tab.set_title(2, "Trip's probability")
    
    # setting widget dropdown for start station 
    
    #sort alphabetically all possible stations
    names_1 = sorted(graph_nodes.stop_name.unique())
    #declare widget with default station
    textbox_src = widgets.Dropdown(
    description='Start stations : ',
    value='Oberhasli, Industrie',
    options=names_1)
    
    # setting widget dropdown for start station 
    #sort alphabetically all possible stations
    names_2 = sorted(graph_nodes.stop_name.unique())
    textbox_dst = widgets.Dropdown(
    description='End stations : ',
    value='Zürich, Sädlenweg',
    options=names_2)
     
    # setting widget for arrival time entry
    # declare widget
    textbox_time =widgets.Text(
    description='Arrival time : ',
    value='15:00:00',
    placeholder='Type something',
    disabled=False)
    
    # setting widget for confidence threshold
    # declare widget
    textbox_conf =widgets.Text(
    description='Confidence Threshold (between 0 and 1) : ',
    value='0.9',
    placeholder='Type something',
    disabled=False)
    
    # declare widget to compute journey for specified start and stop stations as well as arrival time
    button_apply = widgets.Button(
    description='Apply',
    disabled=False,
    button_style='warning',
    tooltip='Search your route',
    icon='check')
    
    #declare accordion object to embed widget in same object
    accordion = widgets.Accordion(children=[textbox_src,textbox_dst,textbox_time,textbox_conf])
    
    # declare name of the toggle corresponding to declared widgets
    accordion.set_title(0, 'Start station')
    accordion.set_title(1, 'Arrival station')
    accordion.set_title(2, 'Arrival time')
    accordion.set_title(3, 'Confidence Threshold :')

    # widget to display computation status before planne
    computation_state =  widgets.HTML(
    value="<font color='green'><b>READY !</b></font>",
    description="<font size='2'><b>Status</b></font>:"
)
    # widget to display if trip details are available
    route_plan =  widgets.HTML(
    value="<font color='black'><b>No route computed </b></font>",
    description="<font size='3'><b>Satuts</b></font>:"
)
    
    #widget to display trip probability     
    probability_widget =  widgets.HTML(
    value="",
    description=""
)

    # compute cooridinates for start and end points
    origin = create_point(get_entry(graph_nodes,textbox_src.value))
    stop = create_point(get_entry(graph_nodes,textbox_dst.value))
   
    # adding the lines joining the nodes
    line = go.Scattermapbox(
        name = "Path",
        mode = "lines",
        lon = [],
        lat =[],
        marker = {'size': 5},
        line = dict(width = 4.5, color = 'blue'))
    
    # adding stations marker
    points=go.Scattermapbox(
        name = "Stops",
        mode = "markers",
        text=graph_nodes.stop_name,
        lon = graph_nodes.lon,
        lat = graph_nodes.lat,
        marker = {'size': 5, 'color':"red"})
    
    # adding origin marker
    source=go.Scattermapbox(
        name = "Source",
        mode = "markers",
        lon =  origin["lon"] ,
        lat =  origin["lat"],
        text = textbox_src.value,
        marker = {'size': (15), 'color':"orange"})
    
    # adding end marker
    end = go.Scattermapbox(
        name = "Arrival",
        mode = "markers",
        lon =  stop["lon"] ,
        lat =  stop["lat"],
        text = textbox_dst.value,
        marker = {'size': 15, 'color':"green"})
 
    # sanity check on widget entries before computing the route 
    def validate():
        if textbox_src.value in sorted(graph_nodes.stop_name)\
        and textbox_dst.value in  sorted(graph_nodes.stop_name)\
        and time_callback():
            return True
        else:
            return False
        
    # sanity check on text entry for the arrival time 
    def time_callback():
            try:
                string_datetime(textbox_time.value)
                return True
            except ValueError:
                computation_state.value = "<font color='red'><b>TERMINATED</b> (Incorrect data format, should be hh:mm:ss)</font>"
                raise ValueError("Incorrect data format, should be hh:mm:ss")
                return False
   
    # getting center for plots:
    lat_center = graph_nodes.lat.mean()
    long_center =graph_nodes.lon.mean()
    # defining the layout using mapbox_style
    fig= go.FigureWidget(data=[line, points,source,end],
                    layout=go.Layout(
                    mapbox_style="open-street-map",
                    margin={"r":0,"t":0,"l":0,"b":0},
                    mapbox = {
                          'center': {'lat': lat_center, 
                          'lon': long_center},
                          'zoom':11}))
    
    # Declaring callback function for start station
    def point_update_start(a):
        # create origin point
        new_origin = create_point(get_entry(graph_nodes,textbox_src.value))
        # update start point
        with fig.batch_update():
            fig.data[2].lat =new_origin["lat"]
            fig.data[2].lon=new_origin["lon"]        
        
    # declare call back function for end station 
    def point_update_end(b):
        new_stop = create_point(get_entry(graph_nodes,textbox_dst.value))
        with fig.batch_update():
            fig.data[3].lat=new_stop["lat"]
            fig.data[3].lon=new_stop["lon"]
            
     #declare callback for path ploting        
    def response(c):
        if validate():
            with fig.batch_update() :
                # set status at runnging
                computation_state.value = " <font color='orange'><b>Running ...</b></font>"
                
                try : 
                    # check arrival time format 
                    time_callback()
                    
                    # copy graph to get all nodes
                    graph =  graph_strong_connected.copy()
                    
                    # compute the journey
                    found_routes, probs = final_journey(graph = graph, 
                                                        nodes = graph_nodes, 
                                                        start = textbox_src.value, 
                                                        stop = textbox_dst.value, 
                                                        time = string_datetime(str(textbox_time.value)),
                                                        confidence_thres = float(textbox_conf.value), 
                                                        number_of_routes = 5, 
                                                        top_k = 30)
                    
                    # Due to time constraints, consider only the first route
                    found_journey = found_routes[0]
                    prob = probs[0]
                    
                    # If journey is found, get plan and duration 
                    if(found_journey.shape[0]> 0):
                        
                        #update status
                        computation_state.value = "<font color='green'><b>TERMINATED !</b></font>"
                        
                        #update probability_widget
                        probability_widget.value = "<font size='1'><b>Probability</b></font>: <font color='black'><b> Your trip has a success rate of {probability}</b></font>"\
                                                            .format(probability =probs[0])

                        # update figure value
                        route_plan.value = found_journey.to_html()
                        route_plan.description = "<font size='3'><b>Satuts</b></font>:"
            
                        #update stations of path 
                        fig.data[0].lat = found_journey.lat
                        fig.data[0].lon = found_journey.lon
                        
                        # update centering
                        lat_center = found_journey.lat.mean()
                        long_center =found_journey.lon.mean()
                        
                        # set centering
                        fig.layout.mapbox.center.lat = lat_center
                        fig.layout.mapbox.center.lon = long_center
                        
                        print("ok")
                        
                except (KeyError,ValueError) : 
                    probability_widget.value = "<font color='red'><b>TERMINATED</b> (no stations found for your arrival time)</font>"
                    print('There was an Error : ', KeyError)
                    found_journey = pd.DataFrame()
                    
    # embed outer widgets
    accordion_box = widgets.HBox([button_apply,computation_state,probability_widget])
    
   
    # event listners for ipywidgets
    textbox_src.observe(point_update_start,names="value")
    textbox_dst.observe(point_update_end,names="value")
    button_apply.on_click(response)
    # wrap widgets and figure 
    tab.children  =  [widgets.VBox([accordion,accordion_box, fig]),route_plan,probability_widget]
    return tab
    

def create_point(coord) : 
    """
    point formating
    """
    return {'id' :coord[0] , "lat": [coord[2]], "lon":[coord[3]]}