import datetime
import pandas as pd
from heapq import heappush, heappop
from collections import deque

def string_datetime(time):
    """
    Convert a string to a datetime
    """

    return datetime.datetime.strptime(time, '%H:%M:%S').time()

def filter_max_arrival(graph, node, arrival_time):
    """
    Filter out edges of a node by finding latest arrival time

    After finding the possible connections, it iterates over them and find the latest arrival time.

    Then, it removes all of the edges that are not the latest arrival time one.
    """

    # Get all the incoming connections to node
    connections = set(graph.in_edges(node))

    # Iterate over the connections (pairs of nodes)
    for (src, dst) in connections:

        # Get all of the edges between the two nodes
        journeys = graph.get_edge_data(src, dst)

        # Find latest arrival time

        # Get all of the departure times of the edges that satisfy the arrival condition
        departure = [tj.get("departure_time") for tj in graph.get_edge_data(src, dst).values() \
                     if tj.get("arrival_time") <= arrival_time]

        # Iterable of the edges
        journeys_iterable = list(journeys.values())

        # If the departure list is empty, it means that no edges satisfy condition --> put max departure time
        if not departure:

            # Remove all of the edges
            [graph.remove_edge(src, dst, str(trip.get('arrival_time'))) for trip in journeys_iterable]

            # Move on to the next pair of nodes
            continue

        # If there are some edges that satisfy the condition
        else:

            # The latest departure time from the source node such that is satisfies the arrival condition
            latest_time = max(departure)

            # Iterate over the edges linking the two nodes
            for trip in journeys_iterable:

                # Remove the edges that are not the latest departure time
                if trip.get("departure_time") != latest_time:
                    graph.remove_edge(src, dst, str(trip.get('arrival_time')))

            # Update the latest arrival time to the predecessor node so that
            # it can communicate this information to its predecessors
            if (graph.nodes[src]["latest_arrival"] == datetime.time(0,0,0)):
                # update arrival time
                graph.nodes[src]["latest_arrival"] = latest_time

            else :
                # if value has already been modified in a path, keep the smaller one
                   if (graph.nodes[src]["latest_arrival"] > latest_time):
                        graph.nodes[src]["latest_arrival"] = latest_time


    return graph

def filter_max_departure(graph, node):
    """
    Filter out edges of a node by ensuring that departure times are after arrival times.

    After finding the possible connections, it iterates over the edges and remove

    the ones representing a departure happening after the latest arrivel.
    """

    # Get all the outgoing connections from the node
    connections = set(graph.out_edges(node))

    # Iterate through each pair of nodes
    for (src, dst) in connections:

        # Get all the edges of the corresponding pair of nodes
        journeys = graph.get_edge_data(src, dst)

        # Get the iterable of the edges
        journeys_iterable = list(journeys.values())

        # Iterate through the edges
        for trip in journeys_iterable:

            # Filter any trip that has a departure time from the src which is sooner than
            # the latest arrival time at the node
            if trip.get("departure_time") > graph.nodes[node]["latest_arrival"]:
                graph.remove_edge(src, dst, str(trip.get('arrival_time')))

    return graph


def backward_filtering(graph, target, arrival_time):
    """
    Personalized bfs that explore the graph starting from the target at a given arrival time.

    It filters the explored nodes by removing arrivals which are not the latest ones.
    """

    # Initializations
    graph.nodes[target]["latest_arrival"] = arrival_time
    visited_nodes = []
    pending_nodes = deque()
    visited_nodes.append(target)
    pending_nodes.append(target)

    # While there are still some nodes to explore
    while pending_nodes:

        # Explore the node
        node = pending_nodes.popleft()

        # Get his latest arrival time
        arrival_time = graph.nodes[node]["latest_arrival"]

        # Communicate the latest arrival time to its predecessors to filter out their edges
        graph = filter_max_arrival(graph, node, arrival_time)

        # Iterate through its predecessors
        for pred in graph.predecessors(node):

            # Queue them if not already visited
            if pred not in visited_nodes:
                visited_nodes.append(pred)
                pending_nodes.append(pred)

    return graph


def forward_filtering(graph, target):
    """
    Personalized bfs that explore the graph starting from target.

    It filters the explored nodes by removing departure happening before the arrival.
    """

    # Initializations
    visited_nodes = []
    pending_nodes = deque()
    visited_nodes.append(target)
    pending_nodes.append(target)

    # While there are still some nodes to explore
    while pending_nodes:

        # Explore the node
        node = pending_nodes.popleft()

        # Filter its successors
        graph = filter_max_departure(graph, node)

        # Iterate backwards from the target
        for pred in graph.predecessors(node):
            if pred not in visited_nodes:
                visited_nodes.append(pred)

    return graph


def time_difference(arrival, departure):
    """
    Computing time difference between 2 times
    """

    diff = datetime.datetime.combine(datetime.date.min, departure) \
           - datetime.datetime.combine(datetime.date.min, arrival)
    return diff.seconds / 60


def waiting_weighting(graph, node, incoming_arr_time):
    """
    Add the waiting time to the weight of edges - weight of edges are initially equal to the traveling time.

    This function adds the waiting time to the weight, which is equal by the arrival time from the last trip and the

    departure time of the said edge.
    """

    # Iterate through the outgoing connections of the input node
    for (src, dst) in graph.out_edges(node):

        # Get the connection information, of the first edge only (unique list)
        connection_spec = list(graph.get_edge_data(src, dst).values())[0]

        # Get the current weight of the edge
        weight_previous = float(graph[src][dst][str(connection_spec.get('arrival_time'))]["weight"])

        # Compute the new weights, by adding the previous value of the
        update = weight_previous + float(time_difference(incoming_arr_time, connection_spec.get("departure_time")))

        # Update the graph with the new value
        graph[src][dst][str(connection_spec.get('arrival_time'))]["weight"] = str(update)

def dijkstra(graph, start, end):
    """
    Dijkstra's algorithm for the Naive Journey Planner -

    computes the shortest path between a start and end nodes, assuming that there are no delays.

    Returns a list of ids of the nodes to follow.
    """

    # Instantiate the graph object holding the successors of each node
    G_succ = graph.succ

    # Special case
    if start == end:
        return {start: [start]}

    # Initializations
    routes = {start: [start]}
    push = heappush
    pop = heappop
    duration = {}  # Explored nodes along with their weights
    seen = {start: 0}  # Weight associated to nodes
    fringe = []
    push(fringe, (0, start, None))  # Format : travel time, present node, incoming arrival time

    # While there are some nodes to explore
    while fringe:

        # Pop the next node (incoming_arr_time = previous incoming_arr_time + travel_time)
        (travel_time, present_node, incoming_arr_time) = pop(fringe)

        # Check whether the current node has already been explored
        if present_node in duration:
            continue

        # Save the current node's travel time
        duration[present_node] = travel_time

        # If we have successfully searched up to the last node, then the shortest path computation is done
        if present_node == end:
            break

        # If not the source node
        if incoming_arr_time is not None:

            # Updating the weight of each of outgoing neighbors
            waiting_weighting(graph, present_node, incoming_arr_time)

            # Update the G_succ variable
            G_succ = graph.succ

        # Iterate through each successor - Choose the best successor
        for idx, e in G_succ[present_node].items():

            # Retrieve the weight to the best outgoing edge to the successor (unique list so [0] works)
            weight = min([float(specs.get('weight')) for specs in e.values()])

            # Get the arrival time of the successor
            arrival_time = list(G_succ[present_node][idx].values())[0]['arrival_time']

            # Current node's weight + new weight assigned to the successor
            traveled = duration[present_node] + weight

            # If one of the successors has already been explored and that the new weight is smaller,
            # it means that there is a contradiction
            if idx in duration:
                if traveled < duration[idx]:
                    raise ValueError('Contradictory paths found:',
                                     'negative waiting time ?')

            # Update the seen set
            elif idx not in seen or traveled < seen[idx]:

                # Update the weight
                seen[idx] = traveled

                # Push the successor into the nodes to be explored
                push(fringe, (traveled, idx, arrival_time))

                # Update the route to the successor
                routes[idx] = routes[present_node] + [idx]

    # Get the route to the end node
    route = routes[end]

    return route


def get_departure_time(graph, node, start):
    """
    Get the departure time
    """
    connection_spec = list(graph.get_edge_data(start, node).values())[0]
    return graph[start][node][str(connection_spec.get('arrival_time'))]['departure_time'].strftime("%H:%M:%S")

def convert_time(time) : 
    return time.hour*3600 + time.minute*60+time.second
        

def shortest_path(graph, start, end, arrival_time):
    """
    Computing shortest path between a starting and end point, given that we should arrive before arrival_time
    """

    # Filtering the graph
    filtered_graph_backward = backward_filtering(graph, end, arrival_time)

    # Computing the shortest path
    return dijkstra(filtered_graph_backward, start, end)

def get_times_and_mot(graph, route):
    """
    Retrieve the departure & arrival times & mean of transport, at each segment of the route
    """

    # Initializations
    departures = []
    arrivals = []
    
    mots = []

    # Iterate through each segment of the route
    for src, dst in zip(route, route[1:]):

        # Retrieve the data of the corresponding segment, save the departure & arrival times
        edge_data = list(graph.get_edge_data(src, dst).values())[0]
        departures.append(edge_data.get('departure_time').strftime("%H:%M:%S"))
        arrivals.append(edge_data.get('arrival_time').strftime("%H:%M:%S"))
        mots.append(edge_data.get('type'))

    # Quick length fix
    arrivals.insert(0, "")
    departures.append("")
    mots.append("Arrival")
    
    #compute duration 
    
    arrival = string_datetime(arrivals[-1])
    departure = string_datetime(departures[0])
    
        
    arrival_time= convert_time(arrival)
    departure_time= convert_time(departure)
    diff_time = arrival_time - departure_time
    
    hours = round(diff_time/3600)
    minutes = round((diff_time -hours*3600)/60)
    second = round(diff_time - hours*3600 -minutes*60)
    
    

    second= arrival.second-departure.second
     # Return the times
    return (departures, arrivals, mots,hours,minutes,second)


def get_entry(nodes, id_, name=False):
    """
    Get the id of the node, by id or name.
    """
    if not name:
        return nodes[nodes['stop_name'] == id_].head(1).values[0]
    else:
        return nodes[nodes['id'] == id_].head(1).values[0]


def final_journey(graph, nodes, start, stop, time):
    """
    Given the start & end nodes, and the arrival time, returns the paths and the schedule

    to reach the stop before the specified arrival time
    """

    # Initializations
    start_id = get_entry(nodes, start)[0]
    stop_id = get_entry(nodes, stop)[0]

    # Compute the shortest path between the start & end nodes, given the time
    route = shortest_path(graph, start_id, stop_id, time)

    # Get the information regarding each node on the path to display them
    names = [get_entry(nodes, stop, True)[1] for stop in route]
    lat = [get_entry(nodes, stop, True)[2] for stop in route]
    lon = [get_entry(nodes, stop, True)[3] for stop in route]

    # Get the departure & arrival times at each segment of the route
    (departures, arrivals, mots,hours,minutes,second) = get_times_and_mot(graph, route)

    # Return the results
    return (hours,minutes,second,pd.DataFrame(data={'stop': names, 'lat': lat, 'lon': lon, 'mean_of_transport': mots,
                              'arrivals': arrivals, 'departures': departures}))