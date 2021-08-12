import datetime
import pandas as pd
import random
from heapq import heappush, heappop
from collections import deque
from scipy.stats import gamma
from math import radians, cos, sin, asin, sqrt

class Path:
    """

    Class to model easily a path.

    """
    def __init__(self, nodes, last_weight, prob, last_g_d, trip_id, types, departure_times,
                 arrival_times, going_away):

        self.nodes = nodes
        self.last_weight = last_weight
        self.prob = prob
        self.last_g_d = last_g_d
        self.last_trip_id = trip_id
        self.types = types
        self.departure_times = departure_times
        self.arrival_times = arrival_times
        self.going_away = going_away

    def __str__(self):

        nodes = ', '.join([str(node) for node in self.nodes])
        types = ', '.join([str(t) for t in self.types])
        departure_times = ', '.join([str(dt) for dt in self.departure_times])
        arrival_times = ', '.join([str(at) for at in self.arrival_times])
        return 'Nodes : ' + nodes + '\n' + \
               'Path\'s weight : ' + str(self.last_weight) + '\n' + \
               'Path Probability : ' + str(self.prob) + '\n' + \
               'Path\'s last edge\'s gamma distribution :' + str(self.last_g_d) + '\n' + \
               'Path\'s types of transports :' + types + '\n' + \
               'Path\'s departure times : ' + departure_times + '\n' + \
               'Path\'s arrival times : ' + arrival_times


def dijkstra(graph, start, end, arrival_time, confidence_thres, number_of_routes, top_k, graph_nodes):

    """
    Dijkstra's algorithm for the Robust Journey Planner -

    computes the shortest path between a start and end nodes, assuming that there are no delays.

    Returns a list of ids of the nodes to follow, with its associated probability

    Vocabulary :
        - A node that has been explored is a node for which the outgoing edges have been evaluated
        - A node that has been discovered is a node for which only ingoing edges have been evaluated
            That node is to be explored afterwards.
    """

    # Instantiate the graph object holding the successors of each node
    G_succ = graph.succ

    # Special case
    if start == end:
        return {start: ([start], 1)}

    # Initializations
    end_lon = float(graph_nodes[graph_nodes['id'] == end]['lon'].head(1))
    end_lat = float(graph_nodes[graph_nodes['id'] == end]['lat'].head(1))
    push = heappush
    pop = heappop
    explored = set()
    fringe = []

    # We can consider that any trip should take less than 3 hours
    arrival_time = arrival_time.hour - 3
    arrival_time = string_datetime(str(arrival_time)+':00:00')

    # Define the start path
    start_path = Path([start], 0, 1, None, '-1', [], [], [arrival_time], False)

    # The start node's distance from the end node
    start_lon = float(graph_nodes[graph_nodes['id'] == start]['lon'].head(1))
    start_lat = float(graph_nodes[graph_nodes['id'] == start]['lat'].head(1))
    start_dist = haversine_distance(end_lon, end_lat, start_lon, start_lat)

    # Initializations of the lists
    node_to_paths = {start: [start_path]}
    node_to_min_weight = {start: 0}  # A node's minimal weight
    node_to_min_prob = {start: 1}  # Probability associated to the node's path with minimal weight
    node_to_dist = {start: start_dist}
    push(fringe, start)

    # While there are some nodes to explore
    while fringe:

        # Pop the next node (incoming_arr_time = previous incoming_arr_time + travel_time)
        present_node = pop(fringe)

        # Check whether the current node has already been explored
        # This should not be the case as we only push to the fringe nodes that haven't been explored
        if present_node in explored:

            raise ValueError("\nPresent node already explored\n")

        # Save the current node as explored
        explored.add(present_node)

        # Iterate through each successor node
        for successor_node, e in G_succ[present_node].items():

            # If we have already explored this node using this algorithm, then there is no use in considering it
            if successor_node in explored:
                continue

            # Compute the distance of the successor node from the target node
            try:
                succ_lon = float(graph_nodes[graph_nodes['id'] == successor_node]['lon'].head(1))
                succ_lat = float(graph_nodes[graph_nodes['id'] == successor_node]['lat'].head(1))
                succ_dist = haversine_distance(end_lon, end_lat, succ_lon, succ_lat)

            # In case the position of the successor node cannot be found
            except:
                # print("Exception - the successor node's localisation cannot be found - id : ", successor_node)
                succ_dist = node_to_dist[present_node]

            # Update the distance dictionary
            node_to_dist[successor_node] = succ_dist

            # Check whether we are going away from the destination
            succ_going_away = succ_dist < node_to_dist[present_node]

            # Initialize a boolean for pruning
            new_succ = True

            # Pruning - In case we already have considered edges to that successor from another node
            if successor_node in node_to_paths:
                new_succ = False

            # Retrieve the edges between the present node and the successor node
            edges = graph.get_edge_data(present_node, successor_node)
            edges_list = list(edges.values())

            # Iterate through each edge between the present node and the successor node
            for edge in edges_list:

                # Iterate through each of the paths in the present node
                for path in node_to_paths[present_node]:

                    # Prune if the path is going away from the destination node twice
                    # if path.going_away & succ_going_away:
                        # print("no direction pruning")
                        # continue

                    # Do not consider incoherent edges
                    if path.arrival_times[-1] > edge.get('departure_time'):
                        # print('incoherent edges : ', path.arrival_times[-1], 'vs', edge.get('departure_time'))
                        continue

                    # Compute the extra time between the path's arrival time and the edge's departure time - Case Start
                    if present_node == start:
                        extra_time = 0.0

                    # Case other nodes than start
                    else:

                        # Prune if two consecutive transfers
                        if (path.types[-1] == 'transfer') & (edge.get('type') == 'transfer'):
                            continue

                        # Compute waiting time
                        extra_time = float(time_difference(path.arrival_times[-1], edge.get("departure_time")))

                    # Case source node or transfer
                    if path.last_g_d is None:
                        success_prob = 1

                    # Case risk of missing connection
                    else:

                        # If we are changing the mean of transport, compute the risk of missing the connection
                        if edge.get("trip_id") != path.last_trip_id:
                            success_prob = gamma.cdf(extra_time, a=path.last_g_d['fit_alpha'],
                                                     loc=path.last_g_d['fit_loc'],
                                                     scale=path.last_g_d['fit_scale'])

                        # If we are staying in the same mean of transport, the probability is equal to 1
                        else:
                            success_prob = 1

                    # Compute the path's new probability
                    new_prob = path.prob * success_prob

                    # Prune if the path has a lower probability than the specified confidence threshold
                    if new_prob < confidence_thres:
                        continue

                    # Compute the new weight = current weight + waiting time + travel time
                    new_weight = float(path.last_weight) + extra_time + float(edge.get('weight'))

                    # If we are on the start node, penalize the departure stall
                    if present_node == start:
                        new_weight += time_difference(graph.nodes[start]["latest_arrival"], edge.get('departure_time'))

                    # Retrieve the edge's gamma distribution - Case not defined
                    if edge.get('fit_alpha') is None:
                        gamma_distribution = None

                    # Case distribution defined
                    else:
                        gamma_distribution = {'fit_alpha': float(edge.get('fit_alpha')),
                                              'fit_loc': float(edge.get('fit_loc')),
                                              'fit_scale': float(edge.get('fit_scale'))}

                    # Instantiate a new path for the successor node
                    new_path = Path(nodes=path.nodes + [successor_node],
                                    last_weight=new_weight,
                                    prob=new_prob,
                                    last_g_d=gamma_distribution,
                                    trip_id=edge.get("trip_id"),
                                    types=path.types + [edge.get('type')],
                                    departure_times=path.departure_times + [edge.get('departure_time')],
                                    arrival_times=path.arrival_times + [edge.get('arrival_time')],
                                    going_away=succ_going_away)

                    # If the successor node has been already discovered by another node, then apply the pruning strategy
                    if not new_succ:

                        # Prune the path if its weight is bigger than the node's minimal weight & probability smaller
                        if ((new_weight < node_to_min_weight[successor_node]) &
                                (new_prob > node_to_min_prob[successor_node])):
                            continue

                        # If we're not pruning, we are adding it amongst the path of the successor node
                        else:
                            node_to_paths[successor_node].append(new_path)

                    # If the successor node has not been already discovered by another node before
                    else:

                        # In case the successor node hasn't been added to the dictionary yet
                        if not successor_node in node_to_paths:
                            node_to_paths[successor_node] = [new_path]

                        # Otherwise, append it
                        else:
                            node_to_paths[successor_node].append(new_path)

            # In case we didn't build any path to the successor node (happens when none of the edge can be taken
            # due to the arrival time being later than any departure time to attain the successor node)
            if successor_node not in node_to_paths:
                continue

            # Keep the smallest k/2 weights
            node_to_paths[successor_node].sort(key=lambda x: x.last_weight, reverse=False)
            paths_to_keep = node_to_paths[successor_node][:top_k]

            # Add some random paths in case the paths are skewed to the same arrival time
            if len(node_to_paths[successor_node][top_k:]) > int(top_k/2):
                paths_to_keep += random.sample(node_to_paths[successor_node][top_k:], int(top_k/2))

            # Prune the paths to the new node
            node_to_paths[successor_node] = paths_to_keep

            # If the node is the end node, return all of its paths
            if successor_node == end:
                node_to_paths[successor_node].sort(key=lambda x: x.prob, reverse=True)
                return node_to_paths[successor_node][:number_of_routes]

            # Compute the minimal weight & associated probability among all the new paths - Initializations
            min_weight = node_to_paths[successor_node][0].last_weight
            min_prob = node_to_paths[successor_node][0].prob
            for p in node_to_paths[successor_node]:
                if p.last_weight < min_weight:
                    min_weight = min_weight
                    min_prob = p.prob
            node_to_min_weight[successor_node] = min_weight
            node_to_min_prob[successor_node] = min_prob

        # Finished exploring all the present node's neighbors - can free the paths of the present node stored in a dict
        node_to_paths.pop(present_node)
        node_to_min_weight.pop(present_node)
        node_to_min_prob.pop(present_node)

        # Exception when there are no more nodes to explore - by definition, any node to explore is stored in this list
        if len(node_to_min_weight) == 0:
            raise ValueError("\nNode to Min Weight is empty\n")

        # Compute the minimal minimal weights among all the discovered nodes - Initializations
        min_min_weight = 86400  # seconds in a day
        next_node = start

        # Iterate through the discovered nodes, choose the one with the lowest minimal weight
        for key, value in node_to_min_weight.items():

            # Consider only nodes that haven't been explored
            # (should not be the case since we pop every node explored from node_to_min_weight)
            if key in explored:
                continue

            # Keep the node having the smallest minimal weight amongst all the nodes
            if value < min_min_weight:
                next_node = key

        # Push the chosen node into the nodes to be explored
        push(fringe, next_node)

    # Return the routes to the end node
    print("Route found !")
    
    # Sorting by probability
    node_to_paths[end].sort(key=lambda x: x.prob, reverse=True)
    
    return node_to_paths[end][:number_of_routes]

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def filter_max_arrival(graph, node, arrival_time):
    """
    Filter out edges of a node by finding latest arrival time

    After finding the possible connections, it iterates over them and find the latest arrival time.

    Differences with the function from V2 : Does not remove all of the edges that are not the latest arrival time one.
    Instead, it filters by keeping only the edges that depart before the latest departure time
    """

    # Get all the incoming connections to node
    connections = set(graph.in_edges(node))

    # Iterate over the connections (pairs of nodes)
    for (src, dst) in connections:

        # Get all of the edges between the two nodes
        journeys = graph.get_edge_data(src, dst)

        # Get all of the departure times of the edges that satisfy the arrival condition
        departure = [tj.get("departure_time") for tj in graph.get_edge_data(src, dst).values() \
                     if tj.get("arrival_time") <= arrival_time]

        # Iterable of the edges
        journeys_iterable = list(journeys.values())

        # If the departure list is empty, it means that no edges satisfy condition
        if not departure:

            # Remove all of the edges
            [graph.remove_edge(src, dst, str(trip.get('arrival_time'))) for trip in journeys_iterable]

            # Move on to the next pair of nodes
            continue

        # If there are some edges that satisfy the condition
        else:

            # Sort the list
            departure = sorted(departure, reverse=True)
            two_latest_times = departure[:2]

            # The latest departure time from the source node such that is satisfies the arrival condition
            latest_time = max(departure)

            # Iterate over the edges linking the two nodes
            for trip in journeys_iterable:

                # Keep only the edges that depart before the latest departure time
                if (trip.get("departure_time") > latest_time) and (trip.get("departure_time") not in two_latest_times):
                    graph.remove_edge(src, dst, str(trip.get('arrival_time')))

            # Update the latest arrival time to the predecessor node so that
            # it can communicate this information to its predecessors
            graph.nodes[src]["latest_arrival"] = latest_time

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

    # Return the times
    return (departures, arrivals, mots)


def shortest_path(graph, start, end, arrival_time, confidence_thres, number_of_routes, top_k, graph_nodes):
    """
    Computing shortest path between a starting and end point, given that we should arrive before arrival_time
    """

    # Filtering the graph
    filtered_graph_backward = backward_filtering(graph, end, arrival_time)

    # Computing the shortest path
    return dijkstra(filtered_graph_backward, start, end, arrival_time, confidence_thres, number_of_routes, top_k, graph_nodes)


def final_journey(graph, nodes, start, stop, time, confidence_thres, number_of_routes, top_k):
    """
    Given the start & end nodes, and the arrival time, returns the paths and the schedule

    to reach the stop before the specified arrival time
    """

    # Initializations
    start_id = get_entry(nodes, start)[0]
    stop_id = get_entry(nodes, stop)[0]

    # Compute the shortest path between the start & end nodes, given the time
    routes = shortest_path(graph, start_id, stop_id, time, confidence_thres, number_of_routes, top_k, nodes)
    
    # Result accumulator
    routes_pd = []
    route_probs = []
    
    # Get the informations regarding each node on the path to display them
    for r in routes :
        
        # Retrieve all the Path class information
        r_d_t = r.departure_times
        r_a_t = r.arrival_times[1:]
        r_m_t = r.types
        r_prob = r.prob
        r_nodes = r.nodes
        r_names = []
        r_lats = []
        r_lons = []
        
        # Retrieve the names, lats & lons
        for node in r_nodes :
            
            # Accumulate the info
            r_names.append(str(list(nodes[nodes['id'] == node]['stop_name'])[0]))
            r_lats.append(str(list(nodes[nodes['id'] == node]['lat'])[0]))
            r_lons.append(str(list(nodes[nodes['id'] == node]['lon'])[0]))
                
        # Accumulate the results
        routes_pd.append(pd.DataFrame(data={'stop': r_names, 'lat': r_lats, 'lon': r_lons, 
                                            'mean_of_transport': [None] + r_m_t,
                                            'departures': r_d_t+[None],
                                            'arrivals': [None] + r_a_t}))
        route_probs.append(r_prob)
        
    # Return the results
    return routes_pd, route_probs

def string_datetime(time):
    """
    Convert a string to a datetime
    """

    return datetime.datetime.strptime(time, '%H:%M:%S').time()


def time_difference(arrival, departure):
    """
    Computing time difference between 2 times
    """

    diff = datetime.datetime.combine(datetime.date.min, departure) \
           - datetime.datetime.combine(datetime.date.min, arrival)
    return diff.seconds


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


def get_entry(nodes, id_, name=False):
    """
    Get the id of the node, by id or name.
    """
    if not name:
        return nodes[nodes['stop_name'] == id_].head(1).values[0]
    else:
        return nodes[nodes['id'] == id_].head(1).values[0]