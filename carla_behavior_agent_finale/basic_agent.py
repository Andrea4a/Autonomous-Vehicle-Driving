# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights.
It can also make use of the global route planner to follow a specifed route
"""

import carla
from shapely.geometry import Polygon

from local_planner import LocalPlanner, RoadOption
from global_route_planner import GlobalRoutePlanner
from misc import (get_speed, is_within_distance,
                               get_trafficlight_trigger_location,
                               compute_distance)
# from perception.perfectTracker.gt_tracker import PerfectTracker

debug = True

class BasicAgent(object):
    """
    BasicAgent implements an agent that navigates the scene.
    This agent respects traffic lights and other vehicles, but ignores stop signs.
    It has several functions available to specify the route that the agent must follow,
    as well as to change its parameters in case a different driving mode is desired.
    """

    def __init__(self, vehicle, opt_dict={}, map_inst=None, grp_inst=None):
        """
        Initialization the agent paramters, the local and the global planner.

            :param vehicle: actor to apply to agent logic onto
            :param opt_dict: dictionary in case some of its parameters want to be changed.
                This also applies to parameters related to the LocalPlanner.
            :param map_inst: carla.Map instance to avoid the expensive call of getting it.
            :param grp_inst: GlobalRoutePlanner instance to avoid the expensive call of getting it.

        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        if map_inst:
            if isinstance(map_inst, carla.Map):
                self._map = map_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._map = self._world.get_map()
        else:
            self._map = self._world.get_map()
        self._last_traffic_light = None
        self._last_stop_sign = None

        # Base parameters
        self._ignore_traffic_lights = False
        self._ignore_stop_signs = False
        self._ignore_vehicles = False
        self._ignore_static_obstacle = False
        self._use_bbs_detection = False
        self._target_speed = 5.0
        self._sampling_resolution = 2.0
        self._base_tlight_threshold = 5.0  
        self._base_vehicle_threshold = 20.0  
        self._base_static_obstacle_threshold = 50 
        self._speed_ratio = 1
        self._max_brake = 0.5
        self._offset = 0

        # Change parameters according to the dictionary
        if 'target_speed' in opt_dict:
            self._target_speed = opt_dict['target_speed']
        if 'ignore_traffic_lights' in opt_dict:
            self._ignore_traffic_lights = opt_dict['ignore_traffic_lights']
        if 'ignore_stop_signs' in opt_dict:
            self._ignore_stop_signs = opt_dict['ignore_stop_signs']
        if 'ignore_vehicles' in opt_dict:
            self._ignore_vehicles = opt_dict['ignore_vehicles']
        if 'use_bbs_detection' in opt_dict:
            self._use_bbs_detection = opt_dict['use_bbs_detection']
        if 'sampling_resolution' in opt_dict:
            self._sampling_resolution = opt_dict['sampling_resolution']
        if 'base_tlight_threshold' in opt_dict:
            self._base_tlight_threshold = opt_dict['base_tlight_threshold']
        if 'base_vehicle_threshold' in opt_dict:
            self._base_vehicle_threshold = opt_dict['base_vehicle_threshold']
        if 'detection_speed_ratio' in opt_dict:
            self._speed_ratio = opt_dict['detection_speed_ratio']
        if 'max_brake' in opt_dict:
            self._max_brake = opt_dict['max_brake']
        if 'offset' in opt_dict:
            self._offset = opt_dict['offset']
        
        if debug:
            print("Max brake:", self._max_brake)
            print("Ignore vehicles:", self._ignore_vehicles)

        # Initialize the planners
        self._local_planner = LocalPlanner(self._vehicle, opt_dict=opt_dict, map_inst=self._map)
        if grp_inst:
            if isinstance(grp_inst, GlobalRoutePlanner):
                self._global_planner = grp_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)
        else:
            self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)

        # Get the static elements of the scene
        self._lights_list = self._world.get_actors().filter("*traffic_light*")
        self._lights_map = {}  # Dictionary mapping a traffic light to a wp corresponding to its trigger volume location

    def add_emergency_stop(self, control):
        """
        Overwrites the throttle and brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns.

            :param control (carl.VehicleControl): control to be modified
        """
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def set_target_speed(self, speed):
        """
        Changes the target speed of the agent to the speed passed as argument.

            :param speed (float): target speed in Km/h
        """
        self._target_speed = speed
        self._local_planner.set_speed(speed)

    def follow_speed_limits(self, value=True):
        """
        If active, the agent will dynamically change the target speed according to the speed limits.

            :param value (bool): whether or not to activate this behavior
        """
        self._local_planner.follow_speed_limits(value)

    def get_local_planner(self):
        """
        Get method for protected member local planner.
        """
        return self._local_planner

    def get_global_planner(self):
        """
        Get method for protected member local planner.
        """
        return self._global_planner

    def set_destination(self, end_location, start_location=None):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location
            clean_queue = True
        else:
            start_location = self._vehicle.get_location()
            clean_queue = False

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)

    def set_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a specific plan to the agent.

            :param plan: list of [carla.Waypoint, RoadOption] representing the route to be followed
            :param stop_waypoint_creation: stops the automatic random creation of waypoints
            :param clean_queue: resets the current agent's plan
        """
        self._local_planner.set_global_plan(
            plan,
            stop_waypoint_creation=stop_waypoint_creation,
            clean_queue=clean_queue
        )

    def trace_route(self, start_waypoint, end_waypoint):
        """
        Calculates the shortest route between a starting and ending waypoint.

            :param start_waypoint (carla.Waypoint): initial waypoint
            :param end_waypoint (carla.Waypoint): final waypoint
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def run_step(self):
        """Execute one step of navigation.
        hazard_detected = False 
        retrieving all vehicle actors
         vehicle_list = self._world.get_actors().filter("*vehicle*") 
         vehicle_speed = get_speed(self._vehicle) / 3.6 
        # Check for possible vehicle obstacles
       max_vehicle_distance = self._base_vehicle_threshold + self._speed_ratio * vehicle_speed
       affected_by_vehicle, _, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            hazard_detected = True 
        if debug:
            print("Harzard detected:", hazard_detected) 
         # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = self._base_tlight_threshold + self._speed_ratio * vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(self._lights_list, max_tlight_distance)
        if affected_by_tlight:
            hazard_detected = True 
        control = self._local_planner.run_step()
        if hazard_detected:
            control = self.add_emergency_stop(control) 
        return control
        """

    def reset(self):
        pass

    def done(self):
        """
        Check whether the agent has reached its destination.
        """
        return self._local_planner.done()

    def ignore_traffic_lights(self, active=True):
        """
        (De)activates the checks for traffic lights.
        """
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active=True):
        """
        (De)activates the checks for stop signs.
        """
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active=True):
        """
        (De)activates the checks for vehicle.
        """
        self._ignore_vehicles = active

    def ignore_static_obstacle(self, active=True):
        """
        (De)activates the checks for static obstacles.
        """
        self._ignore_static_obstacle = active

    def lane_change(self, direction, same_lane_time=0, other_lane_time=0, lane_change_time=2):
        """
        Changes the path so that the vehicle performs a lane change.
        Use 'direction' to specify either a 'left' or 'right' lane change,
        and the other 3 fine tune the maneuver.
        """
        speed = self._vehicle.get_velocity().length()
        path = self._generate_lane_change_path(
            self._map.get_waypoint(self._vehicle.get_location()),
            direction,
            same_lane_time * speed,
            other_lane_time * speed,
            lane_change_time * speed,
            False,
            1,
            self._sampling_resolution
        )
        if not path:
            print("WARNING: Ignoring the lane change as no path was found")

        self.set_global_plan(path)

    # Metodo per il controllo della presenza di un segnale di stop che influisce sul veicolo.
    def _affected_by_stop_sign(self, stop_sign_list=None, max_distance=None):
        if self._ignore_stop_signs:
            return False, None, -1

        # Usa tutti i segnali di stop nella scena se non è stata fornita una lista
        if stop_sign_list is None:
            stop_sign_list = self._world.get_actors().filter("*stop*")

        # Usa la soglia di distanza predefinita se non viene fornita una distanza massima
        if max_distance is None:
            max_distance = self._base_tlight_threshold

        ego_loc = self._vehicle.get_location()
        ego_wp = self._map.get_waypoint(ego_loc)

        for stop_sign in stop_sign_list:
            # Recupera il waypoint del segnale di stop (da cache o da trigger)
            if stop_sign.id in self._lights_map:
                stop_wp = self._lights_map[stop_sign.id]
            else:
                trigger_loc = get_trafficlight_trigger_location(stop_sign)
                stop_wp = self._map.get_waypoint(trigger_loc)
                self._lights_map[stop_sign.id] = stop_wp

            # Ignora se troppo lontano
            if stop_wp.transform.location.distance(ego_loc) > max_distance:
                continue

            # Ignora se su una strada diversa
            if stop_wp.road_id != ego_wp.road_id:
                continue

            # Verifica che il segnale sia nella stessa direzione del veicolo
            ego_dir = ego_wp.transform.get_forward_vector()
            stop_dir = stop_wp.transform.get_forward_vector()
            direction_dot = ego_dir.x * stop_dir.x + ego_dir.y * stop_dir.y + ego_dir.z * stop_dir.z

            if direction_dot < 0:
                continue

            # Verifica che il veicolo sia entro la distanza e angolo dal segnale
            if is_within_distance(stop_wp.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
                self._last_stop_sign = stop_sign
                dist = compute_distance(self._vehicle.get_transform().location, stop_wp.transform.location)
                return True, stop_sign, dist

        return False, None, -1


    def _affected_by_traffic_light(self, lights_list=None, max_distance=None):
        """
        Method to check if there is a red light affecting the vehicle.

            :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
                If None, all traffic lights in the scene are used
            :param max_distance (float): max distance for traffic lights to be considered relevant.
                If None, the base threshold value is used
        """
        if self._ignore_traffic_lights:
            return (False, None)

        if not lights_list:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        if not max_distance:
            max_distance = self._base_tlight_threshold

        if self._last_traffic_light:
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                return (True, self._last_traffic_light)

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            if traffic_light.id in self._lights_map:
                trigger_wp = self._lights_map[traffic_light.id]
            else:
                trigger_location = get_trafficlight_trigger_location(traffic_light)
                trigger_wp = self._map.get_waypoint(trigger_location)
                self._lights_map[traffic_light.id] = trigger_wp

            if trigger_wp.transform.location.distance(ego_vehicle_location) > max_distance:
                continue

            if trigger_wp.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = trigger_wp.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            if is_within_distance(trigger_wp.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
                self._last_traffic_light = traffic_light
                return (True, traffic_light)

        return (False, None)

    
    # Metodo per il controllo della presenza di un ostacolo statico che influisce sul veicolo.
    def _static_obstacle_detected(self, static_obstacle_list=None, max_distance=None):
        # Se configurato per ignorare ostacoli statici, ritorna un placeholder negativo
        if self._ignore_static_obstacle:
            return [(False, None, -1)]

        # Recupera tutti gli ostacoli statici dalla scena se non specificati
        if static_obstacle_list is None:
            static_obstacle_list = self._world.get_actors().filter("*static.prop*")

        if max_distance is None:
            max_distance = self._base_static_obstacle_threshold

        ego_loc = self._vehicle.get_location()
        ego_wp = self._map.get_waypoint(ego_loc)

        detected_obstacles = []

        for obs in static_obstacle_list:
            obs_tf = obs.get_transform()
            obs_wp = self._map.get_waypoint(obs_tf.location, lane_type=carla.LaneType.Any)

            # Ignora ostacoli troppo lontani
            if obs_wp.transform.location.distance(ego_loc) > max_distance:
                continue

            # Ignora ostacoli su strada o corsia differente
            if obs_wp.road_id != ego_wp.road_id or obs_wp.lane_id != ego_wp.lane_id:
                continue

            ego_dir = ego_wp.transform.get_forward_vector()
            obs_dir = obs_wp.transform.get_forward_vector()
            alignment = ego_dir.x * obs_dir.x + ego_dir.y * obs_dir.y + ego_dir.z * obs_dir.z

            # Ignora ostacoli non allineati con la direzione del veicolo
            if alignment < 0:
                continue

            ego_tf = self._vehicle.get_transform()

            # Se l'ostacolo è rilevante (vicino e nella direzione del veicolo)
            if is_within_distance(obs_wp.transform, ego_tf, max_distance, [0, 30]):
                distance = compute_distance(obs_tf.location, ego_tf.location)
                detected_obstacles.append((True, obs, distance))

        if detected_obstacles:
            return sorted(detected_obstacles, key=lambda x: x[2])
        else:
            return [(False, None, -1)]

        
    # Metodo per il controllo della presenza di un pedone che influisce sul veicolo.    
    def _walker_obstacle_detected(self, walker_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        # Recupera tutti i pedoni nella scena se non specificato
        if walker_list is None:
            walker_list = self._world.get_actors().filter("*walker.pedestrian*")

        if max_distance is None:
            max_distance = self._base_vehicle_threshold

        ego_tf = self._vehicle.get_transform()
        ego_wp = self._map.get_waypoint(self._vehicle.get_location())

        if ego_wp.lane_id < 0 and lane_offset != 0:
            lane_offset = -lane_offset

        ego_dir = ego_tf.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_tf = ego_tf
        ego_front_tf.location += carla.Location(
            x=ego_extent * ego_dir.x,
            y=ego_extent * ego_dir.y,
        )

        detected_walkers = []

        for walker in walker_list:
            same_lane = True
            walker_tf = walker.get_transform()
            walker_wp = self._map.get_waypoint(walker_tf.location, lane_type=carla.LaneType.Any)

            # Controlla se almeno un vertice della bounding box del pedone è sulla corsia target
            walker_bb = walker.bounding_box.get_world_vertices(walker_tf)
            in_same_lane = False
            for point in walker_bb:
                if self._map.get_waypoint(point, lane_type=carla.LaneType.Any).lane_id == (ego_wp.lane_id + lane_offset):
                    in_same_lane = True
                    break

            # Se almeno uno dei due non è in una junction, verifica anche la strada e la corsia
            if not ego_wp.is_junction or not walker_wp.is_junction:
                next_wp = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                if walker_wp.road_id != ego_wp.road_id or \
                   (walker_wp.lane_id != ego_wp.lane_id + lane_offset and walker_wp.lane_id != ego_wp.get_right_lane().lane_id + lane_offset):
                    if not next_wp:
                        continue
                    if walker_wp.road_id != next_wp.road_id or \
                       (walker_wp.lane_id != next_wp.lane_id + lane_offset and walker_wp.lane_id != next_wp.get_right_lane().lane_id + lane_offset):
                        continue

                walker_dir = walker_tf.get_forward_vector()
                walker_extent = walker.bounding_box.extent.x
                walker_rear_tf = walker_tf
                walker_rear_tf.location -= carla.Location(
                    x=walker_extent * walker_dir.x,
                    y=walker_extent * walker_dir.y,
                )

                if is_within_distance(walker_rear_tf, ego_front_tf, max_distance, [low_angle_th, up_angle_th]):
                    if walker_wp.lane_id != next_wp.lane_id and not in_same_lane:
                        same_lane = False

                    distance = compute_distance(walker_tf.location, ego_tf.location)
                    detected_walkers.append((True, walker, distance, same_lane))

            else:
                # Fall-back: verifica la prossimità alla traiettoria in mancanza di waypoint affidabili
                ego_loc = ego_tf.location
                extent_y = self._vehicle.bounding_box.extent.y
                r_vec = ego_tf.get_right_vector()

                bb_poly = []
                p1 = ego_loc + carla.Location(x=extent_y * r_vec.x, y=extent_y * r_vec.y)
                p2 = ego_loc + carla.Location(x=-extent_y * r_vec.x, y=-extent_y * r_vec.y)
                bb_poly.append([p1.x, p1.y, p1.z])
                bb_poly.append([p2.x, p2.y, p2.z])

                for wp, _ in self._local_planner.get_plan():
                    if ego_loc.distance(wp.transform.location) > max_distance:
                        break
                    r_vec = wp.transform.get_right_vector()
                    p1 = wp.transform.location + carla.Location(x=extent_y * r_vec.x, y=extent_y * r_vec.y)
                    p2 = wp.transform.location + carla.Location(x=-extent_y * r_vec.x, y=-extent_y * r_vec.y)
                    bb_poly.append([p1.x, p1.y, p1.z])
                    bb_poly.append([p2.x, p2.y, p2.z])

                if len(bb_poly) < 3:
                    continue

                route_polygon = Polygon(bb_poly)

                if walker.id == self._vehicle.id:
                    continue
                if ego_loc.distance(walker.get_location()) > max_distance:
                    continue

                walker_bb = walker.bounding_box.get_world_vertices(walker.get_transform())
                walker_poly = Polygon([[v.x, v.y, v.z] for v in walker_bb])

                if route_polygon.intersects(walker_poly):
                    distance = compute_distance(walker.get_location(), ego_loc)
                    detected_walkers.append((True, walker, distance, False))

        if detected_walkers:
            return sorted(detected_walkers, key=lambda x: x[2])
        else:
            return [(False, None, -1, None)]


    # Metodo per il controllo della presenza di un ciclista 
    def _biker_obstacle_detected(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        if self._ignore_vehicles:
            return [(False, None, -1, None)]
    
        if vehicle_list is None:
            vehicle_list = self._world.get_actors().filter("*vehicle*")
    
        if max_distance is None:
            max_distance = self._base_vehicle_threshold
    
        ego_tf = self._vehicle.get_transform()
        ego_wp = self._map.get_waypoint(self._vehicle.get_location())
    
        if ego_wp.lane_id < 0 and lane_offset != 0:
            lane_offset = -lane_offset
    
        ego_dir = ego_tf.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_tf = ego_tf
        ego_front_tf.location += carla.Location(
            x=ego_extent * ego_dir.x,
            y=ego_extent * ego_dir.y,
        )
    
        detected_vehicles = []
    
        for veh in vehicle_list:
            same_lane = True
            veh_tf = veh.get_transform()
            veh_wp = self._map.get_waypoint(veh_tf.location, lane_type=carla.LaneType.Any)
    
            # Verifica se almeno un vertice della bounding box si trova sulla corsia corretta
            veh_bb = veh.bounding_box.get_world_vertices(veh_tf)
            on_target_lane = False
            for pt in veh_bb:
                if self._map.get_waypoint(pt, lane_type=carla.LaneType.Any).lane_id == (ego_wp.lane_id + lane_offset):
                    on_target_lane = True
                    break
    
            if not ego_wp.is_junction or not veh_wp.is_junction:
                next_wp = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                if veh_wp.road_id != ego_wp.road_id or \
                   (veh_wp.lane_id != ego_wp.lane_id + lane_offset and veh_wp.lane_id != ego_wp.get_right_lane().lane_id + lane_offset):
                    if not next_wp:
                        continue
                    if veh_wp.road_id != next_wp.road_id or \
                       (veh_wp.lane_id != next_wp.lane_id + lane_offset and veh_wp.lane_id != next_wp.get_right_lane().lane_id + lane_offset):
                        continue
    
                veh_dir = veh_tf.get_forward_vector()
                veh_extent = veh.bounding_box.extent.x
                veh_rear_tf = veh_tf
                veh_rear_tf.location -= carla.Location(
                    x=veh_extent * veh_dir.x,
                    y=veh_extent * veh_dir.y,
                )
    
                if is_within_distance(veh_rear_tf, ego_front_tf, max_distance, [low_angle_th, up_angle_th]):
                    if veh_wp.lane_id != next_wp.lane_id and not on_target_lane:
                        same_lane = False
    
                    distance = compute_distance(veh_tf.location, ego_tf.location)
                    detected_vehicles.append((True, veh, distance, same_lane))
    
            else:
                # Fallback geometrico per intersezioni complesse: costruzione del poligono percorso
                ego_loc = ego_tf.location
                extent_y = self._vehicle.bounding_box.extent.y
                r_vec = ego_tf.get_right_vector()
    
                path_bb = []
                p1 = ego_loc + carla.Location(x=extent_y * r_vec.x, y=extent_y * r_vec.y)
                p2 = ego_loc + carla.Location(x=-extent_y * r_vec.x, y=-extent_y * r_vec.y)
                path_bb.append([p1.x, p1.y, p1.z])
                path_bb.append([p2.x, p2.y, p2.z])
    
                for wp, _ in self._local_planner.get_plan():
                    if ego_loc.distance(wp.transform.location) > max_distance:
                        break
                    r_vec = wp.transform.get_right_vector()
                    p1 = wp.transform.location + carla.Location(x=extent_y * r_vec.x, y=extent_y * r_vec.y)
                    p2 = wp.transform.location + carla.Location(x=-extent_y * r_vec.x, y=-extent_y * r_vec.y)
                    path_bb.append([p1.x, p1.y, p1.z])
                    path_bb.append([p2.x, p2.y, p2.z])
    
                if len(path_bb) < 3:
                    continue
    
                ego_poly = Polygon(path_bb)
    
                if veh.id == self._vehicle.id:
                    continue
                if ego_loc.distance(veh.get_location()) > max_distance:
                    continue
    
                veh_vertices = veh.bounding_box.get_world_vertices(veh.get_transform())
                veh_poly = Polygon([[v.x, v.y, v.z] for v in veh_vertices])
    
                if ego_poly.intersects(veh_poly):
                    dist = compute_distance(veh.get_location(), ego_loc)
                    detected_vehicles.append((True, veh, dist, False))
    
        if detected_vehicles:
            return sorted(detected_vehicles, key=lambda x: x[2])
        else:
            return [(False, None, -1, None)]



    def _vehicle_obstacle_detected(self, vehicle_list=None, max_distance=None, up_angle_th=150, low_angle_th=0, lane_offset=0):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used

            
            return: sorted list contained all vehicles or a list with a single element that indicates that is not a vehicles
        """
        if self._ignore_vehicles:
            return [(False, None, -1)]

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )

        temp_vehicle_list = []

        for target_vehicle in vehicle_list:
            target_transform = target_vehicle.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # Simplified version for outside junctions
            if not ego_wpt.is_junction and not target_wpt.is_junction:

                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id  + lane_offset:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue

                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id  + lane_offset:
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                )

                if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [low_angle_th, up_angle_th]):
                    temp_vehicle_list.append((True, target_vehicle, compute_distance(target_transform.location, ego_transform.location)))


            # Waypoints aren't reliable, check the proximity of the vehicle to the route
            else:
                route_bb = []
                ego_location = ego_transform.location
                extent_y = self._vehicle.bounding_box.extent.y
                r_vec = ego_transform.get_right_vector()
                p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                route_bb.append([p1.x, p1.y, p1.z])
                route_bb.append([p2.x, p2.y, p2.z])

                for wp, _ in self._local_planner.get_plan():
                    if ego_location.distance(wp.transform.location) > max_distance:
                        break

                    r_vec = wp.transform.get_right_vector()
                    p1 = wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                    p2 = wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                    route_bb.append([p1.x, p1.y, p1.z])
                    route_bb.append([p2.x, p2.y, p2.z])

                if len(route_bb) < 3:
                    # 2 points don't create a polygon, nothing to check
                    continue
                ego_polygon = Polygon(route_bb)

                # Compare the two polygons
                target_extent = target_vehicle.bounding_box.extent.x
                if target_vehicle.id == self._vehicle.id:
                    continue
                if ego_location.distance(target_vehicle.get_location()) > max_distance:
                    continue

                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if ego_polygon.intersects(target_polygon):
                    temp_vehicle_list.append((True, target_vehicle, compute_distance(target_transform.location, ego_location)))

        if len(temp_vehicle_list) > 0:
            return sorted(temp_vehicle_list, key=lambda x: x[2])
        else:
            return [(False, None, -1)]
        
    # Metodo per il controllo della presenza di un veicolo in un incrocio 
    def _vehicle_obstacle_in_junction_detected(self, vehicle_list=None, max_distance=None, up_angle_th=90, low_angle_th=0, lane_offset=0):
        if self._ignore_vehicles:
            return False, 0.0

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )

        for target_vehicle in vehicle_list:
            target_transform = target_vehicle.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            if target_wpt.is_junction:
                target_speed = get_speed(target_vehicle)
                return True, target_speed
                
        return False, 0.0

    def _generate_lane_change_path(self, waypoint, direction='left', distance_same_lane=10,
                                distance_other_lane=25, lane_change_distance=25,
                                check=True, lane_changes=1, step_distance=2):
        """
        This methods generates a path that results in a lane change.
        Use the different distances to fine-tune the maneuver.
        If the lane change is impossible, the returned path will be empty.
        """
        distance_same_lane = max(distance_same_lane, 0.1)
        distance_other_lane = max(distance_other_lane, 0.1)
        lane_change_distance = max(lane_change_distance, 0.1)

        plan = []
        plan.append((waypoint, RoadOption.LANEFOLLOW))  # start position

        option = RoadOption.LANEFOLLOW

        # Same lane
        distance = 0
        while distance < distance_same_lane:
            next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        if direction == 'left':
            option = RoadOption.CHANGELANELEFT
        elif direction == 'right':
            option = RoadOption.CHANGELANERIGHT
        else:
            # ERROR, input value for change must be 'left' or 'right'
            return []

        lane_changes_done = 0
        lane_change_distance = lane_change_distance / lane_changes

        # Lane change
        while lane_changes_done < lane_changes:

            # Move forward
            next_wps = plan[-1][0].next(lane_change_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]

            # Get the side lane
            if direction == 'left':
                if check and str(next_wp.lane_change) not in ['Left', 'Both']:
                    return []
                side_wp = next_wp.get_left_lane()
            else:
                if check and str(next_wp.lane_change) not in ['Right', 'Both']:
                    return []
                side_wp = next_wp.get_right_lane()

            if not side_wp or side_wp.lane_type != carla.LaneType.Driving:
                return []

            # Update the plan
            plan.append((side_wp, option))
            lane_changes_done += 1

        # Other lane
        distance = 0
        while distance < distance_other_lane:
            next_wps = plan[-1][0].next(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        return plan
