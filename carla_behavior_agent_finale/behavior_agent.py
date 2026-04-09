# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import random
import math
import numpy as np
import carla
from basic_agent import BasicAgent
from local_planner import RoadOption
from behavior_types import Cautious, Aggressive, Normal

from misc import get_speed, positive, is_within_distance, compute_distance


class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    def __init__(self, vehicle, behavior='normal', opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._curve_speed = 30
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 4.5   
        self._past_steering = 0.0     

        self._ignore_stop = None
        self._stop_counter = 0
        self._in_junction = False
        self._junction_waypoint_list = None
        self._last_junction_waypoint_index = 0

        self._overtaking = None
        
        # supporto per le biciclette
        self._bike_type_list = ['vehicle.gazelle.omafiets', 'vehicle.bh.crossbike', 'vehicle.diamondback.century']

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()
        
        self._initial_max_speed = self._behavior.max_speed


    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._past_steering = self._local_planner._vehicle_controller.past_steering
        self._direction = self._local_planner.target_road_option
        
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

        # print("Vehicle velocity:", self._speed)


    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected
    

    def stop_sign_manager(self):
        """
        This method is in charge of behaviors for stop sign.

        return: presence of stop sign and distance from it
        """
        actor_list = self._world.get_actors()
        stop_sign_list = actor_list.filter("*stop*")
        if self._ignore_stop is not None:
            stop_sign_list = [stop_sign for stop_sign in stop_sign_list if stop_sign.id != self._ignore_stop.id]
        
        affected, stop_sign, distance = self._affected_by_stop_sign(stop_sign_list)

        if round(self._speed, ndigits=1) == 0.0 and affected:
            if self._stop_counter > 20:
                self._ignore_stop = stop_sign
                self._stop_counter = 0
            else:
                self._stop_counter += 1

        return affected, distance


    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """
        vehicle_list = self._world.get_actors().filter("*vehicle*")

        def dist(v):
            return v.get_location().distance(waypoint.transform.location)

        # getting the list of vehicle that respect a limit distance and are not a bike
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id and not v.type_id in self._bike_type_list]

        if self._direction == RoadOption.CHANGELANELEFT:
            return self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            return self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            return self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=120)


    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")

        def dist(w):
            return w.get_location().distance(waypoint.transform.location)

        walker_list = [w for w in walker_list if dist(w) < 20]

        if self._direction == RoadOption.CHANGELANELEFT:
            tmp_walker_list = self._walker_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            tmp_walker_list= self._walker_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            tmp_walker_list = self._walker_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        return tmp_walker_list
    
    # Metodo per evitare i ciclisti
    def biker_avoid_manager(self, waypoint):
        nearby_vehicles = self._world.get_actors().filter("*vehicle*")

        def distance_to_wp(actor):
            return actor.get_location().distance(waypoint.transform.location)

        filtered_bikers = [
            veh for veh in nearby_vehicles
            if distance_to_wp(veh) < 35 and veh.id != self._vehicle.id and veh.type_id in self._bike_type_list
        ]

        if filtered_bikers:
            proximity = max(self._behavior.min_proximity_threshold, self._speed_limit / 2)

            if self._direction == RoadOption.CHANGELANELEFT:
                return self._biker_obstacle_detected(filtered_bikers, proximity, up_angle_th=180, lane_offset=-1)

            elif self._direction == RoadOption.CHANGELANERIGHT:
                return self._biker_obstacle_detected(filtered_bikers, proximity, up_angle_th=180, lane_offset=1)

            else:
                proximity = max(self._behavior.min_proximity_threshold, self._speed_limit / 3)
                return self._biker_obstacle_detected(filtered_bikers, proximity, up_angle_th=135)

        return [(False, None, -1, None)]


    # Metodo per evitare gli ostacoli laterali
    def side_obstacle_avoid_manager(self, waypoint):
        all_vehicles = self._world.get_actors().filter("*vehicle*")

        def distance(actor): return actor.get_location().distance(waypoint.transform.location)
        close_vehicles = [veh for veh in all_vehicles if distance(veh) < 50]

        ego_tf = self._vehicle.get_transform()
        ego_wp = self._map.get_waypoint(self._vehicle.get_location())

        if waypoint.is_junction or self._incoming_waypoint.is_junction:
            return []

        right_lane = waypoint.get_right_lane()
        right_lane_id = right_lane.lane_id

        ego_dir = ego_tf.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_tf = ego_tf
        ego_front_tf.location += carla.Location(
            x=ego_extent * ego_dir.x,
            y=ego_extent * ego_dir.y,
        )

        nearby_obstacles = []

        for veh in close_vehicles:
            veh_tf = veh.get_transform()
            veh_wp = self._map.get_waypoint(veh_tf.location, lane_type=carla.LaneType.Any)

            veh_bb = veh.bounding_box.get_world_vertices(veh_tf)
            on_same_lane = False
            for pt in veh_bb:
                if self._map.get_waypoint(pt, lane_type=carla.LaneType.Any).lane_id == ego_wp.lane_id:
                    on_same_lane = True
                    break

            if (
                veh_wp.road_id == waypoint.road_id and
                veh_wp.lane_id == right_lane_id and
                veh_wp.lane_type == carla.LaneType.Shoulder and
                is_within_distance(veh_tf, ego_front_tf, max_distance=40, angle_interval=[0, 45]) and
                on_same_lane
            ):
                dist = compute_distance(veh_tf.location, ego_tf.location)
                nearby_obstacles.append((True, veh, dist))

        return sorted(nearby_obstacles, key=lambda x: x[2])



    # Metodo per evitare gli ostacoli statici
    def static_obstacle_avoid_manager(self, waypoint, lane='same'):
        all_obstacles = self._world.get_actors().filter("*static.prop*")

        def distance_to_wp(actor):
            return actor.get_location().distance(waypoint.transform.location)

        filtered_obstacles = [
            obs for obs in all_obstacles
            if distance_to_wp(obs) < 30 and "dirtdebris" not in obs.type_id
        ]

        ego_tf = self._vehicle.get_transform()
        ego_dir = ego_tf.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_tf = ego_tf
        ego_front_tf.location += carla.Location(
            x=ego_extent * ego_dir.x,
            y=ego_extent * ego_dir.y,
        )

        relevant_obstacles = []

        for obs in filtered_obstacles:
            obs_tf = obs.get_transform()
            obs_wp = self._map.get_waypoint(obs_tf.location, lane_type=carla.LaneType.Any)

            if lane == "left":
                # Controllo ostacoli nella corsia a sinistra
                if (
                    obs_wp.road_id == waypoint.road_id and
                    obs_wp.lane_id != waypoint.lane_id and
                    is_within_distance(obs_tf, ego_front_tf, max_distance=50, angle_interval=[0, 135], preference="left")
                ):
                    distance = compute_distance(obs_tf.location, ego_tf.location)
                    relevant_obstacles.append((True, obs, distance))

            else:
                # Controllo ostacoli nella stessa corsia
                if (
                    obs_wp.road_id == waypoint.road_id and
                    obs_wp.lane_id == waypoint.lane_id and
                    is_within_distance(obs_tf, ego_front_tf, max_distance=50, angle_interval=[0, 30])
                ):
                    distance = compute_distance(obs_tf.location, ego_tf.location)
                    relevant_obstacles.append((True, obs, distance))

        return sorted(relevant_obstacles, key=lambda x: x[2])



    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        # Freno d’emergenza se troppo vicino
        if distance < self._behavior.braking_distance:
            return carla.VehicleControl(
                throttle=0.0,
                brake=self._max_brake,
                hand_brake=False
            )


        vehicle_speed = get_speed(vehicle)

        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
           
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    # Metodo per generare il percorso di sorpasso
    def _generate_path_for_overtaking(self, distance_for_overtaking, direction='left', two_way=False):
        current_wp = self._map.get_waypoint(self._vehicle.get_location())
        step = self._sampling_resolution
        previous_plan = self._local_planner._waypoints_queue

        new_plan = [(current_wp, RoadOption.LANEFOLLOW)]

        next_wp = current_wp.next(step)[0]

        if direction == 'left':
            overtaking_wp = next_wp.get_left_lane()
            new_plan.append((next_wp, RoadOption.CHANGELANELEFT))
            new_plan.append((overtaking_wp, RoadOption.LANEFOLLOW))
        else:
            overtaking_wp = next_wp.get_lane_right()
            new_plan.append((next_wp, RoadOption.CHANGELANERIGHT))
            new_plan.append((overtaking_wp, RoadOption.LANEFOLLOW))

        traveled = 0.0
        while traveled < distance_for_overtaking:
            if two_way:
                following_wps = new_plan[-1][0].previous(step)
            else:
                following_wps = new_plan[-1][0].next(step)

            next_segment = following_wps[0]
            traveled += next_segment.transform.location.distance(new_plan[-1][0].transform.location)
            new_plan.append((next_segment, RoadOption.LANEFOLLOW))

        if two_way:
            final_wp = new_plan[-1][0].previous(10)[0]
            return_wp = final_wp.get_left_lane()
            new_plan.append((return_wp, RoadOption.CHANGELANERIGHT))
        else:
            final_wp = new_plan[-1][0].next(10)[0]
            return_wp = final_wp.get_left_lane()
            new_plan.append((return_wp, RoadOption.CHANGELANERIGHT))

        previous_wp_only = [wp_pair[0] for wp_pair in previous_plan]
        closest_index = self._global_planner._find_closest_in_list(new_plan[-1][0], previous_wp_only)

        for i in range(closest_index, len(previous_wp_only)):
            new_plan.append(self._local_planner._waypoints_queue[i])

        self.set_global_plan(new_plan)


    # Metodo per controllare la presenza di veicoli nella corsia di sorpasso
    def _check_for_vehicle_on_overtaking_lane(self, waypoint, distance):
        potential_vehicles = self._world.get_actors().filter("*vehicle*")

        def compute_dist(actor):
            return actor.get_location().distance(waypoint.transform.location)

        left_lane = waypoint.get_left_lane()
        left_lane_id = left_lane.lane_id

        is_danger = False
        blocking_vehicle = False
        min_distance = distance

        for vehicle in potential_vehicles:
            vehicle_tf = vehicle.get_transform()
            vehicle_wp = self._map.get_waypoint(vehicle_tf.location, lane_type=carla.LaneType.Any)

            if vehicle_wp.road_id == waypoint.road_id and vehicle_wp.lane_id == left_lane_id:
                curr_distance = compute_dist(vehicle)
                if is_within_distance(vehicle_tf, waypoint.transform, min_distance, [0, 90]):
                    is_danger = True
                    blocking_vehicle = vehicle
                    min_distance = curr_distance

        return is_danger, blocking_vehicle, min_distance


    # Metodo per gestire il sorpasso
    def overtake(self, waypoint, target, target_length, target_distance, target_speed=0, debug=False, security_distance=5):
        total_overtake_dist = target_distance + target_length
        ego_speed = min(self._behavior.max_speed, self._speed_limit - self._behavior.speed_lim_dist) / 3.6
        overtake_time = total_overtake_dist / (ego_speed - target_speed)

        danger_detected, blocking_vehicle, block_distance = self._check_for_vehicle_on_overtaking_lane(waypoint, 100)

        # Se la corsia di sorpasso è libera
        if not danger_detected:
            if target_distance < self._behavior.braking_distance:
                self._generate_path_for_overtaking(total_overtake_dist - 1.5, 'left', True)
                self._local_planner.set_speed(ego_speed * 3.6)
                control = self._local_planner.run_step(debug=debug)
            else:
                control = self._local_planner.run_step(debug=debug)

        else:
            block_speed = get_speed(blocking_vehicle) / 3.6
            available_space = block_distance - block_speed * (overtake_time + 3)

            if available_space > total_overtake_dist + 5:
                if target_distance < self._behavior.braking_distance + security_distance:
                    self._generate_path_for_overtaking(total_overtake_dist - 1.5, 'left', True)
                    self._local_planner.set_speed(ego_speed * 3.6)
                    self._local_planner.set_speed(min(ego_speed * 3.6, 15.0))  # max 15 km/h durante rientro
                    control = self._local_planner.run_step(debug=debug)
                else:
                    control = self._local_planner.run_step(debug=debug)
            else:
                if target_distance < self._behavior.braking_distance + security_distance:
                    return self.emergency_stop()

                control = self.car_following_manager(target, target_distance)

        return control


    # Metodo per ottenere la direzione di movimento dell'attore
    def get_actor_moving_direction(self, target_transform, reference_transform):
        ego_fwd = reference_transform.get_forward_vector()
        ego_vec = np.array([ego_fwd.x, ego_fwd.y])

        actor_fwd = target_transform.get_forward_vector()
        actor_vec = np.array([actor_fwd.x, actor_fwd.y])

        actor_norm = np.linalg.norm(actor_vec)

        angle_deg = math.degrees(math.acos(np.clip(np.dot(ego_vec, actor_vec) / actor_norm, -1.0, 1.0)))

        move_dir = "forward"
        if not (0 < angle_deg < 60):
            move_dir = "crossing"

        return move_dir

    
    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        control.steer = self._past_steering

        return control

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        self._update_information()
        
        precipitations_deposits = self._world.get_weather().precipitation_deposits
        self._behavior.max_speed = self._initial_max_speed - 15 * (precipitations_deposits/100)

        control = None

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        self._previous_wp = ego_vehicle_wp
        self._next_wp = ego_vehicle_wp.next(3)[0]

        # ------------------------------------------------------------------
        # ---------- GETTING ALL INFORMATION NEAR THE EGO VEHICLE ----------
        # ------------------------------------------------------------------

        # ----- PEDESTRIAN INFORMATION -----
        walker_list = self.pedestrian_avoid_manager(ego_vehicle_wp)
        walker_state, walker, w_distance, is_walker_same_lane= walker_list[0] if len(walker_list) > 0 else (False, None, -1, None)
        
        # ----- BIKER INFORMATION -----
        biker_list = self.biker_avoid_manager(ego_vehicle_wp)
        biker_state, biker, biker_distance, is_biker_same_lane = biker_list[0] if len(biker_list) > 0 else (False, None, -1, None)
              
        # ----- VEHICLE INFORMATION -----
        vehicle_list = self.collision_and_car_avoid_manager(ego_vehicle_wp)
        vehicle_state, vehicle, v_distance = vehicle_list[0] if len(vehicle_list) > 0 else (False, None, -1)

        # ----- SIDE VEHICLE INFORMATION -----
        side_vehicle_list = self.side_obstacle_avoid_manager(ego_vehicle_wp)
        side_vehicle_state, side_vehicle, side_vehicle_distance = side_vehicle_list[0] if len(side_vehicle_list) > 0 else (False, None, -1)

        # ----- SIDE OBSTACLE INFORMATION -----
        obstacle_list = self.static_obstacle_avoid_manager(ego_vehicle_wp)
        obstacle_state, obstacle, o_distance = obstacle_list[0] if len(obstacle_list) > 0 else (False, None, -1)

        # ----- LEFT LANE SIDE OBSTACLE INFORMATION -----
        side_other_line_obstacle_list = self.static_obstacle_avoid_manager(ego_vehicle_wp, lane="left")
        side_other_line_obstacle_state, side_other_line_obstacle, side_other_line_obstacle_distance = \
                side_other_line_obstacle_list[0] if len(side_other_line_obstacle_list) > 0 else (False, None, -1)

        # ----- JUNCTION INFORMATION -----
        self._in_junction = ego_vehicle_wp.is_junction
        if self._incoming_waypoint.is_junction and self._junction_waypoint_list is None:
            self._junction_waypoint_list = self._local_planner.get_junction_waypoint()

        # ------------------------------------------------------------
        # ----------- MANAGING ALL THE INFORMATION OBTAINED ----------
        # ------------------------------------------------------------

        # ------ TRAFFIC LIGHT AND STOP BEHAVIOR -----
        if self.traffic_light_manager():
            print("TRAFFIC LIGHT STATE - EMERGENCY STOP")
            return self.emergency_stop()

        affected_by_stop_sign, stop_sign_distance = self.stop_sign_manager()

        if affected_by_stop_sign:
            if stop_sign_distance < self._behavior.braking_distance:
                print("STOP STATE - EMERGENCY STOP")
                return self.emergency_stop()
            else:
                print("STOP STATE - SLOWING DOWN")
                if vehicle_state:
                    distance = v_distance \
                    - max(
                        vehicle.bounding_box.extent.y, 
                        vehicle.bounding_box.extent.x) \
                            - max(
                                self._vehicle.bounding_box.extent.y, 
                                self._vehicle.bounding_box.extent.x)
            
                    # emergency brake if the car is very close
                    if distance < self._behavior.braking_distance:
                        print("STOP STATE WITH VEHICLE - EMERGENCY STOP")
                        return self.emergency_stop()
                    else:
                        print("STOP STATE WITH VEHICLE STATE - CAR FOLLOWING")
                        control = self.car_following_manager(vehicle, distance)
                else:
                    target_speed = min([
                        self._behavior.max_speed,
                        self._speed_limit - self._behavior.speed_lim_dist,
                        20])
                    self._local_planner.set_speed(target_speed)
                    control = self._local_planner.run_step(debug=debug)

                return control

        # ----- WALKER AND BIKER BEHAVIOR -----
        if walker_state:
            distance = w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            if distance < self._behavior.braking_distance and (is_walker_same_lane or round(get_speed(walker), ndigits=1) > 0.0):
                print("WAKLER STATE - EMERGENCY STOP")
                return self.emergency_stop()
            else:
                print("WAKLER STATE - SLOWING DOWN")
                target_speed = min([
                    self._behavior.max_speed,
                    self._speed_limit - 20,
                    self._curve_speed,
                    20]
                    )
                self._local_planner.set_speed(target_speed)
                return self._local_planner.run_step(debug=debug)

        if biker_state:
            direction = self.get_actor_moving_direction(biker.get_transform(), self._vehicle.get_transform())
            
            if direction == "forward": 
                # if biker is moving forward, just change the vehicle offset
                print("BIKER STATE - BIKER ON THE RIGHT")
                self._local_planner._vehicle_controller._lat_controller.set_offset(-0.9)
                return self._local_planner.run_step(debug=debug)
            else: 
                # else the biker is crossing the road, same behavior as the walker
                distance = biker_distance \
                    - max(
                        biker.bounding_box.extent.y,
                        biker.bounding_box.extent.x) \
                            - max(
                                self._vehicle.bounding_box.extent.y,
                                self._vehicle.bounding_box.extent.x)
                
                if distance < self._behavior.braking_distance and (is_biker_same_lane or round(get_speed(biker), ndigits=1) > 0.0):
                    print("BIKER STATE - EMERGENCY STOP")
                    return self.emergency_stop()
                else:
                    print("BIKER STATE - SLOWING DOWN")
                    # return self.car_following_manager(biker, distance)
                    target_speed = min([
                    self._behavior.max_speed,
                    self._speed_limit - 20,
                    self._curve_speed,
                    20]
                    )
                    self._local_planner.set_speed(target_speed)
                    return self._local_planner.run_step(debug=debug)

        # ----- VEHICLE BEHAVIOR -----
        if vehicle_state:
            distance = v_distance \
                - max(
                    vehicle.bounding_box.extent.y, 
                    vehicle.bounding_box.extent.x) \
                        - max(
                            self._vehicle.bounding_box.extent.y, 
                            self._vehicle.bounding_box.extent.x)
            
            # emergency brake if the car is very close
            if distance < self._behavior.braking_distance:
                print("VEHICLE STATE - EMERGENCY STOP")
                return self.emergency_stop()
            else:
                print("VEHICLE STATE - CAR FOLLOWING")
                control = self.car_following_manager(vehicle, distance)

        elif side_vehicle_state:
            print("SIDE VEHICLE STATE - OVERTAKE")
            # car stopped shoulders
            if len(side_vehicle_list) == 1:
                sv = side_vehicle_list[0]
                side_vehicle_length = max(sv[1].bounding_box.extent.x, sv[1].bounding_box.extent.y) * 2
            else:
                side_vehicle_length = side_vehicle_list[-1][2] - side_vehicle_list[0][2]

            control = self.overtake(ego_vehicle_wp, side_vehicle, side_vehicle_length, side_vehicle_distance)
            
        # 3: Intersection behavior
        elif self._incoming_waypoint.is_junction or self._in_junction:

            vehicle_in_junction_list = self._world.get_actors().filter("*vehicle*")
            def dist(v): return v.get_location().distance(ego_vehicle_wp.transform.location)

            # getting the list of vehicle that respect a limit distance and are not a bike
            vehicle_in_junction_list = [v for v in vehicle_in_junction_list if dist(v) < 45 and v.id != self._vehicle.id and not v.type_id in self._bike_type_list]

            is_vehicle_in_junction, vehicle_speed = self._vehicle_obstacle_in_junction_detected(vehicle_in_junction_list, max(
                        self._behavior.min_proximity_threshold, self._speed_limit / 3))

            if self._incoming_direction == RoadOption.RIGHT:
                self._local_planner._vehicle_controller._lat_controller.set_offset(-0.2)
            else:
                self._local_planner._vehicle_controller._lat_controller.set_offset(0.0)

            index = 0
            total_length = 0
            current_waypoint = self._local_planner.get_current_waypoint()

            if self._junction_waypoint_list is not None:
                try:
                    index = self._junction_waypoint_list.index(current_waypoint.id) + 1
                    self._last_junction_waypoint_index = index
                except:
                    index = self._last_junction_waypoint_index
                
                total_length = len(self._junction_waypoint_list)
            
            percentage = (index/total_length) * 100
            is_junction_committed = percentage > 70

            if not is_vehicle_in_junction or (self._in_junction and round(vehicle_speed, ndigits=1) == 0.0) or is_junction_committed:
                print("INCOMING JUNCTION STATE - NORMAL BEHAVIOR")
                target_speed = min([
                    self._behavior.max_speed,
                    self._speed_limit - 15,
                    self._curve_speed]
                    )
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)
            else:
                print("INCOMING JUNCTION STATE - EMERGENCY STOP")
                return self.emergency_stop()

        # static obstacle
        elif obstacle_state:
            print(f"INCOMING ACCIDENT STATE")

            if len(obstacle_list) == 1:
                ob = obstacle_list[0]
                obstacle_length = max(ob[1].bounding_box.extent.x, ob[1].bounding_box.extent.y) * 2
            else:
                obstacle_length = obstacle_list[-1][2] - obstacle_list[0][2]

            control = self.overtake(ego_vehicle_wp, obstacle, obstacle_length, o_distance)
                               
        # obstacle in the left lane 
        elif side_other_line_obstacle_state:
            print("SIDE OBSTACLE STATE - MOVING TO THE RIGHT")
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])

            self._local_planner.set_speed(target_speed)
            self._local_planner._vehicle_controller._lat_controller.set_offset(0.85)
            control = self._local_planner.run_step(debug=debug)

        # 4: Normal behavior
        else:
            print("NORMAL BEHAVIOR STATE")
            self._junction_waypoint_list = None
            self._last_junction_waypoint_index = 0

            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])

            self._local_planner.set_speed(target_speed)
            if self._incoming_direction == RoadOption.RIGHT:
                self._local_planner._vehicle_controller._lat_controller.set_offset(-0.2)
            else:
                self._local_planner._vehicle_controller._lat_controller.set_offset(0.0)
            control = self._local_planner.run_step(debug=debug)

        return control
