# Autonomous Driving Agent in CARLA

This repository contains a project for the **Autonomous Vehicle Driving** course focused on the design, implementation, and evaluation of an autonomous driving agent in the **CARLA simulator**.

The goal of the project is to improve a baseline driving agent so that it can safely navigate complex urban scenarios, respect traffic rules, react to dynamic obstacles, and achieve a higher **Global Driving Score**. The report describes the project as an implementation of an autonomous vehicle in CARLA, tested on multiple simulated scenarios and evaluated with route completion and traffic-infraction penalties. 

## Project Overview

The agent was developed to:

- follow a predefined route to a target destination
- comply with traffic rules such as traffic lights and stop signs
- react to dynamic road conditions
- handle pedestrians, cyclists, static obstacles, and lateral vehicles
- behave safely at intersections and during overtaking maneuvers
- adapt speed to adverse weather conditions

The project was tested on different CARLA scenarios including lane changes, intersections, pedestrian crossings, emergency situations, and poor weather conditions.

## Evaluation Metric

The main evaluation metric is the **Global Driving Score**, computed as the product of:

- **Route Completion**: percentage of the route successfully completed
- **Infraction Penalty**: value starting from `1.0` and reduced according to committed infractions

Relevant infractions include:

- collisions with pedestrians
- collisions with vehicles
- collisions with static obstacles
- running red lights
- ignoring stop signs
- scenario timeout
- failing minimum speed constraints
- failing to give way to emergency vehicles
- going off-road

## Tested Routes

### Route 1
Route 1 takes place in **Town12** and includes:

- curved roads
- multiple intersections
- static and dynamic obstacles
- cyclists on side lanes
- vehicles invading the ego lane
- nighttime driving
- worsening weather conditions
- dense fog and strong wind

### Route 4
Route 4 also takes place in **Town12**, but focuses more on:

- sudden pedestrian crossings
- cyclist interactions
- dynamic obstacle handling

The route is executed during daytime with moderate rain and dense fog.

## Baseline Analysis

The initial baseline was based on several existing CARLA modules:

- `BasicAgent`
- `Basic Autonomous Agent`
- `BehaviorAgent`
- `BehaviourTypes`
- `VehicleController`
- `LocalPlanner`
- `GlobalRoutePlanner`

The baseline provided core functionalities such as:

- traffic light handling
- nearby vehicle detection
- lane-change path generation
- longitudinal PID control
- lateral Stanley control
- waypoint following

However, the baseline showed important limitations:

- poor reaction to dangerous situations
- missing stop-sign handling
- multiple collisions
- weak obstacle management
- poor intersection behavior

### Baseline Results

**Route 1**
- Route Completion: `12.14%`
- Infraction Penalty: `0.455`
- Driving Score: `5.5237`

**Route 4**
- Route Completion: `100%`
- Infraction Penalty: `0.011612`
- Driving Score: `1.161216`

These results highlighted the need for significant improvements in obstacle handling, intersection management, and overall driving safety.

## Final Results

After applying the improvements, the agent showed a much better behavior.

### Route 1
- Route Completion: `100%`
- OutsideRouteLanesTest: failed with `0.53%`
- CollisionTest: `1` collision
- MinSpeedTest: failed with `93.7%`
- all other main safety tests passed

### Route 4
- Route Completion: `100%`
- stayed entirely within the lane
- respected traffic lights and stop signs
- no timeouts or traffic blocking
- MinSpeedTest passed
- still had `1` collision
