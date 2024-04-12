import numpy as np
import cv2

# CARLA_SEGMENT_TAGS = {
#     "Unlabeled": 0,     # Elements that have not been categorized are considered Unlabeled. This category is meant to be empty or at least contain elements with no collisions.
#     "Building": 1,      # Buildings like houses, skyscrapers,... and the elements attached to them. E.g. air conditioners, scaffolding, awning or ladders and much more.
#     "Fence": 2,         # Barriers, railing, or other upright structures. Basically wood or wire assemblies that enclose an area of ground.
#     "Other": 3,         # Everything that does not belong to any other category.
#     "Pedestrian": 4,    # Humans that walk or ride/drive any kind of vehicle or mobility system. E.g. bicycles or scooters, skateboards, horses, roller-blades, wheel-chairs, etc.
#     "Pole": 5,          # Small mainly vertically oriented pole. If the pole has a horizontal part (often for traffic light poles) this is also considered pole. E.g. sign pole, traffic light poles.
#     "RoadLine": 6,      # The markings on the road.
#     "Road": 7,          # Part of ground on which cars usually drive. E.g. lanes in any directions, and streets.
#     "SideWalk": 8,      # Part of ground designated for pedestrians or cyclists. Delimited from the road by some obstacle (such as curbs or poles), not only by markings. This label includes a possibly delimiting curb, traffic islands (the walkable part), and pedestrian zones.
#     "Vegetation": 9,    # Trees, hedges, all kinds of vertical vegetation. Ground-level vegetation is considered Terrain.
#     "Vehicles": 10,     # Cars, vans, trucks, motorcycles, bikes, buses, trains.
#     "Wall": 11,         # Individual standing walls. Not part of a building.
#     "TrafficSign": 12,  # Signs installed by the state/city authority, usually for traffic regulation. This category does not include the poles where signs are attached to. E.g. traffic- signs, parking signs, direction signs...
#     "Sky": 13,          # Open sky. Includes clouds and the sun.
#     "Ground": 14,       # Any horizontal ground-level structures that does not match any other category. For example areas shared by vehicles and pedestrians, or flat roundabouts delimited from the road by a curb.
#     "Bridge": 15,       # Only the structure of the bridge. Fences, people, vehicles, an other elements on top of it are labeled separately.
#     "RailTrack": 16,    # All kind of rail tracks that are non-drivable by cars. E.g. subway and train rail tracks.
#     "GuardRail": 17,    # All types of guard rails/crash barriers.
#     "TrafficLight": 18, # Traffic light boxes without their poles.
#     "Static": 19,       # Elements in the scene and props that are immovable. E.g. fire hydrants, fixed benches, fountains, bus stops, etc.
#     "Dynamic": 20,      # Elements whose position is susceptible to change over time. E.g. Movable trash bins, buggies, bags, wheelchairs, animals, etc.
#     "Water": 21,        # Horizontal water surfaces. E.g. Lakes, sea, rivers.
#     "Terrain": 22       # Grass, ground-level vegetation, soil or sand. These areas are not meant to be driven on. This label includes a possibly delimiting curb.
# }

CARLA_SEGMENT_TAGS = {
    "None": 0,
    "Roads": 1,
    "Sidewalks": 2,
    "Buildings": 3,
    "Walls": 4,
    "Fences": 5,
    "Poles": 6,
    "TrafficLight": 7,
    "TrafficSigns": 8,
    "Vegetation": 9,
    "Terrain": 10,
    "Sky": 11,
    "Pedestrians": 12,
    "Rider": 13,
    "Car": 14,
    "Truck": 15,
    "Bus": 16,
    "Train": 17,
    "Motorcycle": 18,
    "Bicycle": 19,
    "Static": 20,
    "Dynamic": 21,
    "Other": 22,
    "Water": 23,
    "RoadLines": 24,
    "Ground": 25,
    "Bridge": 26,
    "RailTrack": 27,
    "GuardRail": 28,
    "Any": 255,
}


class SegWrapper:
    def __init__(self):
        self.idx_static = [
            CARLA_SEGMENT_TAGS[key]
            for key in [
                "Sidewalks",
                "Buildings",
                "Walls",
                "Fences",
                "Vegetation",
                "Terrain",
                "Static",
                "Other",
                "Water",
                "Ground",
                "RailTrack",
            ]
        ]
        self.idx_dynamic = [
            CARLA_SEGMENT_TAGS[key]
            for key in [
                "Pedestrians",
                "Rider",
                "Car",
                "Truck",
                "Bus",
                "Train",
                "Motorcycle",
                "Bicycle",
                "Dynamic",
            ]
        ]
        self.idx_road = [CARLA_SEGMENT_TAGS[key] for key in ["RoadLines", "Roads"]]

    def seg2gray(self, image):
        dst = np.ones(image.shape, dtype=np.uint8)
        dst = np.where(np.isin(image, self.idx_road), 0, dst)
        seg_gray = np.where(np.isin(image, self.idx_dynamic), 2, dst)

        # return seg_gray
        return seg_gray * 127

    def stream2seg(self, input_data):
        pass
        # seg_raw = input_data["bev_seg"][1]
        # seg_1d = np.frombuffer(seg_raw.raw_data, dtype=np.uint8)
        # h, w = AgentConfig.sensor_seg_height, AgentConfig.sensor_seg_width
        # seg_label = np.reshape(seg_1d, (h, w, 4))[..., 2]

        # wrapper = SegWrapper()
        # seg_gray = wrapper.seg2gray(seg_label)

        # state["bev_feature"] = np.reshape(seg_gray, -1)

    def seg2rgb(self, image):
        pass
        # seg_raw.save_to_disk('./test.png', carla.ColorConverter.CityScapesPalette)
        # seg_raw.convert(carla.ColorConverter.CityScapesPalette)
        # seg_bgra = np.frombuffer(seg_raw.raw_data, dtype=np.uint8)
        # seg_rgb = np.reshape(seg_bgra, (h, w, 4))[-2::-1]
