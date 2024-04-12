import os
from collections import deque

import numpy as np


DEBUG = int(os.environ.get("HAS_DISPLAY", 0))


class Plotter(object):
    def __init__(self, size):
        self.size = size
        self.clear()
        self.title = str(self.size)

    def clear(self):
        from PIL import Image, ImageDraw

        self.img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        self.draw = ImageDraw.Draw(self.img)

    def dot(self, pos, node, color=(255, 255, 255), r=2):
        x, y = 5.5 * (pos - node)
        x += self.size / 2
        y += self.size / 2

        self.draw.ellipse((x - r, y - r, x + r, y + r), color)

    def show(self):
        if not DEBUG:
            return

        import cv2

        cv2.imshow(self.title, cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)


class RoutePlanner(object):
    def __init__(self, min_distance, max_distance, window_size=3, debug_size=256):
        self.route = deque()
        self.window_size = window_size
        self.min_distance = min_distance
        self.max_distance = max_distance

        self.T_gw = None
        self.compass = None

        self.debug = Plotter(debug_size)

    def set_route(self, global_plan_geo, global_plan_world):
        self.route.clear()

        loc_geo = np.array([[pt[0]["lon"], pt[0]["lat"]] for pt in global_plan_geo])
        loc_world = np.array([[pt[0].location.x, pt[0].location.y] for pt in global_plan_world])

        self.T_gw = np.array(
            [
                np.polyfit(loc_geo[:, 0], loc_world[:, 0], deg=1)[0],
                np.polyfit(loc_geo[:, 1], loc_world[:, 1], deg=1)[0],
            ]
        )

        for pos, cmd in global_plan_world[1:]:
            pos = np.array([pos.location.x, pos.location.y, int(cmd)])
            self.route.append(pos)

    def run_step(self, gps):
        self.debug.clear()

        gps = gps[::-1] * self.T_gw

        if len(self.route) < self.window_size:
            self.route = np.array(self.route)
            tmp = np.repeat(self.route[-1:,:], self.window_size - len(self.route), axis=0)
            return np.concatenate((self.route, tmp))

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(self.route[i][:2] - self.route[i - 1][:2])
            distance = np.linalg.norm(self.route[i][:2] - gps)

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

            # r = 255 * int(distance > self.min_distance)
            # g = 255 * int(self.route[i][1].value == 4)
            # b = 255
            # self.debug.dot(gps, self.route[i][0], (r, g, b))

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        # self.debug.dot(gps, self.route[0][0], (0, 255, 0))
        # self.debug.dot(gps, self.route[1][0], (255, 0, 0))
        # self.debug.dot(gps, gps, (0, 0, 255))
        # self.debug.show()

        if len(self.route) < self.window_size:
            tmp = np.repeat(self.route[-1], self.window_size - len(self.route))
            return np.concatenate(np.array(self.route), tmp)
        else:
            # print('TIPS: showing nearest 5 waypoints: \n', np.array(self.route)[:self.window_size])
            return np.array(self.route)[: self.window_size]

    def _get_angle_to(self, pos, theta, target):
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

        aim = R.T.dot(target - pos)
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
        angle = 0.0 if np.isnan(angle) else angle

        return angle
