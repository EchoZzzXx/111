from leaderboard.envs import sensor_interface


class CallBack(sensor_interface.CallBack):
    def _parse_image_cb(self, image, tag):
        self._data_provider.update_sensor(tag, image, image.frame)
