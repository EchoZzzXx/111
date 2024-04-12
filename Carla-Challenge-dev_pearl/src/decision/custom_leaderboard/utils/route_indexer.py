from leaderboard.utils import route_indexer


class RouteIndexer(route_indexer.RouteIndexer):
    def get_next_config(self, next_route):
        if self.index >= self.total:
            return None

        if next_route:
            self.index += 1
        config = self._configs_list[self.index]

        return config
