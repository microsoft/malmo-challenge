# Copyright (c) 2017 Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================

from __future__ import absolute_import

from heapq import heapify, heappop, heappush
from collections import deque

from . import BaseAgent


class AStarAgent(BaseAgent):
    def __init__(self, name, nb_actions, visualizer=None):
        super(AStarAgent, self).__init__(name, nb_actions, visualizer)

    def _find_shortest_path(self, start, end, **kwargs):
        came_from, cost_so_far = {}, {}
        explorer = []
        heapify(explorer)

        heappush(explorer, (0, start))
        came_from[start] = None
        cost_so_far[start] = 0
        current = None

        while len(explorer) > 0:
            _, current = heappop(explorer)

            if self.matches(current, end):
                break

            for nb in self.neighbors(current, **kwargs):
                cost = nb.cost if hasattr(nb, "cost") else 1
                new_cost = cost_so_far[current] + cost

                if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                    cost_so_far[nb] = new_cost
                    priority = new_cost + self.heuristic(end, nb, **kwargs)
                    heappush(explorer, (priority, nb))
                    came_from[nb] = current

        # build path:
        path = deque()
        while current is not start:
            path.appendleft(current)
            current = came_from[current]
        return path, cost_so_far

    def neighbors(self, pos, **kwargs):
        raise NotImplementedError()

    def heuristic(self, a, b, **kwargs):
        raise NotImplementedError()

    def matches(self, a, b):
        return a == b
