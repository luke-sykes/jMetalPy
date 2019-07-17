import logging
import numpy
from abc import ABC, abstractmethod
from typing import TypeVar, List

from jmetal.core.solution import Solution
from scipy.spatial import distance_matrix
from scipy.spatial.distance import euclidean

LOGGER = logging.getLogger('jmetal')

S = TypeVar('S')

"""
.. module:: crowding_distance
   :platform: Unix, Windows
   :synopsis: Crowding distance implementation.

.. moduleauthor:: Álvaro Gómez Jáuregui <alvarogj@lcc.uma.es>
"""


class DensityEstimator(List[S], ABC):
    """This is the interface of any density estimator algorithm.
    """

    @abstractmethod
    def compute_density_estimator(self, solution_list: List[S]) -> float:
        pass


class CrowdingDistance(DensityEstimator[List[S]]):
    """This class implements a DensityEstimator based on the crowding distance.
    In consequence, the main method of this class is :func:`compute_density_estimator`.
    """

    def compute_density_estimator(self, front: List[S]):
        """This function performs the computation of the crowding density estimation over the solution list.

        .. note::
           This method assign the distance in the inner elements of the solution list.

        :param front: The list of solutions.
        """
        size = len(front)

        if size is 0:
            return
        elif size is 1:
            front[0].attributes['crowding_distance'] = float("inf")
            return
        elif size is 2:
            front[0].attributes['crowding_distance'] = float("inf")
            front[1].attributes['crowding_distance'] = float("inf")
            return

        for i in range(len(front)):
            front[i].attributes['crowding_distance'] = 0.0

        number_of_objectives = front[0].number_of_objectives

        for i in range(number_of_objectives):
            # Sort the population by Obj n
            front = sorted(front, key=lambda x: x.objectives[i])
            objective_minn = front[0].objectives[i]
            objective_maxn = front[len(front) - 1].objectives[i]

            # Set de crowding distance
            front[0].attributes['crowding_distance'] = float('inf')
            front[size - 1].attributes['crowding_distance'] = float('inf')

            for j in range(1, size - 1):
                distance = front[j + 1].objectives[i] - front[j - 1].objectives[i]

                # Check if minimum and maximum are the same (in which case do nothing)
                if objective_maxn - objective_minn == 0:
                    pass;  # LOGGER.warning('Minimum and maximum are the same!')
                else:
                    distance = distance / (objective_maxn - objective_minn)

                distance += front[j].attributes['crowding_distance']
                front[j].attributes['crowding_distance'] = distance


class KNearestNeighborDensityEstimator(DensityEstimator[List[S]]):
    """This class implements a density estimator based on the distance to the k-th nearest solution.
    """

    def __init__(self, k=1):
        self.k = k
        self.distance_matrix = []

    def compute_density_estimator(self, solutions: List[S]):
        solutions_size = len(solutions)
        if solutions_size <= self.k:
            return

        points = []
        for i in range(solutions_size):
            points.append(solutions[i].objectives)

        # Compute distance matrix
        self.distance_matrix = numpy.zeros(shape=(solutions_size, solutions_size))
        for i in range(solutions_size):
            for j in range(solutions_size):
                self.distance_matrix[i, j] = self.distance_matrix[j, i] = euclidean(solutions[i].objectives,
                                                                                    solutions[j].objectives)

        print(self.distance_matrix)

        for i in range(solutions_size):
            distances = []
            for j in range(solutions_size):
                distances.append(self.distance_matrix[i, j])
            distances.sort()
            solutions[i].attributes['knn_density'] = distances[self.k]


