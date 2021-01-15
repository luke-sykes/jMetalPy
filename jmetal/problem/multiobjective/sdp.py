from abc import ABC, abstractmethod
from math import floor, sqrt, pow, sin, pi, cos
import numpy

from jmetal.core.problem import DynamicProblem, FloatProblem
from jmetal.core.solution import FloatSolution


class SDP(DynamicProblem, FloatProblem, ABC):
    def __init__(self):
        super(SDP, self).__init__()
        self.tau_T = 5
        self.nT = 10
        self.time = 1.0
        self.problem_modified = False

    def update(self, *args, **kwargs):
        counter: int = kwargs["COUNTER"]
        self.time = (1.0 / self.nT) * floor(counter * 1.0 / self.tau_T)
        self.problem_modified = True

    def the_problem_has_changed(self) -> bool:
        return self.problem_modified

    def clear_changed(self) -> None:
        self.problem_modified = False

    @abstractmethod
    def evaluate(self, solution: FloatSolution):
        pass


class SDP1(SDP):
    """ Problem SDP1.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 100.
    """

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(SDP1, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = self.number_of_variables * [-1.0]
        self.upper_bound = self.number_of_variables * [1.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0
        self.sdp_y = []

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        g = self.__eval_g(solution)
        h = self.__eval_h(solution.variables)

        for j in range(self.number_of_objectives):
            xx = 3 * solution.variables[j] + 1
            solution.objectives[j] = (1 + g)*xx / pow(h / xx, 1.0 / (self.number_of_objectives-1))

        return solution

    def __eval_g(self, solution: FloatSolution):
        g = 0
        for i in range(self.number_of_objectives, self.number_of_variables):
            g += pow(solution.variables[i] - self.sdp_y[i], 2)

        return g

    def __eval_h(self, f: float) -> float:
        prd = 1
        for j in range(self.number_of_objectives):
            xx = 3 * f[j] + 1
            prd *= xx
        return prd

    def get_name(self):
        return 'SDP1'


class SDP2(SDP):
    """ Problem SDP2.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 100.
    """

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(SDP2, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)']

        self.lower_bound = self.number_of_variables * [-1.0]
        self.upper_bound = self.number_of_variables * [1.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        g = self.__eval_g(solution)
        h = self.__eval_h(solution)

        for j in range(self.number_of_objectives-1):
            xx = 3 * solution.variables[j] + 1
            solution.objectives[j] = (1 + g)*(h - xx + self.time) / xx
        solution.objectives[self.number_of_objectives-1] = (1+g)*(h-1)/(1+self.time)
        return solution

    def __eval_g(self, solution: FloatSolution):
        g = 0
        x1 = 3 * solution.variables[0] + 1
        for i in range(self.number_of_objectives-1, self.number_of_variables):
            xx = 2 * (solution.variables[i] - 0.5) - cos(self.time+2*x1)
            g += xx*xx
        g *= sin(pi*x1/8)

        return g

    def __eval_h(self, f: float) -> float:
        sm = 0
        for j in range(self.number_of_objectives-1):
            xx = 3 * f.variables[j] + 1
            sm += xx
        sm += 1
        return sm

    def get_name(self):
        return 'SDP2'
