from abc import ABC, abstractmethod
from math import floor
import numpy as np

from jmetal.core.problem import DynamicProblem, FloatProblem
from jmetal.core.solution import FloatSolution


def fix_numerical_instability(x):
    """Check whether x is close to zero, sqrt(0.5) or not. If it is close to
    these two values, changes x to the value. Otherwise, return x.
    """
    if np.allclose(0.0, x):
        return 0.0

    if np.allclose(np.sqrt(0.5), x):
        return np.sqrt(0.5)
    return x


def additive(alpha, beta):
    """Additive form of the benchmark problem.
    """
    return [a + b for a, b in zip(alpha, beta)]


def multiplicative(alpha, beta):
    """Multiplicative form of the benchmark problem.
    """
    return [a * (1 + b) for a, b in zip(alpha, beta)]


def beta_uni(x, t, g, lower_bound, obj_num=2):
    """This function is used to calculate the unimodal beta function. Input are
    the decision variable (x), time (t) and g function (g).
    """
    beta = [0.0] * obj_num
    for i in range(obj_num - 1, len(x)):
        beta[(i + 1) % obj_num] += (x[i] - g(x, t)) * (x[i] - g(x, t))

    beta = [(2.0 / int(len(lower_bound) / obj_num)) * b for b in beta]
    return beta


def beta_multi(x, t, g, lower_bound, obj_num=2):
    """This function is used to calculate the multi-modal beta function. Input
    are the decision variable (x), time (t) and g function (g).
    """
    beta = [0.0] * obj_num
    for i in range(obj_num - 1, len(x)):
        beta[(i + 1) % obj_num] += (x[i] - g(x, t)) * (x[i] - g(x, t)) * \
                                   (1 + np.abs(np.sin(4 * np.pi * (x[i] - g(x, t)))))

    beta = [(2.0 / int(len(lower_bound) / obj_num)) * b for b in beta]
    return beta


def alpha_conv(x):
    """This function is used to calculate the alpha function with convex POF.
    Input is decision variable (x).
    """
    return [x[0], 1 - np.sqrt(x[0])]


def alpha_disc(x):
    """This function is used to calculate the alpha function with discrete POF.
    Input is decision variable (x).
    """
    return [x[0], 1.5 - np.sqrt(x[0]) - 0.5 * np.sin(10 * np.pi * x[0])]


def g(x, t):
    """This function is used to calculate the g function used in the paper.
    Input are decision variable (x) and time (t).
    """
    return np.sin(0.5 * np.pi * (t - x[0]))


def check_boundary(x, upper_bound, lower_bound):
    """Check the dimension of x and whether it is in the decision boundary. x is
    decision variable, upper_bound and lower_bound are upperbound and lowerbound
    lists of the decision space
    """
    if len(x) != len(upper_bound) or len(x) != len(lower_bound):
        return False

    output = True
    for e, upp, low in zip(x, upper_bound, lower_bound):
        output = output and (e >= low) and (e <= upp)
    return output


class GTA(DynamicProblem, FloatProblem, ABC):

    def __init__(self):
        super(GTA, self).__init__()
        self.tau_T = 5
        self.nT = 10
        self.time = 1.0
        self.problem_modified = False
        self.DELTA_STATE = 1

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

    def beta_mix(self, x, t, g, lower_bound, obj_num=2):
        """This function is used to calculate the mixed unimodal and multi-modal
        beta function. Input are the decision variable (x), time (t) and g function
        (g).
        """
        beta = [0.0] * obj_num
        k = int(abs(5.0 * (int(self.DELTA_STATE * int(t) / 5.0) % 2) - (self.DELTA_STATE * int(t) % 5)))

        for i in range(obj_num - 1, len(x)):
            temp = 1.0 + (x[i] - g(x, t)) * (x[i] - g(x, t)) - np.cos(2 * np.pi * k * (x[i] - g(x, t)))
            beta[(i + 1) % obj_num] += temp
        beta = [(2.0 / int(len(lower_bound) / obj_num)) * b for b in beta]
        return beta

    def alpha_mix(self, x, t):
        """This function is used to calculate the alpha function with mixed
        continuous POF and discrete POF.
        """
        k = int(abs(5.0 * (int(self.DELTA_STATE * int(t) / 5.0) % 2) - (self.DELTA_STATE * int(t) % 5)))
        return [x[0], 1 - np.sqrt(x[0]) + 0.1 * k * (1 + np.sin(10 * np.pi * x[0]))]

    def alpha_conf(self, x, t):
        """This function is used to calculate the alpha function with time-varying
        conflicting objective. Input are decision variable (x) and time (t).
        """
        k = int(abs(5.0 * (int(self.DELTA_STATE * int(t) / 5.0) % 2) - (self.DELTA_STATE * int(t) % 5)))
        return [x[0], 1 - np.power(x[0], np.log(1 - 0.1 * k) / np.log(0.1 * k + np.finfo(float).eps))]

    def alpha_conf_3obj_type1(self, x, t):
        """This function is used to calculate the alpha unction with time-varying
        conflicting objective (3-objective, type 1). Input are decision variable
        (x) and time (t).
        """
        k = int(abs(5.0 * (int(self.DELTA_STATE * int(t) / 5.0) % 2) - (self.DELTA_STATE * int(t) % 5)))
        alpha1 = fix_numerical_instability(np.cos(0.5 * x[0] * np.pi) * np.cos(0.5 * x[1] * np.pi))
        alpha2 = fix_numerical_instability(np.cos(0.5 * x[0] * np.pi) * np.sin(0.5 * x[1] * np.pi))
        alpha3 = fix_numerical_instability(np.sin(0.5 * x[0] * np.pi + 0.25 * (k / 5.0) * np.pi))
        return [alpha1, alpha2, alpha3]

    def alpha_conf_3obj_type2(self, x, t):
        """This function is used to calculate the alpha unction with time-varying
        conflicting objective (3-objective, type 2). Input are decision variable (x)
        and time (t).
        """
        k = int(abs(5.0 * (int(self.DELTA_STATE * int(t) / 5.0) % 2) - (self.DELTA_STATE * int(t) % 5)))
        k_ratio = (5.0 - k) / 5.0
        alpha1 = fix_numerical_instability(np.cos(0.5 * x[0] * np.pi) * np.cos(0.5 * x[1] * np.pi * k_ratio))
        alpha2 = fix_numerical_instability(np.cos(0.5 * x[0] * np.pi) * np.sin(0.5 * x[1] * np.pi * k_ratio))
        alpha3 = fix_numerical_instability(np.sin(0.5 * x[0] * np.pi))
        return [alpha1, alpha2, alpha3]


class GTA1a(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA1a, self).__init__()
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
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = alpha_conv(solution.variables)
            beta = beta_uni(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = additive(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA1'


class GTA1m(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA1m, self).__init__()
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
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = alpha_conv(solution.variables)
            beta = beta_uni(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = multiplicative(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA2'


class GTA2a(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA2a, self).__init__()
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
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = alpha_conv(solution.variables)
            beta = beta_multi(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = additive(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA3'


class GTA2m(GTA):

    def __init__(self, number_of_variables: int = 100):
        super(GTA2m, self).__init__()
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
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = alpha_conv(solution.variables)
            beta = beta_multi(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = multiplicative(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA4'


class GTA3a(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA3a, self).__init__()
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
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = alpha_conv(solution.variables)
            beta = self.beta_mix(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = additive(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA5'


class GTA3m(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA3m, self).__init__()
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
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = alpha_conv(solution.variables)
            beta = self.beta_mix(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = multiplicative(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA6'


class GTA4a(GTA):
    """ Problem FDA1.
    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 100.
    """

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA4a, self).__init__()
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
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = alpha_disc(solution.variables)
            beta = self.beta_mix(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = additive(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA1'


class GTA4m(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA4m, self).__init__()
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
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = alpha_disc(solution.variables)
            beta = self.beta_mix(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = multiplicative(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA8'


class GTA5a(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA5a, self).__init__()
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
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = self.alpha_mix(solution.variables, self.time)
            beta = beta_multi(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = additive(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA9'


class GTA5m(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA5m, self).__init__()
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
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = self.alpha_mix(solution.variables, self.time)
            beta = beta_multi(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = multiplicative(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA10'


class GTA6a(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA6a, self).__init__()
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
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = self.alpha_mix(solution.variables, self.time)
            beta = self.beta_mix(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = additive(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA11'


class GTA6m(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA6m, self).__init__()
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
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = self.alpha_mix(solution.variables, self.time)
            beta = self.beta_mix(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = multiplicative(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA12'


class GTA7a(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA7a, self).__init__()
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
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = self.alpha_conf(solution.variables, self.time)
            beta = beta_multi(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = additive(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA13'


class GTA7m(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA7m, self).__init__()
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
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = self.alpha_conf(solution.variables, self.time)
            beta = beta_multi(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = multiplicative(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA14'


class GTA8a(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA8a, self).__init__()
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
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = self.alpha_conf(solution.variables, self.time)
            beta = self.beta_mix(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = additive(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA15'


class GTA8m(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA8m, self).__init__()
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
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = self.alpha_conf(solution.variables, self.time)
            beta = self.beta_mix(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = multiplicative(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA16'


class GTA9a(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA9a, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)', 'f(z)']

        self.lower_bound = self.number_of_variables * [-1.0]
        self.upper_bound = self.number_of_variables * [1.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = self.alpha_conf_3obj_type1(solution.variables, self.time)
            beta = beta_multi(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = additive(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA17'


class GTA9m(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA9m, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)', 'f(z)']

        self.lower_bound = self.number_of_variables * [-1.0]
        self.upper_bound = self.number_of_variables * [1.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = self.alpha_conf_3obj_type1(solution.variables, self.time)
            beta = beta_multi(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = multiplicative(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA18'


class GTA10a(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA10a, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)', 'f(z)']

        self.lower_bound = self.number_of_variables * [-1.0]
        self.upper_bound = self.number_of_variables * [1.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = self.alpha_conf_3obj_type1(solution.variables, self.time)
            beta = self.beta_mix(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = additive(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA19'


class GTA10m(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA10m, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)', 'f(z)']

        self.lower_bound = self.number_of_variables * [-1.0]
        self.upper_bound = self.number_of_variables * [1.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = self.alpha_conf_3obj_type1(solution.variables, self.time)
            beta = self.beta_mix(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = multiplicative(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA20'


class GTA11a(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA11a, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)', 'f(z)']

        self.lower_bound = self.number_of_variables * [-1.0]
        self.upper_bound = self.number_of_variables * [1.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = self.alpha_conf_3obj_type2(solution.variables, self.time)
            beta = beta_multi(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = additive(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA21'


class GTA11m(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA11m, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)', 'f(z)']

        self.lower_bound = self.number_of_variables * [-1.0]
        self.upper_bound = self.number_of_variables * [1.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = self.alpha_conf_3obj_type2(solution.variables, self.time)
            beta = beta_multi(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = multiplicative(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA22'


class GTA12a(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA12a, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)', 'f(z)']

        self.lower_bound = self.number_of_variables * [-1.0]
        self.upper_bound = self.number_of_variables * [1.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = self.alpha_conf_3obj_type2(solution.variables, self.time)
            beta = self.beta_mix(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = additive(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA23'


class GTA12m(GTA):

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(GTA12m, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 3
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['f(x)', 'f(y)', 'f(z)']

        self.lower_bound = self.number_of_variables * [-1.0]
        self.upper_bound = self.number_of_variables * [1.0]
        self.lower_bound[0] = 0.0
        self.upper_bound[0] = 1.0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        if check_boundary(solution.variables, self.upper_bound, self.lower_bound):
            alpha = self.alpha_conf_3obj_type2(solution.variables, self.time)
            beta = self.beta_mix(solution.variables, self.time, g, self.lower_bound, obj_num=self.number_of_objectives)
            solution.objectives[0], solution.objectives[1] = multiplicative(alpha, beta)
            return solution
        else:
            raise Exception("Error")

    def get_name(self):
        return 'GTA24'
