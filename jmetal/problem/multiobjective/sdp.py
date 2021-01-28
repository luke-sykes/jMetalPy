from abc import ABC, abstractmethod
from math import floor, sqrt, pow, sin, pi, cos, fabs, acos
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

    def SafeAcos(self, x: float):
        if x < -1.0:
            x = -1
        elif x > 1.0:
            x = 1
        return acos(x)

    def SafeCot(self, x: float):
        if sin(x) == 0:
            return float("inf")
        else:
            return cos(x) / sin(x)

    def calculate_G(self):
        return sin(0.5*pi*self.time)
"""
    def rnd_uni(self, idum):
        
        #the random generator in [0,1)
        
        idum2 = 123456789
        iy = 0

        if idum <= 0:
            if -(idum) < 1:
                idum = 1
            else
                idum = -1 * idum
            idum2 = idum
            for j in range(NTAB+7, 0, -1):
                k = idum / IQ1
                idum = IA1 * (idum-k*IQ1)-k*IR1
                if (idum < 0):
                    idum += IM1
                if j < NTAB:
                    iv[j] = idum
            iy = iv[0]
        k = idum / IQ1
        idum = IA1 * (idum-k*IQ1)-k*IR1
        if idum < 0:
            idum += IM1
        k = idum2/IQ2
        idum2 = IA2*(idum2-k*IQ2)-k*IR2
        if idum2 < 0:
            idum2 += IM2
        j = iy / NDIV
        iy = iv[j] - idum2
        iv[j] = idum
        if iy < 1:
            iy += IMM1
        if ((temp=AM*iy) > RNMX):
            return RNMX
        else:
            return temp
    """


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


class SDP3(SDP):
    """ Problem SDP3.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 100.
    """

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(SDP3, self).__init__()
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
        pd = 1
        for j in range(self.number_of_objectives-1):
            solution.objectives[j] = (1 + g) * (1 - solution.variables[j] + 0.05 * sin(6 * pi * solution.variables[j])) * pd
            pd *= solution.variables[j] + 0.05 * sin(6 * pi * solution.variables[j])
        solution.objectives[self.number_of_objectives-1] = (1 + g) * pd

        return solution

    def __eval_g(self, solution: FloatSolution):
        g = 0
        pt = floor(5 * fabs(sin(pi*self.time)))
        for i in range(self.number_of_objectives-1, self.number_of_variables):
            y = 2 * (solution.variables[i] - 0.5) - cos(self.time)
            g += 4 * pow(y, 2) - cos(2 * pt * pi * y) + 1

        return g

    def get_name(self):
        return 'SDP3'


class SDP4(SDP):
    """ Problem SDP4.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 100.
    """

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(SDP4, self).__init__()
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
        w = rnd_rt - 0.5
        if w > 0:
            w = 1.0
        elif w < 0:
            w = -1.0
        w = w * floor(6 * fabs(self.calculate_G()))

        g = self.__eval_g(solution)

        sm = 0
        for j in range(self.number_of_objectives-2):
            solution.objectives[j] = solution.variables[j]
            sm += solution.variables[j]

        sm += solution.variables[self.number_of_objectives-2]
        sm = sm / (self.number_of_objectives - 1)

        solution.objectives[self.number_of_objectives-2] = (1 + g) * (sm + 0.05 * sin(w * pi * sm))
        solution.objectives[self.number_of_objectives-1] = (1 + g) * (1.0 - sm + 0.05 * sin(w * pi * sm))

        return solution

    def __eval_g(self, solution: FloatSolution):
        g = 0

        for i in range(self.number_of_objectives-1, self.number_of_variables):
            y = 2 * (solution.variables[i] - 0.5) - cos(self.time + solution.variables[i-1] + solution.variables[0])
            g += y * y

        return g

    def get_name(self):
        return 'SDP4'


class SDP5(SDP):
    """ Problem SDP5.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 100.
    """

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(SDP5, self).__init__()
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
        Gt = fabs(self.calculate_g())
        g = self.__eval_g(solution, Gt)

        pd = 1
        for j in range(self.number_of_objectives-1):
            y = pi * Gt / 6 + (pi / 2 - pi * Gt / 3)*solution.variables[j]
            solution.objectives[j] (1 + g) * sin(y) * pd
            pd *= cos(y)
        solution.objectives[self.number_of_objectives-1] = (1+g) * pd
        return solution

    def __eval_g(self, solution: FloatSolution, Gt: float):
        g = 0
        for i in range(self.number_of_objectives-1, self.number_of_variables):
            y = solution.variables[i] - 0.5*Gt*solution.variables[0]
            g += y*y

        return g + Gt

    def get_name(self):
        return 'SDP5'


class SDP6(SDP):
    """ Problem SDP6.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 100.
    """

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(SDP6, self).__init__()
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
        at = 0.5 * fabs(sin(pi * self.time))
        pt = 10 * cos(2.5 * pi * self.time)

        g = self.__eval_g(solution)

        pd = 1
        for j in range(self.number_of_objectives-1):
            solution.objectives[self.number_of_objectives - 1 - j] = (1 + g) * sin(0.5 * pi * solution.variables[j]) * pd
            pd *= cos(0.5 * pi * solution.variables[j])
        solution.objectives[0] = (1+g) * pd

        if solution.variables[0] < at:
            solution.objectives[self.number_of_objectives - 1] = (1 + g) * fabs(pt * (cos(0.5 * pi * solution.variables[0]) - cos(0.5*pi*at)) + sin(0.5*pi*at))
        #else:
            #solution.objectives[self.number_of_objectives-1] = (1 + g) * sin(0.5 * pi * solution.variables[j])
        return solution

    def __eval_g(self, solution: FloatSolution):
        g = 0
        for i in range(self.number_of_objectives-1, self.number_of_variables):
            y = solution.variables[i] = 0.5
            g += y*y * (1+ fabs(cos(8 * pi * solution.variables[i])))

        return g

    def get_name(self):
        return 'SDP6'


class SDP7(SDP):
    """ Problem SDP7.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 100.
    """

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(SDP7, self).__init__()
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
        at = 0.5 * sin(pi*self.time)

        g = self.__eval_g(solution)
        pd = 1
        for j in range(self.number_of_objectives-1):
            solution.objectives[j] = (0.5 + g) * (1 - solution.number_of_variables[j]) * pd
            pd *= solution.variables[j]
        solution.objectives[self.number_of_objectives-1] = (0.5 + g)*pd
        return solution

    def __eval_g(self, solution: FloatSolution):

        def minPeaksOfSDP7(x, pt):
            if pt < 1:
                pt = 1
            if pt > 5:
                pt = 5
            minimum = float("inf")
            for i in range(1, 6):
                if i == pt:
                    tmp = 0.5 + 10 * pow(10 * x - 2 * (i - 1), 2.0)
                else:
                    tmp = i + 10 * pow(10 * x - 2 * (i - 1), 2.0)
                if tmp <= minimum:
                    minimum = tmp
            return minimum
        pt = 1 + floor(5 * rnd_rt)
        g = 0
        for i in range(self.number_of_objectives-1, self.number_of_variables):
            g += minPeaksOfSDP7(solution.variables[i], pt)
        g /= (self.number_of_variables - self.number_of_objectives + 1)

        return g

    def get_name(self):
        return 'SDP7'


class SDP8(SDP):
    """ Problem SDP8.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 100.
    """

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(SDP8, self).__init__()
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
        Gt = fabs(self.calculate_g())
        pt = floor(6 * Gt)
        g = self.__eval_g(solution, Gt)

        sm = 0
        for j in range(self.number_of_objectives-1):
            solution.objectives[j] = (1 + g) * pow(cos(0.5 * pi * solution.variables[j]), 2.0) + Gt
            sm += pow(sin(0.5 * pi * solution.variables[j]), 2.0) + sin(0.5*pi*solution.variables[j]) * pow(cos(pt*pi*solution.variables[j]), 2.0)

        solution.objectives[self.number_of_objectives-1] = sm + Gt
        return solution

    def __eval_g(self, solution: FloatSolution, Gt: float):
        g = 0
        for i in range(self.number_of_objectives-1, self.number_of_variables):
            y = self.number_of_variables[i] - fabs(atan(SafeCot(3 * pi*Tt*Tt))) / pi
            g += y*y
        g += Gt

        return g

    def get_name(self):
        return 'SDP8'


class SDP9(SDP):
    """ Problem SDP9.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 100.
    """

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(SDP9, self).__init__()
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
        kt = 10 * sin(pi * self.time)

        pd = 1
        for j in range(self.number_of_objectives-1):
            pd *= sin(floor(kt*(2*solution.variables[j] - 1.0)) * pi / 2)

        g = pd + self.__eval_g(solution)

        pd = 1
        for j in range(self.number_of_objectives-1):
            solution.objectives[self.number_of_objectives - 1 - j] = (1 + g) * sin(0.5 * pi * solution.number_of_variables[j]) * pd
            pd *= cos(0.5*pi*solution.variables[j])

        solution.objectives[0] = (1 + g) * pd
        return solution

    def __eval_g(self, solution: FloatSolution):
        g = 0
        for i in range(self.number_of_objectives-1, self.number_of_variables):
            y = 2 * (solution.variables[i] - 0.5) - sin(self.time * solution.variables[0])
            g += y*y

        return g

    def get_name(self):
        return 'SDP9'


class SDP10(SDP):
    """ Problem SDP10.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 100.
    """

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(SDP10, self).__init__()
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
        r = floor(10 * fabs(self.calculate_G()))

        sm = 0
        for j in range(self.number_of_objectives-1):
            sm += pow(self.number_of_variables[j], 2.0)
        sm /= self.number_of_objectives - 1

        g = self.__eval_g(solution)

        for j in range(self.number_of_objectives-1):
            solution.objectives[j] = solution.variables[j]

        solution.objectives[self.number_of_objectives-1] = (1+g) * (2 - sm - pow(sm, 0.5))*pow(-sin(2.5*pi*sm), r)
        return solution

    def __eval_g(self, solution: FloatSolution):
        g = 0
        for i in range(self.number_of_objectives-1, self.number_of_variables):
            y = 2 * (solution.variables[i] - 0.5) - sin(0.5*pi*self.time+solution.variables[0])
            g += y*y

        return g

    def get_name(self):
        return 'SDP10'


class SDP11(SDP):
    """ Problem SDP11.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 100.
    """

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(SDP11, self).__init__()
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
        # sm = 0
        """for j in range(self.number_of_objectives-1):
                sm += solution.variables[j]
                sm /= (self.number_of_objectives - 1)"""

        g = self.__eval_g(solution)
        pd = 1
        for j in range(self.number_of_objectives-1):
            yj = 0.5 * pi * solution.variables[j]
            solution.objectives[j] = (1+g)*sin(yj)*pd
            pd *= cos(yj)
        solution.objectives[self.number_of_objectives-1] = (1+g)*pd

        return solution

    def __eval_g(self, solution: FloatSolution):
        at = 3 * self.time - floor(3 * self.time)
        bt = 3 * self.time + 0.2 - floor(3 * self.time + 0.2)
        g = 0
        ps = 0
        for i in range(self.number_of_objectives-1):
            ps += solution.variables[i]

        if at <= ps < bt:
            for i in range(self.number_of_objectives-1, self.number_of_variables):
                p = solution.variables[i] - fabs(G)
                g += -0.9 * p * p + pow(fabs(p), 0.6)
        else:
            for i in range(self.number_of_objectives-1, self.number_of_variables):
                p = solution.variables[i] - fabs(G)
                g += p * p

        return g

    def get_name(self):
        return 'SDP11'


class SDP12(SDP):
    """ Problem SDP12.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 100.
    """

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(SDP12, self).__init__()
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
        # int nvar = nvar + floor(rnd_rt*(nvar_u - nvar_l))

        g = self.__eval_g(solution)
        pd = 1
        for j in range(self.number_of_objectives-1):
            solution.objectives[j] = (1.0 + g)*(1 - solution.variables[j])*pd
            pd *= solution.variables[j]

        solution.objectives[self.number_of_objectives-1] = (1 + g)*pd
        return solution

    def __eval_g(self, solution: FloatSolution):
        g = 0
        for i in range(self.number_of_objectives-1, self.number_of_variables):
            y = 2 * (solution.variables[i] - 0.5) - sin(self.time)*sin(2 * pi * solution.variables[1])
            g += y*y
        return g

    def get_name(self):
        return 'SDP12'


class SDP13(SDP):
    """ Problem SDP13.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 100.
    """

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(SDP13, self).__init__()
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
        randVal = self.number_of_objectives  # save this random value

        g = self.__eval_g(solution)
        pd = 1
        for j in range(self.number_of_objectives):
            yj = pi * (solution.variables[j] + 1) / 6
            solution.objectives[j] = (1 + g) * sin(yj) * pd
            pd *= cos(yj)

        for j in range(self.number_of_objectives, len(solution.objectives)):
            solution.objectives[j] = 0

        return solution

    def __eval_g(self, solution: FloatSolution):
        g = 0
        for i in range(self.number_of_objectives-1, self.number_of_variables):
            y = solution.number_of_variables[i] - (i*self.time)/(self.number_of_objectives+i/self.time)
            g += pow(y, 2.0)
        return g

    def get_name(self):
        return 'SDP13'


class SDP14(SDP):
    """ Problem SDP14.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 100.
    """

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(SDP14, self).__init__()
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

        pt = 1 + floor(fabs((self.number_of_objectives - 2)*cos(0.5*pi*self.time)))

        g = self.__eval_g(solution)
        pd = 1
        for j in range(self.number_of_objectives-1):
            yj = solution.variables[j]
            if j >= pt:
                yj = 0.5 + solution.variables[j] * g * fabs(G)
            solution.objectives[j] = (1+g)*(1+g-yj)*pd
            pd *= yj

        solution.objectives[self.number_of_objectives-1] = (1 + g)*pd
        return solution

    def __eval_g(self, solution: FloatSolution):
        g = 0
        for i in range(self.number_of_objectives-1, self.number_of_variables):
            g += pow(solution.variables[i] - 0.5, 2.0)
        return g

    def get_name(self):
        return 'SDP14'


class SDP15(SDP):
    """ Problem SDP15.

    .. note:: Bi-objective dynamic unconstrained problem. The default number of variables is 100.
    """

    def __init__(self, number_of_variables: int = 100):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(SDP15, self).__init__()
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
        dt = randVal
        pt = randVal2

        g = self.__eval_g(solution, dt)

        yk = [0 for i in range(self.number_of_objectives-1)]
        for i in range(self.number_of_objectives):
            k = (pt+i-1) % (self.number_of_objectives-1)
            if k <= dt:
                yk[k] = 0.5 * pi * solution.variables[k]
            else:
                yk[k] = SafeAcos(1.0 / (pow(2.0, 0.5)*(1.0 + solution.variables[k])*g*fabs(G)))

        pd = 1
        for j in range(self.number_of_objectives-1):
            solution.objectives[j] = pow(1 + g, j+1)*sin(yk[j])*pd
            pd *= cos(yk[j])
        solution.objectives[self.number_of_objectives-1] = pow(1+g, self.number_of_objectives)*pd

        return solution

    def __eval_g(self, solution: FloatSolution, dt: float):
        g = 0
        for i in range(self.number_of_objectives-1, self.number_of_variables):
            y = solution.variables[i] - dt / (solution.number_of_objectives - 1)
            g += pow(y, 2.0)
        return g

    def get_name(self):
        return 'SDP15'