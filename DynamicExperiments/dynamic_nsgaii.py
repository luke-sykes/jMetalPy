from jmetal.algorithm.multiobjective.nsgaii import DynamicNSGAII
from jmetal.core.problem import DynamicProblem
from jmetal.core.quality_indicator import GenerationalDistance, EpsilonIndicator, HyperVolume
from jmetal.lab.experiment import Job, Experiment, generate_summary_from_experiment
from jmetal.lab.visualization import Plot
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem.multiobjective.fda import FDA2, FDA1, FDA3, FDA4, FDA5
from jmetal.problem.multiobjective.gta import GTA1a
from jmetal.problem.multiobjective.sdp import SDP2
from jmetal.util.observable import TimeCounter
from jmetal.util.observer import PlotFrontToFileObserver, WriteFrontToFileObserver, ProgressBarObserver, \
    VisualizerObserver, BasicObserver
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations


def configure_experiment(problems: dict, n_run: int):
    jobs = []
    max_evaluations = 25000

    for run in range(n_run):
        for problem_tag, problem in problems.items():
            jobs.append(
                Job(
                    algorithm=DynamicNSGAII(
                        problem=problem,
                        population_size=100,
                        offspring_population_size=100,
                        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
                                                    distribution_index=20),
                        crossover=SBXCrossover(probability=1.0, distribution_index=20),
                        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
                    ),
                    algorithm_tag='DynamicNSGAII',
                    problem_tag=problem_tag,
                    run=run,
                )
            )
    return jobs


def run_DynamicNSGAII(problems):

    for problem in problems:

        time_counter = TimeCounter(delay=1)
        time_counter.observable.register(problem[1])
        time_counter.start()

        max_evaluations = 25000
        algorithm = DynamicNSGAII(
            problem=problem[1],
            population_size=100,
            offspring_population_size=100,
            mutation=PolynomialMutation(probability=1.0 / problem[1].number_of_variables, distribution_index=20),
            crossover=SBXCrossover(probability=1.0, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )

        algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
        algorithm.observable.register(observer=VisualizerObserver())
        algorithm.observable.register(observer=PlotFrontToFileObserver(problem[0] + "_dynamic_front_vis"))
        algorithm.observable.register(observer=WriteFrontToFileObserver(problem[0] + "_dynamic_front"))
        #algorithm.observable.register(observer=BasicObserver())

        algorithm.run()
        front = algorithm.get_result()

        non_dominated_solutions = get_non_dominated_solutions(front)

        # save to files
        print_function_values_to_file(front, 'FUN.DYNAMICNSGAII.' + problem[0])
        print_variables_to_file(front, 'VAR.DYNAMICNSGAII.' + problem[0])

        # Plot
        plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y'])
        plot_front.plot(front, label='DynamicNSGAII-FDA2', filename='DYNAMICNSGAII-'+problem[0], format='png')


if __name__ == '__main__':
    problems = [("FDA1", FDA1())]
    run_DynamicNSGAII(problems)

"""
    # Configure the experiments
    jobs = configure_experiment(problems={'FDA1': FDA1(), 'FDA2': FDA2(), 'FDA3': FDA3(), 'FDA4': FDA4(), 'FDA5': FDA5()}, n_run=1)

    # Run the study
    output_directory = 'NSGAII_FDA_data'

    experiment = Experiment(output_dir=output_directory, jobs=jobs)
    experiment.run()

    # Generate summary file
    generate_summary_from_experiment(
        input_dir=output_directory,
        reference_fronts='resources/reference_front',
        quality_indicators=[GenerationalDistance(), HyperVolume([1.0, 1.0])]
    )
    """