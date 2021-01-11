from jmetal.algorithm.multiobjective.nsgaii import DynamicNSGAII
from jmetal.core.problem import DynamicProblem
from jmetal.core.quality_indicator import GenerationalDistance, EpsilonIndicator, HyperVolume
from jmetal.lab.experiment import Job, Experiment, generate_summary_from_experiment
from jmetal.lab.visualization import Plot
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem.multiobjective.fda import FDA2
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


def run_DynamicNSGAII():
    problem: DynamicProblem = FDA2()

    time_counter = TimeCounter(delay=1)
    time_counter.observable.register(problem)
    time_counter.start()

    max_evaluations = 25000
    algorithm = DynamicNSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(observer=VisualizerObserver())
    algorithm.observable.register(observer=PlotFrontToFileObserver("dynamic_front_vis"))
    algorithm.observable.register(observer=WriteFrontToFileObserver("dynamic_front"))
    #algorithm.observable.register(observer=BasicObserver())

    algorithm.run()
    front = algorithm.get_result()

    non_dominated_solutions = get_non_dominated_solutions(front)

    # save to files
    print_function_values_to_file(front, 'FUN.DYNAMICNSGAII.FDA2')
    print_variables_to_file(front, 'VAR.DYNAMICNSGAII.FDA2')

    # Plot
    #plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y'])
    #plot_front.plot(non_dominated_solutions, label='DynamicNSGAII-FDA2', filename='DYNAMICNSGAII-FDA2', format='png')

    # Plot
    plot_front = Plot(title='Pareto front approximation', axis_labels=['x', 'y'])
    plot_front.plot(front, label='DynamicNSGAII-FDA2', filename='DYNAMICNSGAII-FDA2', format='png')


if __name__ == '__main__':
    run_DynamicNSGAII()
    """
    # Configure the experiments
    jobs = configure_experiment(problems={'FDA2': FDA2()}, n_run=1)

    # Run the study
    output_directory = 'data'

    experiment = Experiment(output_dir=output_directory, jobs=jobs)
    experiment.run()

    # Generate summary file
    generate_summary_from_experiment(
        input_dir=output_directory,
        reference_fronts='resources/reference_front',
        quality_indicators=[GenerationalDistance(), EpsilonIndicator(), HyperVolume([1.0, 1.0])]
    )
    """