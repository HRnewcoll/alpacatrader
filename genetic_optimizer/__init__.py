"""
Genetic Algorithm Optimizer for Strategy Parameters
Evolutionary optimization inspired by DEAP and QuantConnect
Automatically finds optimal strategy parameters through natural selection
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from functools import partial


@dataclass
class Gene:
    """Represents a single gene (parameter) in the genome"""
    name: str
    value: float
    min_value: float
    max_value: float
    mutation_rate: float = 0.1
    
    def mutate(self) -> 'Gene':
        """Mutate gene value"""
        if random.random() < self.mutation_rate:
            # Gaussian mutation
            std_dev = (self.max_value - self.min_value) * 0.1
            new_value = self.value + np.random.normal(0, std_dev)
            new_value = np.clip(new_value, self.min_value, self.max_value)
            return Gene(
                name=self.name,
                value=new_value,
                min_value=self.min_value,
                max_value=self.max_value,
                mutation_rate=self.mutation_rate,
            )
        return self
    
    @staticmethod
    def random(name: str, min_value: float, max_value: float, mutation_rate: float = 0.1) -> 'Gene':
        """Create random gene"""
        value = np.random.uniform(min_value, max_value)
        return Gene(
            name=name,
            value=value,
            min_value=min_value,
            max_value=max_value,
            mutation_rate=mutation_rate,
        )


@dataclass
class Individual:
    """Represents an individual in the population (set of strategy parameters)"""
    genes: List[Gene]
    fitness: float = 0.0
    generation: int = 0
    id: int = field(default_factory=lambda: random.randint(0, 10**9))
    
    def get_params(self) -> Dict[str, float]:
        """Get parameters as dictionary"""
        return {gene.name: gene.value for gene in self.genes}
    
    def copy(self) -> 'Individual':
        """Create copy of individual"""
        return Individual(
            genes=[Gene(
                name=g.name,
                value=g.value,
                min_value=g.min_value,
                max_value=g.max_value,
                mutation_rate=g.mutation_rate,
            ) for g in self.genes],
            fitness=self.fitness,
            generation=self.generation,
            id=self.id,
        )


class GeneticOptimizer:
    """
    Genetic Algorithm Optimizer for trading strategy parameters
    Implements: Tournament selection, BLX-α crossover, Gaussian mutation
    """
    
    def __init__(
        self,
        gene_definitions: List[Dict],
        population_size: int = 50,
        generations: int = 100,
        elitism_rate: float = 0.1,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        tournament_size: int = 5,
        parallel: bool = True,
        maximize: bool = True,
    ):
        self.gene_definitions = gene_definitions
        self.population_size = population_size
        self.generations = generations
        self.elitism_rate = elitism_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.parallel = parallel
        self.maximize = maximize
        
        self.population: List[Individual] = []
        self.history: List[Dict] = []
        self.best_individual: Optional[Individual] = None
        self.generation = 0
        
        # Initialize genes
        self.genes = [
            Gene.random(
                name=g['name'],
                min_value=g['min'],
                max_value=g['max'],
                mutation_rate=g.get('mutation_rate', mutation_rate),
            )
            for g in gene_definitions
        ]
        
    def create_initial_population(self) -> List[Individual]:
        """Create initial random population"""
        population = []
        for _ in range(self.population_size):
            genes = [
                Gene.random(
                    name=g['name'],
                    min_value=g['min'],
                    max_value=g['max'],
                    mutation_rate=g.get('mutation_rate', self.mutation_rate),
                )
                for g in self.gene_definitions
            ]
            population.append(Individual(genes=genes, generation=0))
        return population
    
    def evaluate_fitness(
        self,
        individual: Individual,
        fitness_function: Callable[[Dict], float],
    ) -> float:
        """Evaluate fitness of individual"""
        params = individual.get_params()
        fitness = fitness_function(params)
        return fitness
    
    def tournament_selection(self, population: List[Individual]) -> Individual:
        """Select individual using tournament selection"""
        tournament = random.sample(population, self.tournament_size)
        if self.maximize:
            return max(tournament, key=lambda ind: ind.fitness)
        else:
            return min(tournament, key=lambda ind: ind.fitness)
    
    def blx_alpha_crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        alpha: float = 0.5,
    ) -> Tuple[Individual, Individual]:
        """
        BLX-α crossover operator
        Creates offspring with values in extended range between parents
        """
        child1_genes = []
        child2_genes = []
        
        for g1, g2 in zip(parent1.genes, parent2.genes):
            if random.random() < self.crossover_rate:
                min_val = min(g1.value, g2.value)
                max_val = max(g1.value, g2.value)
                interval = max_val - min_val
                
                # Extended range
                lower = max(g1.min_value, min_val - alpha * interval)
                upper = min(g1.max_value, max_val + alpha * interval)
                
                # Create children
                child1_val = np.random.uniform(lower, upper)
                child2_val = np.random.uniform(lower, upper)
                
                child1_genes.append(Gene(
                    name=g1.name,
                    value=child1_val,
                    min_value=g1.min_value,
                    max_value=g1.max_value,
                    mutation_rate=g1.mutation_rate,
                ))
                child2_genes.append(Gene(
                    name=g2.name,
                    value=child2_val,
                    min_value=g2.min_value,
                    max_value=g2.max_value,
                    mutation_rate=g2.mutation_rate,
                ))
            else:
                # No crossover, copy parents
                child1_genes.append(g1.copy() if hasattr(g1, 'copy') else g1)
                child2_genes.append(g2.copy() if hasattr(g2, 'copy') else g2)
        
        child1 = Individual(genes=child1_genes, generation=self.generation)
        child2 = Individual(genes=child2_genes, generation=self.generation)
        
        return child1, child2
    
    def mutate(self, individual: Individual) -> Individual:
        """Apply mutation to individual"""
        mutated_genes = []
        for gene in individual.genes:
            mutated_gene = gene.mutate()
            mutated_genes.append(mutated_gene)
        
        return Individual(
            genes=mutated_genes,
            generation=self.generation,
        )
    
    def evolve(self, fitness_function: Callable[[Dict], float], verbose: bool = True) -> Individual:
        """
        Run genetic algorithm optimization
        fitness_function: Function that takes parameter dict and returns fitness score
        """
        print(f"🧬 Starting Genetic Optimization")
        print(f"   Population: {self.population_size}, Generations: {self.generations}")
        print(f"   Genes: {[g['name'] for g in self.gene_definitions]}")
        print("-" * 70)
        
        # Initialize population
        self.population = self.create_initial_population()
        
        # Evaluate initial population
        if self.parallel and multiprocessing.cpu_count() > 1:
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = [
                    executor.submit(self.evaluate_fitness, ind, fitness_function)
                    for ind in self.population
                ]
                for ind, future in zip(self.population, futures):
                    ind.fitness = future.result()
        else:
            for ind in self.population:
                ind.fitness = self.evaluate_fitness(ind, fitness_function)
        
        # Track best
        if self.maximize:
            self.best_individual = max(self.population, key=lambda ind: ind.fitness)
        else:
            self.best_individual = min(self.population, key=lambda ind: ind.fitness)
        
        # Evolution loop
        for gen in range(self.generations):
            self.generation = gen + 1
            
            # Elitism: keep top performers
            elite_count = int(self.population_size * self.elitism_rate)
            if self.maximize:
                elites = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)[:elite_count]
            else:
                elites = sorted(self.population, key=lambda ind: ind.fitness)[:elite_count]
            
            # Create new population
            new_population = elites.copy()
            
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(self.population)
                parent2 = self.tournament_selection(self.population)
                
                # Crossover
                child1, child2 = self.blx_alpha_crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to population size
            new_population = new_population[:self.population_size]
            
            # Evaluate new population
            if self.parallel and multiprocessing.cpu_count() > 1:
                with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    futures = [
                        executor.submit(self.evaluate_fitness, ind, fitness_function)
                        for ind in new_population
                    ]
                    for ind, future in zip(new_population, futures):
                        ind.fitness = future.result()
            else:
                for ind in new_population:
                    ind.fitness = self.evaluate_fitness(ind, fitness_function)
            
            self.population = new_population
            
            # Update best
            if self.maximize:
                current_best = max(self.population, key=lambda ind: ind.fitness)
            else:
                current_best = min(self.population, key=lambda ind: ind.fitness)
            
            if (self.maximize and current_best.fitness > self.best_individual.fitness) or \
               (not self.maximize and current_best.fitness < self.best_individual.fitness):
                self.best_individual = current_best.copy()
            
            # Record history
            avg_fitness = np.mean([ind.fitness for ind in self.population])
            std_fitness = np.std([ind.fitness for ind in self.population])
            
            self.history.append({
                'generation': gen + 1,
                'best_fitness': self.best_individual.fitness,
                'avg_fitness': avg_fitness,
                'std_fitness': std_fitness,
                'best_params': self.best_individual.get_params(),
            })
            
            if verbose and (gen + 1) % 10 == 0:
                print(f"   Gen {gen+1:3d}: Best={self.best_individual.fitness:.4f}, "
                      f"Avg={avg_fitness:.4f}±{std_fitness:.4f}")
        
        print("-" * 70)
        print(f"✅ Optimization Complete!")
        print(f"   Best Fitness: {self.best_individual.fitness:.4f}")
        print(f"   Best Parameters:")
        for name, value in self.best_individual.get_params().items():
            print(f"      {name}: {value:.4f}")
        
        return self.best_individual
    
    def get_optimization_results(self) -> Dict:
        """Get comprehensive optimization results"""
        if not self.best_individual:
            return {}
        
        return {
            'best_fitness': self.best_individual.fitness,
            'best_params': self.best_individual.get_params(),
            'generations_run': self.generation,
            'history': self.history,
            'convergence': self._analyze_convergence(),
        }
    
    def _analyze_convergence(self) -> Dict:
        """Analyze convergence characteristics"""
        if len(self.history) < 2:
            return {}
        
        best_fitnesses = [h['best_fitness'] for h in self.history]
        avg_fitnesses = [h['avg_fitness'] for h in self.history]
        
        # Calculate improvement rate
        improvements = np.diff(best_fitnesses)
        if self.maximize:
            improvements = np.where(improvements > 0, improvements, 0)
        else:
            improvements = np.where(improvements < 0, -improvements, 0)
        
        total_improvement = sum(abs(improvements))
        final_improvement = sum(abs(improvements[-10:])) if len(improvements) >= 10 else total_improvement
        
        # Convergence score (0-1, higher = more converged)
        convergence_score = 1.0 - (final_improvement / (total_improvement + 1e-10))
        
        return {
            'convergence_score': convergence_score,
            'total_improvement': total_improvement,
            'final_improvement_rate': final_improvement / max(len(improvements) - 10, 1),
            'generations_to_90pct': self._find_90pct_generation(best_fitnesses),
        }
    
    def _find_90pct_generation(self, best_fitnesses: List[float]) -> int:
        """Find generation where 90% of improvement was achieved"""
        if len(best_fitnesses) < 2:
            return 0
        
        total_improvement = abs(best_fitnesses[-1] - best_fitnesses[0])
        target = best_fitnesses[0] + 0.9 * total_improvement * (1 if self.maximize else -1)
        
        for i, fitness in enumerate(best_fitnesses):
            if (self.maximize and fitness >= target) or (not self.maximize and fitness <= target):
                return i + 1
        
        return len(best_fitnesses)


def demo_genetic_optimizer():
    """Demonstrate genetic optimizer"""
    print("=" * 70)
    print("🧬 GENETIC ALGORITHM OPTIMIZER DEMO")
    print("=" * 70)
    
    # Define genes to optimize (example: moving average crossover strategy)
    gene_definitions = [
        {'name': 'fast_ma_period', 'min': 5, 'max': 50},
        {'name': 'slow_ma_period', 'min': 20, 'max': 200},
        {'name': 'stop_loss_pct', 'min': 0.01, 'max': 0.10},
        {'name': 'take_profit_pct', 'min': 0.02, 'max': 0.20},
        {'name': 'position_size_pct', 'min': 0.05, 'max': 0.50},
    ]
    
    # Define fitness function (simulated backtest)
    def fitness_function(params: Dict) -> float:
        """Simulated fitness based on Sharpe ratio"""
        fast_ma = params['fast_ma_period']
        slow_ma = params['slow_ma_period']
        stop_loss = params['stop_loss_pct']
        take_profit = params['take_profit_pct']
        position_size = params['position_size_pct']
        
        # Simulate strategy performance
        # In real use, this would run actual backtest
        np.random.seed(int(fast_ma * slow_ma * 1000))
        
        # Generate synthetic returns
        n_trades = 100
        win_rate = 0.5 + (slow_ma - fast_ma) / 400  # Better with wider MA spread
        win_rate = np.clip(win_rate, 0.3, 0.7)
        
        avg_win = take_profit * position_size
        avg_loss = -stop_loss * position_size
        
        returns = []
        for _ in range(n_trades):
            if random.random() < win_rate:
                returns.append(avg_win * (1 + np.random.normal(0, 0.2)))
            else:
                returns.append(avg_loss * (1 + np.random.normal(0, 0.2)))
        
        returns = np.array(returns)
        
        # Calculate Sharpe ratio (annualized)
        if returns.std() == 0:
            return 0.0
        
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        
        # Penalize extreme parameters
        penalty = 0.0
        if slow_ma <= fast_ma:
            penalty -= 1.0
        if stop_loss > take_profit:
            penalty -= 0.5
        
        return sharpe + penalty
    
    # Run optimization
    optimizer = GeneticOptimizer(
        gene_definitions=gene_definitions,
        population_size=30,
        generations=50,
        elitism_rate=0.15,
        crossover_rate=0.85,
        mutation_rate=0.2,
        tournament_size=4,
        parallel=True,
        maximize=True,
    )
    
    best = optimizer.evolve(fitness_function, verbose=True)
    
    # Get results
    results = optimizer.get_optimization_results()
    
    print("\n" + "=" * 70)
    print("📊 OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"\nBest Parameters:")
    for param, value in results['best_params'].items():
        print(f"   {param}: {value:.4f}")
    
    print(f"\nConvergence Analysis:")
    conv = results['convergence']
    print(f"   Convergence Score: {conv['convergence_score']:.2f}")
    print(f"   Total Improvement: {conv['total_improvement']:.4f}")
    print(f"   90% Improvement at Gen: {conv['generations_to_90pct']}")
    
    # Plot fitness history (if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        generations = [h['generation'] for h in results['history']]
        best_fitness = [h['best_fitness'] for h in results['history']]
        avg_fitness = [h['avg_fitness'] for h in results['history']]
        
        ax.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
        ax.plot(generations, avg_fitness, 'g--', label='Avg Fitness', linewidth=2)
        ax.fill_between(generations, 
                        [a - s for a, s in zip(avg_fitness, [h['std_fitness'] for h in results['history']])],
                        [a + s for a, s in zip(avg_fitness, [h['std_fitness'] for h in results['history']])],
                        alpha=0.3, color='green')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness (Sharpe Ratio)')
        ax.set_title('Genetic Algorithm Optimization Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/workspace/genetic_optimizer/optimization_progress.png', dpi=150)
        print(f"\n📈 Saved optimization plot to: genetic_optimizer/optimization_progress.png")
        
    except ImportError:
        pass
    
    print("\n" + "=" * 70)
    print("✅ Genetic Optimizer Demo Complete!")
    print("=" * 70)
    
    return optimizer


if __name__ == "__main__":
    demo_genetic_optimizer()
