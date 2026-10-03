//! Genetic algorithm search optimizer.

use crate::error::{Error, Result};
use crate::optim::chromosome::Chromosome;
use crate::optim::common::threshold_switch;
use crate::optim::OptimDirection;
use crate::random::randf;
use rand::Rng;

const MAX_POPULATION_SIZE: usize = 1024;

/// Genetic-algorithm search over [`Chromosome`]s.
///
/// The search keeps a population ranked by the utility, best first. Each
/// generation replaces every member outside the selected top group with a
/// crossover of two selected members, the best member always being one parent
/// of the first child. It then flips random bits in every member except the
/// best, and ranks the population again. The selected group defaults to a
/// quarter of the population, and at least two.
///
/// The search draws from the thread-local random number generator, so runs
/// are not repeatable. It does not establish that a result is optimal.
#[doc(alias = "gasearch")]
pub struct GeneticSearch<F> {
    num_traits: usize,
    population_size: usize,
    selection_size: usize,
    bits_per_chromosome: u32,
    mutation_rate: f32,

    population: Vec<Chromosome>,
    utility: Vec<f32>,

    c: Chromosome,    // optimum chromosome
    utility_opt: f32, // fitness of the optimum chromosome

    get_utility: F,
    minimize: bool,
}

impl<F> GeneticSearch<F>
where
    F: FnMut(&Chromosome) -> f32,
{
    /// Creates a search with a population of 16 and a mutation rate of
    /// `0.02`.
    ///
    /// See [`Self::new_with_settings`]. The fixed defaults are always valid.
    pub fn new(utility: F, initial_chromosome: &Chromosome, direction: OptimDirection) -> Self {
        Self::new_with_settings(utility, initial_chromosome, direction, 16, 0.02)
            .expect("fixed population size and mutation rate are valid")
    }

    /// Creates a search with the given population size and mutation rate.
    ///
    /// `initial_chromosome` sets the trait widths and becomes the first
    /// member. The other members are random. The utility is called for every
    /// member, and the population is ranked.
    ///
    /// In each generation every member but the best gets one bit flip, and
    /// then further flips while a random draw falls below `mutation_rate`, up
    /// to one flip per bit. A rate of zero therefore still gives one flip.
    ///
    /// # Errors
    ///
    /// Returns an error when `population_size` is outside `2..=1024` or
    /// `mutation_rate` is outside `[0, 1]`.
    pub fn new_with_settings(
        utility: F,
        initial_chromosome: &Chromosome,
        direction: OptimDirection,
        population_size: usize,
        mutation_rate: f32,
    ) -> Result<Self> {
        // validate input
        if population_size < 2 {
            return Err(Error::Config("population size must be at least 2".into()));
        }
        if population_size > MAX_POPULATION_SIZE {
            return Err(Error::Config(format!("population size must be at most {}", MAX_POPULATION_SIZE)));
        }
        if !(0.0..=1.0).contains(&mutation_rate) {
            return Err(Error::Config("mutation rate must be in [0,1]".into()));
        }

        // initialize selection size to 25% of population, minimum of 2
        let selection_size = if (population_size >> 2) < 2 { 2 } else { population_size >> 2 };

        // create optimum chromosome (clone)
        let c = initial_chromosome.clone();

        // create population
        let mut population: Vec<Chromosome> = (0..population_size).map(|_| initial_chromosome.clone()).collect();

        // initialize population to random, preserving first chromosome
        for chromosome in population.iter_mut().skip(1) {
            chromosome.randomize();
        }

        let mut ga = Self {
            num_traits: initial_chromosome.num_traits(),
            population_size,
            selection_size,
            bits_per_chromosome: initial_chromosome.num_bits(),
            mutation_rate,
            population,
            utility: vec![0.0; population_size],
            c,
            utility_opt: 0.0,
            get_utility: utility,
            minimize: direction == OptimDirection::Minimize,
        };

        // evaluate population
        ga.evaluate();

        // rank chromosomes
        ga.rank();

        // set global optimum to the best-ranked chromosome and its utility
        ga.c.copy_from(&ga.population[0]).expect("population chromosomes share trait widths");
        ga.utility_opt = ga.utility[0];

        Ok(ga)
    }

    /// Returns the number of traits in each chromosome.
    #[doc(alias = "num_parameters")]
    pub fn num_traits(&self) -> usize {
        self.num_traits
    }

    /// Sets the population size and the number of top members selected as
    /// parents.
    ///
    /// Growing the population adds copies of the worst member. Shrinking drops
    /// the worst members. The utility is not called.
    ///
    /// # Errors
    ///
    /// Returns an error when `population_size` is outside `2..=1024` or
    /// `selection_size` is outside `1..population_size`. The search is
    /// unchanged on error.
    pub fn configure_population(&mut self, population_size: usize, selection_size: usize) -> Result<()> {
        // validate input
        if population_size < 2 {
            return Err(Error::Config("population size must be at least 2".into()));
        }
        if population_size > MAX_POPULATION_SIZE {
            return Err(Error::Config(format!("population size must be at most {}", MAX_POPULATION_SIZE)));
        }
        if selection_size == 0 {
            return Err(Error::Config("selection size must be greater than zero".into()));
        }
        if selection_size >= population_size {
            return Err(Error::Config("selection size must be less than population size".into()));
        }

        // initialize new chromosomes (copies) when growing
        if population_size > self.population_size {
            let k = self.population_size - 1; // least optimal
            let clone = self.population[k].clone();
            let u_k = self.utility[k];
            for _ in self.population_size..population_size {
                // clone chromosome, copying internal values
                self.population.push(clone.clone());
                // copy utility
                self.utility.push(u_k);
            }
        } else {
            self.population.truncate(population_size);
            self.utility.truncate(population_size);
        }

        // set internal variables
        self.population_size = population_size;
        self.selection_size = selection_size;
        Ok(())
    }

    /// Sets the mutation rate, which controls the extra bit flips described
    /// on [`Self::new_with_settings`].
    ///
    /// # Errors
    ///
    /// Returns an error when `mutation_rate` is outside `[0, 1]`. The rate is
    /// unchanged on error.
    pub fn set_mutation_rate(&mut self, mutation_rate: f32) -> Result<()> {
        if !(0.0..=1.0).contains(&mutation_rate) {
            return Err(Error::Range("mutation rate must be in [0,1]".into()));
        }

        self.mutation_rate = mutation_rate;
        Ok(())
    }

    /// Runs generations until `max_iterations` have run or the best member
    /// reaches `target_utility`, and returns the best utility seen so far.
    ///
    /// Reaching the target means getting to it or below when minimizing, and
    /// to it or above when maximizing. When `max_iterations` is zero, no
    /// generation or utility evaluation runs. Otherwise at least one
    /// generation runs, even when the target is already met.
    ///
    pub fn run(&mut self, max_iterations: usize, target_utility: f32) -> f32 {
        if max_iterations == 0 {
            return self.utility_opt;
        }

        let mut i = 0;
        loop {
            i += 1;
            self.evolve();
            if !(threshold_switch(self.utility[0], target_utility, self.minimize) && i < max_iterations) {
                break;
            }
        }

        // return optimum utility
        self.utility_opt
    }

    /// Runs one generation and updates the best result when it improves.
    ///
    /// The utility is called once for every member. The last member is set to
    /// random values first, but crossover then overwrites it, so no random
    /// member enters the population. The exception is a search created with a
    /// population of two, where both members are selected and there is no
    /// crossover.
    pub fn evolve(&mut self) {
        // Inject random chromosome at end
        self.population[self.population_size - 1].randomize();

        // Crossover
        self.crossover();

        // Mutation
        self.mutate();

        // Evaluation
        self.evaluate();

        // Rank
        self.rank();

        if threshold_switch(self.utility_opt, self.utility[0], self.minimize) {
            // update optimum
            self.utility_opt = self.utility[0];

            // copy optimum chromosome
            self.c.copy_from(&self.population[0]).expect("population chromosomes share trait widths");
        }
    }

    /// Returns the best chromosome found so far and its utility.
    ///
    /// The chromosome and utility correspond even before the first generation
    /// is run.
    pub fn best(&self) -> (&Chromosome, f32) {
        (&self.c, self.utility_opt)
    }

    // evaluate fitness of entire population
    fn evaluate(&mut self) {
        for i in 0..self.population_size {
            self.utility[i] = (self.get_utility)(&self.population[i]);
        }
    }

    // crossover population
    fn crossover(&mut self) {
        let mut rng = rand::thread_rng();
        for i in self.selection_size..self.population_size {
            // ensure fittest member is used at least once as parent
            let i1 = if i == self.selection_size { 0 } else { rng.gen_range(0..self.selection_size) };
            let i2 = rng.gen_range(0..self.selection_size);
            let threshold = rng.gen_range(0..self.bits_per_chromosome);

            // p1/p2 alias earlier population members
            // clone to satisfy the borrow checker
            let p1 = self.population[i1].clone();
            let p2 = self.population[i2].clone();

            self.population[i]
                .crossover(&p1, &p2, threshold)
                .expect("population chromosomes share trait widths");
        }
    }

    // mutate population
    fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        // mutate all but the first (best) chromosome
        for i in 1..self.population_size {
            let mut num_mutations = 0;
            // force at least one mutation (otherwise nothing has changed)
            while randf() < self.mutation_rate || num_mutations == 0 {
                // generate random mutation index
                let index = rng.gen_range(0..self.bits_per_chromosome);

                // mutate chromosome at index
                let _ = self.population[i].mutate(index);

                num_mutations += 1;

                if num_mutations == self.bits_per_chromosome {
                    break;
                }
            }
        }
    }

    // rank population by fitness (best first)
    fn rank(&mut self) {
        for i in 0..self.population_size {
            for j in (i + 1..self.population_size).rev() {
                if threshold_switch(self.utility[j], self.utility[j - 1], !self.minimize) {
                    // swap chromosomes
                    self.population.swap(j, j - 1);

                    // swap utility values
                    self.utility.swap(j, j - 1);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::cell::Cell;
    use std::f32::consts::FRAC_1_SQRT_2;
    use test_macro::autotest_annotate;

    // peak callback function; value nearest {p, p, p, ...} where p = 1/sqrt(2)
    fn gasearch_autotest_peak_callback(c: &Chromosome) -> f32 {
        let n = c.num_traits();
        let mut u = 1.0f32;
        let sig = 0.2f32;
        let p = FRAC_1_SQRT_2;
        for i in 0..n {
            // extract chromosome values
            let v = c.fraction(i).expect("index is below chromosome trait count");
            let e = v - p;
            u *= (-e * e / (2.0 * sig * sig)).exp();
        }
        u
    }

    // find values which maximize function
    #[test]
    #[autotest_annotate(autotest_gasearch_peak)]
    fn gasearch_peak() {
        let num_parameters = 16; // dimensionality of search (minimum 1)
        let bits_per_parameter = 6; // parameter resolution
        let num_iterations = 8000; // number of iterations to run
        let population_size = 16; // GA population size
        let mutation_rate = 0.2f32; // GA mutation rate
        let tol = 0.1f32; // error tolerance

        // create prototype chromosome
        let prototype = Chromosome::new_uniform(num_parameters, bits_per_parameter).unwrap();

        // create gasearch object
        let mut ga = GeneticSearch::new_with_settings(
            gasearch_autotest_peak_callback,
            &prototype,
            OptimDirection::Maximize,
            population_size,
            mutation_rate,
        )
        .unwrap();

        // execute search at once
        ga.run(num_iterations, 1e6);

        // get optimum utility
        let (opt_c, optimum_utility) = ga.best();
        let v_opt: Vec<f32> = (0..num_parameters)
            .map(|i| opt_c.fraction(i).expect("index is below chromosome trait count"))
            .collect();

        // test results, optimum at {p, p, p, ...} where p = 1/sqrt(2)
        for &vi in &v_opt {
            assert_abs_diff_eq!(vi, FRAC_1_SQRT_2, epsilon = tol);
        }

        // test value of utility (should be nearly 1)
        assert!(optimum_utility > 0.70);
    }

    // test configuration
    #[test]
    #[autotest_annotate(autotest_gasearch_config)]
    fn gasearch_config() {
        // create prototype chromosome
        let prototype = Chromosome::new_uniform(8, 12).unwrap();

        // check invalid function calls (a null utility is unrepresentable in Rust)
        // bad population size (too small)
        assert!(GeneticSearch::new_with_settings(
            gasearch_autotest_peak_callback,
            &prototype,
            OptimDirection::Maximize,
            0,
            0.1
        )
        .is_err());
        // bad mutation rate
        assert!(GeneticSearch::new_with_settings(
            gasearch_autotest_peak_callback,
            &prototype,
            OptimDirection::Maximize,
            16,
            -1.0
        )
        .is_err());

        // create proper object and test configurations
        let mut ga = GeneticSearch::new(gasearch_autotest_peak_callback, &prototype, OptimDirection::Maximize);
        assert_eq!(ga.num_traits(), prototype.num_traits());

        assert!(ga.configure_population(0, 8).is_err()); // population size too small
        assert!(ga.configure_population(usize::MAX, 8).is_err()); // population size too large
        assert!(ga.configure_population(24, 0).is_err()); // selection size too small
        assert!(ga.configure_population(24, 24).is_err()); // selection size too large
        assert!(ga.configure_population(24, 12).is_ok()); // ok
        assert!(ga.set_mutation_rate(-1.0).is_err()); // mutation rate out of range
        assert!(ga.set_mutation_rate(2.0).is_err()); // mutation rate out of range
        assert!(ga.set_mutation_rate(0.1).is_ok()); // ok
    }

    #[test]
    fn initial_best_chromosome_matches_its_utility() {
        let mut parent = Chromosome::new(&[32]).unwrap();
        let score = |chromosome: &Chromosome| chromosome.value(0).expect("chromosome has one trait") as f32;

        let maximize = GeneticSearch::new(score, &parent, OptimDirection::Maximize);
        let (best, utility) = maximize.best();
        assert_eq!(utility, score(best));
        assert_eq!(best.value(0), maximize.population[0].value(0));

        parent.set_values(&[u32::MAX as u64]).unwrap();
        let minimize = GeneticSearch::new(score, &parent, OptimDirection::Minimize);
        let (best, utility) = minimize.best();
        assert_eq!(utility, score(best));
        assert_eq!(best.value(0), minimize.population[0].value(0));
    }

    #[test]
    fn run_zero_iterations_preserves_population() {
        let calls = Cell::new(0);
        let chromosome = Chromosome::new(&[8]).unwrap();
        let mut search = GeneticSearch::new(
            |candidate| {
                calls.set(calls.get() + 1);
                candidate.value(0).unwrap() as f32
            },
            &chromosome,
            OptimDirection::Maximize,
        );

        let calls_before = calls.get();
        let (best, utility) = search.best();
        let best_value = best.value(0);
        let population_values: Vec<_> = search.population.iter().map(|candidate| candidate.value(0)).collect();

        assert_eq!(search.run(0, f32::INFINITY), utility);
        assert_eq!(search.best().0.value(0), best_value);
        assert_eq!(search.population.iter().map(|candidate| candidate.value(0)).collect::<Vec<_>>(), population_values);
        assert_eq!(calls.get(), calls_before);
    }
}
