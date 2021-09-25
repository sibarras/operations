import numpy as np
import pandas as pd

from pathlib import Path
from functools import reduce

def parallel_reduction(capacitance_arr: np.ndarray) -> float:
    return np.sum(capacitance_arr)

def series_reduction(capacitance_arr: np.ndarray) -> float:
    return 1/np.sum(1/capacitance_arr)

def standard_deviation(values: np.ndarray) -> float:
    return np.std(values)


class Capacitor:

    def __init__(self, capacitance:float, desviation:float, serial:str):
        self.serial = serial
        self.capacitance = capacitance
        self.desviation = desviation

    def from_pandas_series(capacitor_data: pd.Series):
        required_fields = ['N°', 'N° DE SERIE', 'CAPACITANCIA ', 'CAP. DESVIACION ']
        try:
            capacitor = Capacitor(
                serial=capacitor_data[required_fields[1]],
                capacitance=capacitor_data[required_fields[2]],
                desviation=capacitor_data[required_fields[3]],
                # resistance=capacitor_data.Resistencia
                )
        except Exception as e:
            print(f"[ERROR]: Dataframe is not valid. {e}")
            exit()
        return capacitor
    
    def __repr__(self) -> str:
        return f"(serial: {self.serial}, capacitance: {self.capacitance}, desviation: {self.desviation})"


class Genoma:
    def __init__(self, len: int, options: np.ndarray, combination: np.ndarray = None):
        self.len = len
        self.options = options
        self.total_options = options
        if type(combination) == type(None):
            self.combination = self.make_random_combination(create=True)
            self.update_options()
        else:
            assert self.len == combination.__len__()
            self.combination = combination
            self.update_options()

    def get_copy(self):
        return type(self)(len=self.len, options=self.total_options, combination=self.combination)

    def make_random_combination(self, create=False) -> np.ndarray:
        if create: return np.random.choice(self.total_options, size=self.len, replace=False)

        gen = type(self)(len=self.len, options=self.total_options, combination=self.combination.copy())
        gen.combination = np.random.choice(gen.total_options, size=gen.len, replace=False)
        gen.update_options()
        return gen

    def mutate_combination(self):
        gen = type(self)(len=self.len, options=self.total_options, combination=self.combination.copy())
        # options_mask = gen.__create_random_mask(mask_length=len(gen.options), mask_choices=np.random.choice(len(gen.combination)))
        options_mask = gen.__create_random_mask(mask_length=len(gen.options), mask_choices=1)
        # mask_choices_number = len(gen.options[options_mask])
        # combination_mask = gen.__create_random_mask(mask_length=len(gen.combination), mask_choices=mask_choices_number)
        combination_mask = gen.__create_random_mask(mask_length=len(gen.combination), mask_choices=1)

        gen.combination[combination_mask] = gen.options[options_mask]
        gen.update_options()
        return gen

    def shuffle_combination(self):
        gen = type(self)(len=self.len, options=self.total_options, combination=self.combination.copy())
        np.random.shuffle(gen.combination)
        return gen

    def __create_random_mask(self, mask_length: int, mask_choices: int):
        assert type(mask_length) == int and type(mask_choices) == int and mask_length > mask_choices
        selected = np.zeros(mask_length, dtype=int)
        selected[:mask_choices] = 1
        np.random.shuffle(selected)
        mask = selected.astype(bool)
        return mask

    def crossover_combination(self, other):
        """returns two new BankGenomas in a tuple, with the combination done."""
        assert issubclass(type(other), Genoma)
        gen1, gen2 = type(self)(self.len, self.total_options, self.combination.copy()), type(self)(other.len, other.total_options, other.combination.copy())
        mask = gen1.__create_random_mask(len(gen1.combination), mask_choices=np.random.choice(gen1.len))
        # comb_1, comb_2 = gen.combination.copy(), other.combination.copy()
        # comb_1[mask], comb_2[mask] = comb_2[mask], comb_1[mask]
        # new_gens = type(self)(self.len, self.total_options, comb_1), type(self)(self.len, self.total_options, comb_2)
        gen1.combination[mask], gen2.combination[mask] = gen2.combination[mask].copy(), gen1.combination[mask].copy()
        return gen1, gen2
    
    def update_options(self):
        self.options = np.array(list(filter(lambda c: c not in list(self.combination), self.total_options)))


class BankGenoma(Genoma):
    def __init__(self, len: int, options: np.ndarray, combination: np.ndarray = None):
        self.num_phases = 3
        self.num_groups_per_phase = 6
        self.num_capacitors_per_group = 9
        assert len == self.num_phases * self.num_groups_per_phase * self.num_capacitors_per_group
        super().__init__(len, options, combination=combination)

    def __repr__(self) -> str:
        return f"{self.map_capacitance}"

    @property
    def by_phases(self) -> np.ndarray:
        return self.map_capacitance.reshape((3, -1))
    
    @property
    def phase_reduction(self) -> np.ndarray:
        return np.array([series_reduction(np.array([parallel_reduction(group) for group in phase])) for phase in self.reshaped])
    
    @property
    def by_groups(self) -> np.ndarray:
        return np.array(self.map_capacitance).reshape((3*6, -1))
    
    @property
    def reshaped(self) -> np.ndarray:
        return np.array(self.map_capacitance).reshape((self.num_phases, self.num_groups_per_phase, self.num_capacitors_per_group))

    @property
    def sum_group_accumulated_deviations(self) -> float:
        return sum([standard_deviation(group) for phase in self.reshaped for group in phase])
    
    @property
    def group_accumulated_deviations(self) -> float:
        return standard_deviation([standard_deviation(group) for phase in self.reshaped for group in phase])
    
    @property
    def phase_accumulated_deviations(self) -> float:
        return standard_deviation(self.phase_reduction)
    
    @property
    def fitness(self):
        return self.phase_accumulated_deviations * 18 + self.group_accumulated_deviations + self.sum_group_accumulated_deviations

    @property
    def map_capacitance(self):
        return np.array([cap.capacitance for cap in self.combination])


if __name__ == "__main__":
    excel_name = 'capacitor_list.xlsx'
    sheet_name = 'sheet 1'
    excel_file = Path(__file__).parent / excel_name
    # df = pd.read_excel(excel_file, sheet_name)

    length = 3 * 6 * 9
    # options = [Capacitor.from_pandas_series(cap) for _, cap in df.itertuples()]
    nominal_value = 0.986
    max_deviation = nominal_value * 0.05
    options = nominal_value + np.random.random(3 * (3 * 6 * 9)) * max_deviation
    capacitor_options = list(map(lambda opt: Capacitor(opt[0], opt[1], (opt[1]-nominal_value)/nominal_value), enumerate(options)))

    genoma = BankGenoma(3*6*9, capacitor_options)
    print(genoma.fitness)


