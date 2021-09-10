import numpy as np
import pandas as pd

from pathlib import Path
from functools import reduce

def parallel_reduction(capacitance_arr: np.ndarray) -> float:
    return reduce(lambda acc, cap: acc + cap, capacitance_arr)

def series_reduction(capacitance_arr: np.ndarray) -> float:
    return 1/reduce(lambda acc, cap: 1/acc + 1/cap, capacitance_arr)

def standard_deviation(values: np.ndarray) -> float:
    assert len(values) > 0
    return sum(reduce(lambda p, n: p.append((p[-1]-n)**2), values, [values[-1]]))

class Capacitor:

    def __init__(self, serial:str, capacitance:float, desviation:float, resistance:float):
        self.serial = serial
        self.capacitance = capacitance
        self.desviation = desviation
        self.resistance = resistance

    def from_pandas_series(capacitor_data: pd.Series):
        required_fields = ['Capacitancia', 'Desviación', 'Serie', 'Resistencia']
        assert set(capacitor_data.columns).issubset(required_fields)
        capacitor = Capacitor(
            serial=capacitor_data.Serie,
            capacitance=capacitor_data.Capacitancia,
            desviation=capacitor_data.Desviación,
            resistance=capacitor_data.Resistencia
            )

        return capacitor


class Genoma:
    def __init__(self, len: int, options: np.ndarray, combination: np.ndarray = None):
        self.len = len
        self.options = options
        self.total_options = options
        if type(combination) == type(None):
            self.make_random_combination()
        else:
            assert self.len == combination.__len__()
            self.combination = combination
            self.update_options()

    def make_random_combination(self) -> np.ndarray:
        self.combination = np.random.choice(self.total_options, size=self.len, replace=False)
        self.update_options()

    def mutate_combination(self):
        options_mask = self.__create_random_mask(mask_length=len(self.options), mask_choices=np.random.choice(len(self.options)))
        mask_choices_number = len(self.options[options_mask])
        combination_mask = self.__create_random_mask(mask_length=len(self.combination), mask_choices=mask_choices_number)
        assert combination_mask.sum() == options_mask.sum()
        self.combination[combination_mask] = self.options[options_mask]
        self.update_options()

    def shuffle_combination(self):
        np.random.shuffle(self.combination)

    def __create_random_mask(self, mask_length: int, mask_choices: int):
        assert type(mask_length) == int and type(mask_choices) == int
        selected = np.zeros(mask_length, dtype=int)
        selected[:mask_choices] = 1
        np.random.shuffle(selected)
        mask = selected.astype(bool)
        return mask

    def crossover_combination(self, other):
        """returns two new BankGenomas in a tuple, with the combination done."""
        assert type(other) == Genoma
        # seleccionar una cantidad de valores a cambiar
        mask = self.__create_random_mask(len(self.combination), mask_choices=np.random.choice(self.len))
        comb_1, comb_2 = self.combination.copy(), other.combination.copy()
        comb_1[mask], comb_2[mask] = comb_2[mask], comb_1[mask]
        new_gens = Genoma(self.len, self.total_options, comb_1), Genoma(self.len, self.total_options, comb_2)
        return new_gens
    
    def update_options(self):
        self.options = np.array(list(filter(lambda c: c not in list(self.combination), self.total_options)))


class BankGenoma(Genoma):
    def __init__(self, len: int, options: np.ndarray, combination: np.ndarray):
        self.num_phases = 3
        self.num_groups_per_phase = 6
        self.num_capacitors_per_group = 9
        assert len == self.num_phases * self.num_groups_per_phase * self.num_capacitors_per_group
        super().__init__(len, options, combination=combination)

    @property
    def by_phases(self) -> np.ndarray:
        return np.split(self.map_capacitance, self.num_phases)
    
    @property
    def phase_reduction(self) -> np.ndarray:
        return np.array([series_reduction([parallel_reduction(group) for group in phase]) for phase in self.reshaped])
    
    @property
    def by_groups(self) -> np.ndarray:
        return np.split(self.map_capacitance, self.num_phases * self.num_groups_per_phase)
    
    @property
    def reshaped(self) -> np.ndarray:
        return self.map_capacitance.reshape((self.num_phases, self.num_groups_per_phase, self.num_capacitors_per_group))

    @property
    def group_accumulated_deviations(self) -> float:
        return sum([standard_deviation(group) for phase in self.reshaped for group in phase])
    
    @property
    def phase_accumulated_deviations(self) -> float:
        return standard_deviation(self.phase_reduction)
    
    @property
    def fitness(self):
        return self.group_accumulated_deviations + self.phase_accumulated_deviations

    @property
    def map_capacitance(self):
        return np.array([cap.capacitance for cap in self.combination])
    
    @property
    def map_serial(self):
        return np.array([cap.serial for cap in self.combination])
    

def main():
    excel_name = 'capacitor_list.xlsx'
    sheet_name = 'sheet 1'
    excel_file = Path(__file__).parent / excel_name
    # df = pd.read_excel(excel_file, sheet_name)
    df = pd.DataFrame()

    length = 3 * 6 * 9
    options = [Capacitor.from_pandas_series(cap) for _, cap in df.itertuples()]

    assert length < len(options)

    gen1 = Genoma(length, options)
    gen2 = Genoma(length, options)

    print('Original')
    print("gen1:", gen1.combination, "\ngen2:", gen2.combination, end="\n\n")

    gen1.make_random_combination()
    gen2.make_random_combination()

    print('random')
    print("gen1:", gen1.combination, "\ngen2:", gen2.combination, end="\n\n")

    gen1.mutate_combination()
    gen2.mutate_combination()

    print('mutation')
    print("gen1:", gen1.combination, "\ngen2:", gen2.combination, end="\n\n")

    gen1.shuffle_combination()
    gen2.shuffle_combination()

    print('shuffle')
    print("gen1:", gen1.combination, "\ngen2:", gen2.combination, end="\n\n")

    gen1, gen2 = gen1.crossover_combination(gen2)

    print('crossover')
    print("gen1:", gen1.combination, "\ngen2:", gen2.combination, end="\n\n")


if __name__ == "__main__":
    main()

