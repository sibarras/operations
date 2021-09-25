from balance import BankGenoma, Capacitor, pd, Path
from sqlite3 import connect
import random
# crossover
# random
# mutation

random_generation = lambda generation_len, genoma_len, cap_options: [BankGenoma(genoma_len, cap_options) for _ in range(int(generation_len))]

def get_copies(top_genomas: list[BankGenoma]) -> list[BankGenoma]:
    print("copy")
    return [gen.get_copy() for gen in top_genomas]

def get_mutations(top_genomas: list[BankGenoma], mut_len: int) -> list[BankGenoma]:
    print("mutation")
    return [gen.mutate_combination() for gen in random.choices(top_genomas, k=mut_len)]

def get_shuffle(top_genomas: list[BankGenoma], shuffle_len: int) -> list[BankGenoma]:
    print("shuffle")
    return [gen.shuffle_combination() for gen in random.choices(top_genomas, k=shuffle_len)]

def get_random(generation: list[BankGenoma], random_len: int) -> list[BankGenoma]:
    print("random")
    return [gen.make_random_combination() for gen in generation[:random_len]]

def get_crossover(top_genomas: list[BankGenoma], cap_options: list[Capacitor], crossover_len: int) -> list[BankGenoma]:
    print("crossover")
    return [g for gen, other in zip(random.sample(top_genomas, k=int(crossover_len/2)), random_generation(crossover_len/2, top_genomas[0].len, cap_options)) for g in gen.crossover_combination(other)]


def get_generation(index: int, last_generation: list[BankGenoma], capacitor_options: list[Capacitor], lengths: tuple[int]) -> list[BankGenoma]:
    BEST_GENOMAS_LENGTH, MUTATION_LENGTH, SHUFFLE_LENGTH, RANDOM_LENGTH, GENOMA_LENGTH, CROSSOVER_LENGTH = lengths
    top_genomas = sorted(last_generation, key=lambda gen: gen.fitness)[:BEST_GENOMAS_LENGTH]
    print(f'generacion {index} -> fitness:', top_genomas[0].fitness)

    tasks = [
        get_copies(top_genomas),
        get_mutations(top_genomas, MUTATION_LENGTH),
        get_shuffle(top_genomas, SHUFFLE_LENGTH),
        get_random(last_generation, RANDOM_LENGTH),
        get_crossover(last_generation, capacitor_options, CROSSOVER_LENGTH),
    ]
    return [gen for gen_list in tasks for gen in gen_list]

def main():
    MAIN_FOLDER = Path(__file__).parent
    DB_FILE = MAIN_FOLDER / 'capacitors.db'
    EXCEL_FILE = MAIN_FOLDER / 'CAPACITORES.xlsx'
    RESULT_FILE = MAIN_FOLDER / 'resultados.xlsx'

    GENOMA_LENGTH = 3 * 6 * 9
    OPTIONS_LENGTH = 3 * (3 * 6 * 9)
    GENERATION_LENGTH = 1000
    BEST_GENOMAS_LENGTH = int(GENERATION_LENGTH * 10 / 100)
    MUTATION_LENGTH = int(GENERATION_LENGTH * 20 / 100)
    SHUFFLE_LENGTH = int(GENERATION_LENGTH * 10 / 100)
    CROSSOVER_LENGTH = int(GENERATION_LENGTH * 20 / 100)
    RANDOM_LENGTH = int(GENERATION_LENGTH * 40 / 100)

    assert GENERATION_LENGTH == BEST_GENOMAS_LENGTH + MUTATION_LENGTH + SHUFFLE_LENGTH + CROSSOVER_LENGTH + RANDOM_LENGTH

    ITERATIONS = 1000

    nominal_value = 0.986
    final_array_ideal_value = nominal_value * 9 / 6

    if not DB_FILE.exists():
        capacitors: pd.DataFrame = pd.read_excel(EXCEL_FILE, sheet_name="DATOS DE CAPACITORES ", header=4, index_col=0, usecols='A:D')
        with connect(DB_FILE) as conn:
            capacitors.to_sql('capacitors', conn)
    else:
        with connect(DB_FILE) as conn:
            capacitors = pd.read_sql_query("SELECT * FROM capacitors", conn, index_col='N°')
    

    capacitors_valid = capacitors.loc[capacitors['CAPACITANCIA '].notna()]
    capacitor_options = capacitors_valid.apply(Capacitor.from_pandas_series, axis=1).to_numpy().tolist()

    generation = random_generation(GENERATION_LENGTH, GENOMA_LENGTH, capacitor_options)
    best_fitness: list[BankGenoma] = []

    for i in range(ITERATIONS):
        generation = get_generation(i, generation, capacitor_options, lengths=(BEST_GENOMAS_LENGTH, MUTATION_LENGTH, SHUFFLE_LENGTH, RANDOM_LENGTH, GENOMA_LENGTH, CROSSOVER_LENGTH))
        
    else:
        top_genomas = sorted(generation, key=lambda genoma: genoma.fitness)
        optimum_gen = top_genomas[0]

    print('optimum gen shaped:\n', optimum_gen.reshaped)
    print('optimum gen fitness:\n', optimum_gen.fitness)
    print('optimum gen by phases:\n', optimum_gen.by_phases)
    print('optimum gen by groups:\n', optimum_gen.by_groups)
    print('optimum gen group deviations:\n', optimum_gen.group_accumulated_deviations)
    print('optimum gen sum group deviations:\n', optimum_gen.sum_group_accumulated_deviations)
    print('optimum gen phase deviations:\n', optimum_gen.phase_accumulated_deviations)
    print('optimum gen phase reduction:\n', optimum_gen.phase_reduction)
    print("Difference with ideal {}: A: {}, B: {}, C: {}\n\n".format(final_array_ideal_value, *optimum_gen.by_phases))

    cols = list(optimum_gen.combination[0].__dict__.keys())
    data = [tuple(c.__dict__.values()) for c in optimum_gen.combination]

    print("cols:", cols)
    print("data:", data)
    result = pd.DataFrame(data, columns=cols)
    result.to_excel(RESULT_FILE, "results")
    print(result)

if __name__ == "__main__":
    main()