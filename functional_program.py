import asyncio
import numpy as np
import pandas as pd
import random
from sqlite3 import connect
from pathlib import Path
from matplotlib import pyplot as plt


class Capacitor:
    def __init__(self, id: int, capacitance: float) -> None:
        self.id = id
        self.capacitance = capacitance
    
    def __repr__(self) -> str:
        return str(self.capacitance)


class Genoma:
    def __init__(self, combination: list[Capacitor]) -> None:
        self.combination = combination


async def aiter(iterable: list, slice: int = None) -> list:
    """aiter creater an asyncronous generator to use in async for and async comprehensions

    Args:
        iterable (list): list to transform into async generator
        split (int, optional): number of items yo hold in a single iteration of the iterable.

    Returns:
        list: async generator

    Yields:
        Iterator[list]: values inside the iterable input or parts that the input is splitted
    """
    assert hasattr(iterable, "__iter__")
    if slice != None:
        assert slice >= 1
        assert len(iterable) % slice == 0
        assert slice <= len(iterable)
        parts = int(len(iterable) / slice)
        for i in range(parts):
            yield iterable[slice * i: slice * (i + 1)]

    else:
        for val in iterable:
            yield val

async def parallel_reduction(capacitance_arr: np.ndarray) -> float:
    return np.sum(capacitance_arr)

async def series_reduction(capacitance_arr: np.ndarray) -> float:
    return 1/np.sum(1/capacitance_arr)

async def standard_deviation(values: np.ndarray) -> float:
    return np.std(values)

async def map_capacitance(gen: Genoma) -> list[float]:
    return [cap.capacitance async for cap in aiter(gen.combination)]

async def map_id(gen: Genoma) -> list[str]:
    return [cap.id async for cap in aiter(gen.combination)]

async def fitness(gen: Genoma) -> float:
    tot_caps = await map_capacitance(gen)
    groups_dev = sum([await standard_deviation(group) async for group in aiter(np.array(tot_caps), slice=9)])
    parallel = [await parallel_reduction(caps) async for caps in aiter(np.array(tot_caps), slice=9)]
    series = [await series_reduction(groups) async for groups in aiter(np.array(parallel), slice=6)]
    return await standard_deviation(series) + await standard_deviation(np.array(parallel)) + groups_dev

async def df_to_cap_list(df: pd.DataFrame) -> list[Capacitor]:
    assert set(df.columns).issuperset(["CAPACITANCIA ", "N° DE SERIE"])
    valid = df.loc[df['CAPACITANCIA '].notna()]
    return valid.apply(lambda s: Capacitor(s["N° DE SERIE"], s["CAPACITANCIA "]), axis=1).to_numpy().tolist()

async def top_genomas(past_generation: list[Genoma], top_len: int) -> list[Genoma]:
    fitness_tuple_calculations = [(await fitness(gen), gen) async for gen in aiter(past_generation)]
    genomas = [gen for _, gen in sorted(fitness_tuple_calculations, key=lambda v:v[0])]
    return genomas[:top_len]

async def random_genoma(capacitor_options: list[Capacitor], genoma_length: int) -> Genoma:
    return Genoma(random.sample(capacitor_options, k=genoma_length))

async def random_genomas(capacitor_options: list[Capacitor], rand_len: int, genoma_len: int) -> list[Genoma]:
    return [await random_genoma(capacitor_options, genoma_len) async for _ in aiter(range(rand_len))]

async def crossover_genoma(gen1: Genoma, gen2: Genoma) -> Genoma:
    return Genoma([g1 if random.getrandbits(1) == 1 else g2 async for g1, g2 in aiter(zip(gen1.combination, gen2.combination))])

async def crossover_genomas(top_genomas: list[Genoma], cross_len: int) -> list[Genoma]:
    return [await crossover_genoma(g1, g2) async for g1, g2 in aiter(zip(random.choices(top_genomas, k=cross_len), random.choices(top_genomas, k=cross_len)))]

async def mutation_genoma(gen: Genoma, capacitor_options: list[Capacitor]) -> Genoma:
    index = random.randint(0, len(gen.combination) - 1)
    return Genoma([prev if n == index else random.choice(capacitor_options) async for n, prev in aiter(enumerate(gen.combination))])

async def mutation_genomas(top_genomas: list[Genoma], capacitor_options: list[Genoma], mut_len: int) -> list[Genoma]:
    return [await mutation_genoma(gen, capacitor_options) async for gen in aiter(random.choices(top_genomas, k=mut_len))]

async def shuffle_genoma(gen: Genoma) -> Genoma:
    comb = gen.combination.copy()
    random.shuffle(comb)
    return Genoma(comb)

async def shuffle_genomas(top_genomas: list[Genoma], shuff_len: int) -> list[Genoma]:
    return [await shuffle_genoma(gen) async for gen in aiter(random.choices(top_genomas, k=shuff_len))]

async def update_generation(top_genomas: list[Genoma], capacitor_options: list[Capacitor], genoma_len: int, rand_len: int, cross_len: int, mut_len: int, shuff_len: int) -> list[Genoma]:
    
    generations = await asyncio.gather(
        mutation_genomas(top_genomas, capacitor_options, mut_len),
        shuffle_genomas(top_genomas, shuff_len),
        crossover_genomas(top_genomas, cross_len),
        random_genomas(capacitor_options, rand_len, genoma_len)
    )
    return [gen async for generation in aiter(generations) async for gen in aiter(generation)] + top_genomas


async def main():
    MAIN_FOLDER = Path(__file__).parent
    DB_FILE = MAIN_FOLDER / 'capacitors.db'
    EXCEL_FILE = MAIN_FOLDER / 'CAPACITORES.xlsx'
    RESULT_FILE = MAIN_FOLDER / 'resultados.xlsx'

    GENOMA_LENGTH = 3 * 6 * 9
    GENERATION_LENGTH = 1000
    ITERATIONS = 1000

    BEST_GENOMAS_LENGTH = int(GENERATION_LENGTH * 10 / 100)
    MUTATION_LENGTH = int(GENERATION_LENGTH * 20 / 100)
    SHUFFLE_LENGTH = int(GENERATION_LENGTH * 10 / 100)
    CROSSOVER_LENGTH = int(GENERATION_LENGTH * 20 / 100)
    RANDOM_LENGTH = int(GENERATION_LENGTH * 40 / 100)

    assert GENERATION_LENGTH == BEST_GENOMAS_LENGTH + MUTATION_LENGTH + SHUFFLE_LENGTH + CROSSOVER_LENGTH + RANDOM_LENGTH

    nominal_value = 0.986
    final_array_ideal_value = nominal_value * 9 / 6

    if not DB_FILE.exists():
        df: pd.DataFrame = pd.read_excel(EXCEL_FILE, sheet_name="DATOS DE CAPACITORES ", header=4, index_col=0, usecols='A:D')
        with connect(DB_FILE) as conn:
            df.to_sql('capacitors', conn)
    else:
        with connect(DB_FILE) as conn:
            df = pd.read_sql_query("SELECT * FROM capacitors", conn, index_col='N°')
    
    capacitor_options = await df_to_cap_list(df)
    generation = await random_genomas(capacitor_options, GENERATION_LENGTH, GENOMA_LENGTH)
    best_fitness: list[Genoma] = []
    

    count = 0
    while count < 100:
        # print("complete gens:", [await fitness(gen) for gen in generation])
        top_results = await top_genomas(generation, BEST_GENOMAS_LENGTH)
        # print("top gens:", [await fitness(gen) for gen in top_results])
        best_fitness.append(await fitness(top_results[0]))
        if len(best_fitness)>2 and best_fitness[-2] == best_fitness[-1]:
            count += 1
        else: count = 0
        print("geneation", len(best_fitness), "->", best_fitness[-1])
        generation = await update_generation(top_results, capacitor_options, GENOMA_LENGTH, RANDOM_LENGTH, CROSSOVER_LENGTH, MUTATION_LENGTH, SHUFFLE_LENGTH)

    best_combination = generation[0].combination
    cols = list(best_combination[0].__dict__.keys())
    data = [tuple(c.__dict__.values()) for c in best_combination]
    result = pd.DataFrame(data, columns=cols)
    result.to_excel(RESULT_FILE, "results")
    print(result)
    
    plt.plot(best_fitness)
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())
