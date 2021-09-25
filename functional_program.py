import asyncio
import numpy as np
import pandas as pd
import random
from sqlite3 import connect
from pathlib import Path


class Capacitor:
    def __init__(self, id: int, capacitance: float) -> None:
        self.id = id
        self.capacitance = capacitance
    
    def __repr__(self) -> str:
        return str(self.capacitance)


class Genoma:
    def __init__(self, combination: list[Capacitor]) -> None:
        self.combination = combination
    
    def __repr__(self) -> str:
        return str(self.combination)


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
    parallel = [await parallel_reduction(caps) async for caps in aiter(np.array(tot_caps), slice=9)]
    series = [await series_reduction(groups) async for groups in aiter(np.array(parallel), slice=6)]
    return await standard_deviation(series)

async def df_to_cap_list(df: pd.DataFrame) -> list[Capacitor]:
    assert set(df.columns).issuperset(["CAPACITANCIA ", "N° DE SERIE"])
    valid = df.loc[df['CAPACITANCIA '].notna()]
    return valid.apply(lambda s: Capacitor(s["N° DE SERIE"], s["CAPACITANCIA "]), axis=1).to_numpy().tolist()

async def random_genoma(capacitor_options: list[Capacitor], genoma_length: int) -> Genoma:
    return Genoma(random.sample(capacitor_options, k=genoma_length))

async def random_generation(capacitor_options: list[Capacitor], generation_length: int, genoma_lenth: int) -> list[Genoma]:
    return [await random_genoma(capacitor_options, genoma_lenth) async for _ in aiter(range(generation_length))]

async def crossover_genoma(gen1: Genoma, gen2: Genoma) -> Genoma:
    # verificar si el orden importa. Si async for cambia el orden o no...
    return Genoma([g1 if random.getrandbits(1) == 1 else g2 async for g1, g2 in aiter(zip(gen1.combination, gen2.combination))])

async def mutation_genoma(gen: Genoma, capacitor_options: list[Capacitor]) -> Genoma:
    index = random.randint(0, len(gen.combination) - 1)
    return Genoma([prev if n == index else random.choice(capacitor_options) async for n, prev in aiter(enumerate(gen.combination))])

async def shuffle_genoma(gen: Genoma) -> Genoma:
    comb = gen.combination.copy()
    random.shuffle(comb)
    return Genoma(comb)

async def update_generation(past_generation: list[Genoma], capacitor_options: list[Capacitor], gen_len: int, top_len: int, rand_len: int, cross_len: int, mut_len: int, shuff_len: int) -> list[Genoma]:
    top_genomas = sorted(past_generation, key = lambda g: await fitness(g))[:top_len] # problema con el async aqui
    mutated_genomas = [await mutation_genoma(gen, capacitor_options) async for gen in aiter(random.choices(top_genomas, k=mut_len))]
    shuffled_genomas = [await shuffle_genoma(gen) async for gen in aiter(random.choices(top_genomas, k=shuff_len))]
    random_genomas = [await random_genoma(capacitor_options, gen_len) async for _ in aiter(range(rand_len))]
    crossovered_genomas = [await crossover_genoma(g1, g2) async for g1, g2 in aiter(zip(random.choices(top_genomas, k=cross_len), random.choices(top_genomas, k=cross_len)))]
    return top_genomas + mutated_genomas + shuffled_genomas + random_genomas + crossovered_genomas

async def test():
    MAIN_FOLDER = Path(__file__).parent
    DB_FILE = MAIN_FOLDER / 'capacitors.db'
    EXCEL_FILE = MAIN_FOLDER / 'CAPACITORES.xlsx'
    RESULT_FILE = MAIN_FOLDER / 'resultados.xlsx'

    GENOMA_LENGTH = 3 * 6 * 9
    OPTIONS_LENGTH = 3 * (3 * 6 * 9)
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
    generation = await random_generation(capacitor_options, GENERATION_LENGTH, GENOMA_LENGTH)
    best_fitness: list[Genoma] = []

    []

    cap1 = Capacitor(1, 50)
    cap2 = Capacitor(2, 40)
    cap3 = Capacitor(3, 60)
    cap4 = Capacitor(4, 20)
    cap5 = Capacitor(5, 70)

    genoma1 = Genoma([cap1 async for _ in aiter(range(3*6*9))])
    genoma2 = Genoma([cap3 async for _ in aiter(range(3*6*9))])

    print("calling mapping functions")
    cap1 = await map_capacitance(genoma1)
    cap2 = await map_capacitance(genoma2)
    print("mapping finished!")

    print("calling mapping id")
    id1 = await map_id(genoma1)
    id2 = await map_id(genoma2)
    print("mapping finished!")

    print("calling fitness")
    fit1 = await fitness(genoma1)
    fit2 = await fitness(genoma2)
    print("fitness finished!")

    print("calling random genoma")
    gen1 = await random_genoma(capacitor_options, GENOMA_LENGTH)
    gen2 = await random_genoma(capacitor_options, GENOMA_LENGTH)
    print("random finished!")

    print("calling mutated genoma")
    gen1 = await mutation_genoma(gen1, capacitor_options)
    print("mutation finished!")

    print("calling shuffle genoma")
    gen1 = await shuffle_genoma(genoma1)
    print("shuffle finished!")

    print("calling crossover genoma")
    gen1 = await crossover_genoma(gen1, gen2)
    print("crossover finished!")

    print("Calling an generation iteration")
    generation = await update_generation(
        generation,
        capacitor_options,
        GENERATION_LENGTH,
        BEST_GENOMAS_LENGTH,
        RANDOM_LENGTH,
        CROSSOVER_LENGTH,
        MUTATION_LENGTH,
        SHUFFLE_LENGTH
    )
    print("new generation finished")




if __name__ == "__main__":
    asyncio.run(test())
