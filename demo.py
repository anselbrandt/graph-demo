import json
from pathlib import Path
import traceback
import tracemalloc
import typing
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from icecream import ic
from pyinstrument import Profiler
import lancedb
import networkx as nx

from constants import (
    LANCEDB_URI,
)

from utils import TextChunk, gen_pyvis, construct_kg

if __name__ == "__main__":
    # start the stochastic call trace profiler and memory profiler
    profiler: Profiler = Profiler()
    profiler.start()
    tracemalloc.start()

    # define the global data structures
    url_list: typing.List[str] = [
        "https://aaic.alz.org/releases-2024/processed-red-meat-raises-risk-of-dementia.asp",
        "https://www.theguardian.com/society/article/2024/jul/31/eating-processed-red-meat-could-increase-risk-of-dementia-study-finds",
    ]

    vect_db: lancedb.db.LanceDBConnection = lancedb.connect(LANCEDB_URI)

    chunk_table: lancedb.table.LanceTable = vect_db.create_table(
        "chunk",
        schema=TextChunk,
        mode="overwrite",
    )

    sem_overlay: nx.Graph = nx.Graph()

    try:
        construct_kg(
            url_list,
            chunk_table,
            sem_overlay,
            Path("data/entity.w2v"),
            debug=False,  # True
        )

        # serialize the resulting KG
        with Path("data/kg.json").open("w", encoding="utf-8") as fp:
            fp.write(
                json.dumps(
                    nx.node_link_data(sem_overlay),
                    indent=2,
                    sort_keys=True,
                )
            )

        # generate HTML for an interactive visualization of a graph
        gen_pyvis(
            sem_overlay,
            "kg.html",
            num_docs=len(url_list),
        )
    except Exception as ex:
        ic(ex)
        traceback.print_exc()

    # stop the profiler and report performance statistics
    profiler.stop()
    profiler.print()

    # report the memory usage
    report: tuple = tracemalloc.get_traced_memory()
    peak: float = round(report[1] / 1024.0 / 1024.0, 2)
    print(f"peak memory usage: {peak} MB")
