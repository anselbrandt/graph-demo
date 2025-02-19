import json
from pathlib import Path
import typing
import warnings

import lancedb
import networkx as nx

from constants import (
    LANCEDB_URI,
)

from utils import TextChunk, gen_pyvis, construct_kg

warnings.simplefilter(action="ignore", category=FutureWarning)

if __name__ == "__main__":

    # define the global data structures
    files: typing.List[str] = [
        f"output/{file.name}" for file in Path("output").iterdir() if file.is_file()
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
            files,
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
            num_docs=len(files),
        )
    except Exception as ex:
        print(ex)
