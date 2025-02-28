from collections import defaultdict
from dataclasses import dataclass
import itertools
import math
from typing import Tuple, Optional, Dict, List, Set
import warnings

from lancedb.embeddings import get_registry
from lancedb.embeddings.transformers import TransformersEmbeddingFunction
from lancedb.pydantic import LanceModel, Vector
from spacy.language import Language
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
import glirel
from networkx import Graph, pagerank
from numpy import ndarray, linspace, digitize
from pandas import DataFrame, Series
import pyvis
import spacy

from constants import (
    CHUNK_SIZE,
    GLINER_MODEL,
    NER_LABELS,
    RE_LABELS,
    SPACY_MODEL,
    STOP_WORDS,
    TR_ALPHA,
    TR_LOOKBACK,
)

__all__ = ["glirel"]

EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"

EMBED_FCN: TransformersEmbeddingFunction = (
    get_registry().get("huggingface").create(name=EMBED_MODEL)
)


class TextChunk(LanceModel):
    uid: int
    file: str
    text: str = EMBED_FCN.SourceField()
    vector: Vector(EMBED_FCN.ndims()) = EMBED_FCN.VectorField(default=None)


@dataclass(order=False, frozen=False)
class Entity:
    loc: tuple[int, int]
    key: str
    text: str
    label: str
    chunk_id: int
    sent_id: int
    span: Span
    node: Optional[int] = None


def make_chunk(
    doc: Doc,
    file: str,
    chunk_list: List[TextChunk],
    chunk_id: int,
) -> int:
    """
    Split the given document into text chunks, returning the last index.
    BTW, for ideal text chunk size see
    <https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5>
    """
    chunks: List[str] = []
    chunk_total: int = 0
    prev_line: str = ""

    for sent_id, sent in enumerate(doc.sents):
        line: str = sent.text
        line_len: int = len(line)

        if (chunk_total + line_len) > CHUNK_SIZE:
            # emit the current chunk
            chunk_list.append(
                TextChunk(
                    uid=chunk_id,
                    file=file,
                    text="\n".join(chunks),
                )
            )

            # start a new chunk
            chunks = [prev_line, line]
            chunk_total = len(prev_line) + line_len
            chunk_id += 1
        else:
            # append line to the current chunk
            chunks.append(line)
            chunk_total += line_len

        prev_line = line

    # emit the trailing chunk
    chunk_list.append(
        TextChunk(
            uid=chunk_id,
            file=file,
            text="\n".join(chunks),
        )
    )

    return chunk_id + 1


def read_file(
    scrape_nlp: Language,
    file: str,
    chunk_list: List[TextChunk],
    chunk_id: int,
) -> int:

    with open(file, "r", encoding="utf-8") as f:
        parsed_text = f.read()

    scrape_doc: Doc = scrape_nlp(parsed_text)

    chunk_id = make_chunk(
        scrape_doc,
        file,
        chunk_list,
        chunk_id,
    )

    return chunk_id


def init_nlp() -> Language:

    # load models for `spaCy`, `GLiNER`, `GLiREL`
    # this may take several minutes when run the first time
    nlp: Language = spacy.load(SPACY_MODEL)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        nlp.add_pipe(
            "gliner_spacy",
            config={
                "gliner_model": GLINER_MODEL,
                "labels": NER_LABELS,
                "chunk_size": CHUNK_SIZE,
                "style": "ent",
            },
        )

        nlp.add_pipe(
            "glirel",
            after="ner",
        )

    return nlp


def parse_text(
    nlp: Language,
    known_lemma: List[str],
    lex_graph: Graph,
    chunk: TextChunk,
    *,
    debug: bool = False,
) -> Doc:
    """
    Parse an input text chunk, returning a `spaCy` document.
    """
    doc: Doc = list(
        nlp.pipe(
            [(chunk.text, RE_LABELS)],
            as_tuples=True,
        )
    )[
        0
    ][0]

    # scan the document tokens to add lemmas to _lexical graph_ using
    # a _textgraph_ approach called the _textrank_ algorithm
    for sent in doc.sents:
        node_seq: List[int] = []

        if debug:
            print(sent)

        for tok in sent:
            text: str = tok.text.strip()

            if tok.pos_ in ["NOUN", "PROPN"]:
                key: str = tok.pos_ + "." + tok.lemma_.strip().lower()
                prev_known: bool = False

                if key not in known_lemma:
                    # create a new node
                    known_lemma.append(key)
                else:
                    # link to an existing node, adding weight
                    prev_known = True

                node_id: int = known_lemma.index(key)
                node_seq.append(node_id)

                if not lex_graph.has_node(node_id):
                    lex_graph.add_node(
                        node_id,
                        key=key,
                        kind="Lemma",
                        pos=tok.pos_,
                        text=text,
                        chunk=chunk,
                        count=1,
                    )

                elif prev_known:
                    node: dict = lex_graph.nodes[node_id]
                    node["count"] += 1

        # create the _textrank_ edges for the lexical graph,
        # which will get used for ranking, but discarded later
        if debug:
            print(node_seq)

        for hop in range(TR_LOOKBACK):
            for node_id, node in enumerate(node_seq[: -1 - hop]):
                neighbor: int = node_seq[hop + node_id + 1]

                if not lex_graph.has_edge(node, neighbor):
                    lex_graph.add_edge(
                        node,
                        neighbor,
                        rel="FOLLOWS_LEXICALLY",
                    )

    return doc


def make_entity(
    span_decoder: Dict[tuple, Entity],
    sent_map: Dict[Span, int],
    span: Span,
    chunk: TextChunk,
    *,
    debug: bool = False,
) -> Entity:
    """
    Instantiate one `Entity` dataclass object, adding to our working "vocabulary".
    """
    key: str = " ".join([tok.pos_ + "." + tok.lemma_.strip().lower() for tok in span])

    ent: Entity = Entity(
        (
            span.start,
            span.end,
        ),
        key,
        span.text,
        span.label_,
        chunk.uid,
        sent_map[span.sent],
        span,
    )

    if ent.loc not in span_decoder:
        span_decoder[ent.loc] = ent

        if debug:
            print(ent)

    return ent


def extract_entity(
    known_lemma: List[str],
    lex_graph: Graph,
    ent: Entity,
    *,
    debug: bool = False,
) -> None:
    """
    Link one `Entity` into this doc's lexical graph.
    """
    prev_known: bool = False

    if ent.key not in known_lemma:
        # add a new Entity node to the graph and link to its component Lemma nodes
        known_lemma.append(ent.key)
    else:
        # phrase for this entity has been previously seen in other documents
        prev_known = True

    node_id: int = known_lemma.index(ent.key)
    ent.node = node_id

    # hydrate a compound phrase in this doc's lexical graph
    if not lex_graph.has_node(node_id):
        lex_graph.add_node(
            node_id,
            key=ent.key,
            kind="Entity",
            label=ent.label,
            pos="NP",
            text=ent.text,
            chunk=ent.chunk_id,
            count=1,
        )

        for tok in ent.span:
            tok_key: str = tok.pos_ + "." + tok.lemma_.strip().lower()

            if tok_key in known_lemma:
                tok_idx: int = known_lemma.index(tok_key)

                lex_graph.add_edge(
                    node_id,
                    tok_idx,
                    rel="COMPOUND_ELEMENT_OF",
                )

    if prev_known:
        # promote a previous Lemma node to an Entity
        node: dict = lex_graph.nodes[node_id]
        node["kind"] = "Entity"
        node["chunk"] = ent.chunk_id
        node["count"] += 1

        # select the more specific label
        if "label" not in node or node["label"] == "NP":
            node["label"] = ent.label

    if debug:
        print(ent)


def extract_relations(
    known_lemma: List[str],
    lex_graph: Graph,
    span_decoder: Dict[tuple, Entity],
    sent_map: Dict[Span, int],
    doc: Doc,
    chunk: TextChunk,
    *,
    debug: bool = False,
) -> None:
    """
    Extract the relations inferred by `GLiREL` adding these to the graph.
    """
    relations: List[dict] = sorted(
        doc._.relations,
        key=lambda item: item["score"],
        reverse=True,
    )

    for item in relations:
        src_loc: Tuple[int] = tuple(item["head_pos"])
        dst_loc: Tuple[int] = tuple(item["tail_pos"])
        redact_rel: bool = False

        if src_loc not in span_decoder:
            if debug:
                print("MISSING src entity:", item["head_text"], item["head_pos"])

            src_ent: Entity = make_entity(
                span_decoder,
                sent_map,
                doc[item["head_pos"][0] : item["head_pos"][1]],
                chunk,
                debug=debug,
            )

            if src_ent.key in STOP_WORDS:
                redact_rel = True
            else:
                extract_entity(known_lemma, lex_graph, src_ent, debug=debug)

        if dst_loc not in span_decoder:
            if debug:
                print("MISSING dst entity:", item["tail_text"], item["tail_pos"])

            dst_ent: Entity = make_entity(
                span_decoder,
                sent_map,
                doc[item["tail_pos"][0] : item["tail_pos"][1]],
                chunk,
                debug=debug,
            )

            if dst_ent.key in STOP_WORDS:
                redact_rel = True
            else:
                extract_entity(known_lemma, lex_graph, dst_ent, debug=debug)

        # link the connected nodes
        if not redact_rel:
            src_ent = span_decoder[src_loc]
            dst_ent = span_decoder[dst_loc]

            rel: str = item["label"].strip().replace(" ", "_").upper()
            prob: float = round(item["score"], 3)

            if debug:
                print(f"{src_ent.text} -> {rel} -> {dst_ent.text} | {prob}")

            lex_graph.add_edge(
                src_ent.node,
                dst_ent.node,
                rel=rel,
                prob=prob,
            )


def calc_quantile_bins(
    num_rows: int,
    *,
    amplitude: int = 4,
) -> ndarray:
    """
    Calculate the bins to use for a quantile stripe,
    using [`numpy.linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)

        num_rows:
    number of rows in the target dataframe

        returns:
    calculated bins, as a `numpy.ndarray`
    """
    granularity = max(round(math.log(num_rows) * amplitude), 1)

    return linspace(
        0,
        1,
        num=granularity,
        endpoint=True,
    )


def stripe_column(
    values: list,
    bins: int,
) -> ndarray:
    """
    Stripe a column in a dataframe, by interpolating quantiles into a set of discrete indexes.

        values:
    list of values to stripe

        bins:
    quantile bins; see [`calc_quantile_bins()`](#calc_quantile_bins-function)

        returns:
    the striped column values, as a `numpy.ndarray`
    """
    s = Series(values)
    q = s.quantile(bins, interpolation="nearest")

    try:
        stripe = digitize(values, q) - 1
        return stripe
    except ValueError as ex:
        # should never happen?
        print("ValueError:", str(ex), values, s, q, bins)
        raise


def root_mean_square(values: List[float]) -> float:
    """
    Calculate the [*root mean square*](https://mathworld.wolfram.com/Root-Mean-Square.html)
    of the values in the given list.

        values:
    list of values to use in the RMS calculation

        returns:
    RMS metric as a float
    """
    s: float = sum(map(lambda x: float(x) ** 2.0, values))
    n: float = float(len(values))

    return math.sqrt(s / n)


def connect_entities(
    lex_graph: Graph,
    span_decoder: Dict[tuple, Entity],
) -> None:
    """
    Connect entities which co-occur within the same sentence.
    """
    ent_map: Dict[int, Set[int]] = defaultdict(set)

    for ent in span_decoder.values():
        if ent.node is not None:
            ent_map[ent.sent_id].add(ent.node)

    for sent_id, nodes in ent_map.items():
        for pair in itertools.combinations(list(nodes), 2):
            if not lex_graph.has_edge(*pair):
                lex_graph.add_edge(
                    pair[0],
                    pair[1],
                    rel="CO_OCCURS_WITH",
                    prob=1.0,
                )


def run_textrank(
    lex_graph: Graph,
) -> DataFrame:
    """
    Run eigenvalue centrality (i.e., _Personalized PageRank_) to rank the entities.
    """
    # build a dataframe of node ranks and counts

    data_list = [
        {
            "node_id": node,
            "weight": rank,
            "count": lex_graph.nodes[node]["count"],
        }
        for node, rank in pagerank(lex_graph, alpha=TR_ALPHA, weight="count").items()
    ]
    df_rank: DataFrame = DataFrame.from_dict(data_list)

    # normalize by column and calculate quantiles
    df1: DataFrame = df_rank[["count", "weight"]].apply(lambda x: x / x.max(), axis=0)
    bins: ndarray = calc_quantile_bins(len(df1.index))

    # stripe each columns
    df2: DataFrame = DataFrame(
        [stripe_column(values, bins) for _, values in df1.items()]
    ).T

    # renormalize the ranks
    df_rank["rank"] = df2.apply(root_mean_square, axis=1)
    rank_col: ndarray = df_rank["rank"].to_numpy()
    rank_col /= sum(rank_col)
    df_rank["rank"] = rank_col

    # move the ranked weights back into the graph
    for _, row in df_rank.iterrows():
        node: int = row["node_id"]
        lex_graph.nodes[node]["rank"] = row["rank"]

    df: DataFrame = DataFrame(
        [
            node_attr
            for node, node_attr in lex_graph.nodes(data=True)
            if node_attr["kind"] == "Entity"
        ]
    ).sort_values(by=["rank"], ascending=False)

    return df


def abstract_overlay(
    file: str,
    chunk_list: List[TextChunk],
    lex_graph: Graph,
    sem_overlay: Graph,
) -> None:
    """
    Abstract a _semantic overlay_ from the lexical graph -- in other words
    which nodes and edges get promoted up to the next level?

    Also connect the extracted entities with their source chunks, where
    the latter first-class citizens within the KG.
    """
    kept_nodes: Set[int] = set()
    skipped_rel: Set[str] = set(["FOLLOWS_LEXICALLY", "COMPOUND_ELEMENT_OF"])

    chunk_nodes: Dict[int, str] = {
        chunk.uid: f"chunk_{chunk.uid}" for chunk in chunk_list
    }

    for chunk_id, node_id in chunk_nodes.items():
        sem_overlay.add_node(
            node_id,
            kind="Chunk",
            chunk=chunk_id,
            file=file,
        )

    for node_id, node_attr in lex_graph.nodes(data=True):
        if node_attr["kind"] == "Entity":
            kept_nodes.add(node_id)
            count: int = node_attr["count"]

            if not sem_overlay.has_node(node_id):
                sem_overlay.add_node(
                    node_id,
                    kind="Entity",
                    key=node_attr["key"],
                    text=node_attr["text"],
                    label=node_attr["label"],
                    count=count,
                )
            else:
                sem_overlay.nodes[node_id]["count"] += count

            sem_overlay.add_edge(
                node_id,
                chunk_nodes[node_attr["chunk"]],
                rel="WITHIN",
                weight=node_attr["rank"],
            )

    for src_id, dst_id, edge_attr in lex_graph.edges(data=True):
        if src_id in kept_nodes and dst_id in kept_nodes:
            rel: str = edge_attr["rel"]
            prob: float = 1.0

            if "prob" in edge_attr:
                prob = edge_attr["prob"]

            if rel not in skipped_rel:
                if not sem_overlay.has_edge(src_id, dst_id):
                    sem_overlay.add_edge(
                        src_id,
                        dst_id,
                        rel=rel,
                        prob=prob,
                    )
                else:
                    sem_overlay[src_id][dst_id]["prob"] = max(
                        prob,
                        sem_overlay.edges[(src_id, dst_id)]["prob"],
                    )


def gen_pyvis(
    graph: Graph,
    html_file: str,
    *,
    num_docs: int = 1,
    notebook: bool = False,
) -> None:
    """
    Use `pyvis` to provide an interactive visualization of the graph layers.
    """
    pv_net: pyvis.network.Network = pyvis.network.Network(
        height="900px",
        width="100%",
        notebook=notebook,
        cdn_resources="remote",
    )

    for node_id, node_attr in graph.nodes(data=True):
        if node_attr.get("kind") == "Entity":
            color: str = "hsla(65, 46%, 58%, 0.80)"
            size: int = round(
                20 * math.log(1.0 + math.sqrt(float(node_attr.get("count"))) / num_docs)
            )
            label: str = node_attr.get("text")
            title: str = node_attr.get("key")
        else:
            color = "hsla(306, 45%, 57%, 0.95)"
            size = 5
            label = node_id
            title = node_attr.get("file")

        pv_net.add_node(
            node_id,
            label=label,
            title=title,
            color=color,
            size=size,
        )

    for src_node, dst_node, edge_attr in graph.edges(data=True):
        pv_net.add_edge(
            src_node,
            dst_node,
            title=edge_attr.get("rel"),
        )

        pv_net.toggle_physics(True)
        pv_net.show_buttons(filter_=["physics"])
        pv_net.save_graph(html_file)
