#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import sys
from pathlib import Path
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
    random_layout,
    draw,
    spring_layout,
)
import matplotlib
from operator import itemgetter
import random
from itertools import combinations

random.seed(9001)
from random import randint
import statistics
import textwrap
import matplotlib.pyplot as plt
from typing import Iterator, Dict, List

matplotlib.use("Agg")

__author__ = "Your Name"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Your Name"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Your Name"
__email__ = "your@email.fr"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with open(fastq_file, "r", encoding="utf-8") as filin:
        for _ in filin:  # ligne @identifiant (consommée par la boucle)
            yield next(filin).strip()  # lecture
            next(filin)  # +
            next(filin)  # qualité
    pass


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i in range(len(read) - kmer_size + 1):
        yield read[i:i + kmer_size]
    pass


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    kmer_dict = {}
    for read in read_fastq(fastq_file):
        for kmer in cut_kmer(read, kmer_size):
            kmer_dict[kmer] = kmer_dict.get(kmer, 0) + 1
    return kmer_dict
    pass


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    graph = DiGraph()
    for kmer, weight in kmer_dict.items():
        graph.add_edge(kmer[:-1], kmer[1:], weight=weight)
    return graph
    pass


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
        start = 0 if delete_entry_node else 1
        end = len(path) if delete_sink_node else len(path) - 1
        graph.remove_nodes_from(path[start:end])
    return graph
    pass

def _std(data: List[float]) -> float:
    """Standard deviation, 0 if fewer than 2 values (stdev would fail).
 
    :param data: (list) A list of numbers
    :return: (float) The standard deviation
    """
    if len(data) < 2:
        return 0.0
    return statistics.stdev(data)

def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    if _std(weight_avg_list) > 0:
        best = weight_avg_list.index(max(weight_avg_list))
    elif _std(path_length) > 0:
        best = path_length.index(max(path_length))
    else:
        best = randint(0, len(path_list) - 1)
    to_remove = path_list[:best] + path_list[best + 1 :]
    return remove_paths(graph, to_remove, delete_entry_node, delete_sink_node)
    pass


def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    path_list = list(all_simple_paths(graph, ancestor_node, descendant_node))
    if len(path_list) < 2:
        return graph
    path_length = [len(path) for path in path_list]
    weight_avg_list = [path_average_weight(graph, path) for path in path_list]
    return select_best_path(
        graph,
        path_list,
        path_length,
        weight_avg_list,
        delete_entry_node=False,
        delete_sink_node=False,
    )
    pass


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    while True:
        bubble = False
        ancestor, node = None, None
        for node in graph.nodes:
            predecessors = list(graph.predecessors(node))
            if len(predecessors) > 1:
                for pred_i, pred_j in combinations(predecessors, 2):
                    ancestor = lowest_common_ancestor(graph, pred_i, pred_j)
                    if ancestor is not None:
                        bubble = True
                        break
            if bubble:
                break
        if not bubble:
            return graph
        graph = solve_bubble(graph, ancestor, node)
    pass


def solve_entry_tips(graph: DiGraph, starting_nodes: List[str]) -> DiGraph:
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of starting nodes
    :return: (nx.DiGraph) A directed graph object
    """
    while True:
        tip_found = False
        for node in list(graph.nodes):
            if len(list(graph.predecessors(node))) < 2:
                continue
            path_list = []
            for start in starting_nodes:
                if start in graph and has_path(graph, start, node):
                    path_list += list(all_simple_paths(graph, start, node))
            if len(path_list) < 2:
                continue
            path_length = [len(path) for path in path_list]
            weight_avg_list = [path_average_weight(graph, p) for p in path_list]
            graph = select_best_path(
                graph,
                path_list,
                path_length,
                weight_avg_list,
                delete_entry_node=True,
                delete_sink_node=False,
            )
            tip_found = True
            break
        if not tip_found:
            return graph
        starting_nodes = get_starting_nodes(graph)
    pass


def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    while True:
        tip_found = False
        for node in list(graph.nodes):
            if len(list(graph.successors(node))) < 2:
                continue
            path_list = []
            for end in ending_nodes:
                if end in graph and has_path(graph, node, end):
                    path_list += list(all_simple_paths(graph, node, end))
            if len(path_list) < 2:
                continue
            path_length = [len(path) for path in path_list]
            weight_avg_list = [path_average_weight(graph, p) for p in path_list]
            graph = select_best_path(
                graph,
                path_list,
                path_length,
                weight_avg_list,
                delete_entry_node=False,
                delete_sink_node=True,
            )
            tip_found = True
            break
        if not tip_found:
            return graph
        ending_nodes = get_sink_nodes(graph)
    pass


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    return [node for node in graph.nodes if not any(graph.predecessors(node))]
    pass


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    return [node for node in graph.nodes if not any(graph.successors(node))]
    pass


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigs = []
    for start in starting_nodes:
        for end in ending_nodes:
            if has_path(graph, start, end):
                for path in all_simple_paths(graph, start, end):
                    # 1er k-mer du nœud d'entrée, puis 1 nucléotide par nœud
                    contig = path[0] + "".join(node[-1] for node in path[1:])
                    contigs.append((contig, len(contig)))
    return contigs
    pass


def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with open(output_file, "w", encoding="utf-8") as filout:
        for i, (contig, length) in enumerate(contigs_list):
            filout.write(f">contig_{i} len={length}\n")
            filout.write(f"{textwrap.fill(contig, width=80)}\n")
    pass


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    # print(elarge)
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        graph, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    # 1. Lecture du fichier et construction du graphe
    kmer_dict = build_kmer_dict(args.fastq_file, args.kmer_size)
    graph = build_graph(kmer_dict)
 
    # 2. Résolution des bulles
    graph = simplify_bubbles(graph)
 
    # 3. Résolution des pointes d'entrée et de sortie
    graph = solve_entry_tips(graph, get_starting_nodes(graph))
    graph = solve_out_tips(graph, get_sink_nodes(graph))
 
    # 4. Écriture du/des contigs
    starting_nodes = get_starting_nodes(graph)
    sink_nodes = get_sink_nodes(graph)
    contigs = get_contigs(graph, starting_nodes, sink_nodes)
    save_contigs(contigs, args.output_file)

    # Fonctions de dessin du graphe
    # A decommenter si vous souhaitez visualiser un petit
    # graphe
    # Plot the graph
    # if args.graphimg_file:
    #     draw_graph(graph, args.graphimg_file)


if __name__ == "__main__":  # pragma: no cover
    main()
