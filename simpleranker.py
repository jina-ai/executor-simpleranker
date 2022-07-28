__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"
import warnings

from itertools import groupby
from typing import Dict, Optional

from docarray.score import NamedScore
from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger


class SimpleRanker(Executor):
    """
    :class:`SimpleRanker` aggregates the score of the matched doc from the
        matched chunks. For each matched doc, the score is aggregated from all the
        matched chunks belonging to that doc. The score of the document is the minimum
        score (min distance) among the chunks. The aggregated matches are sorted by
        score (ascending).
    """

    def __init__(
        self,
        metric: str = 'cosine',
        ranking: str = 'min',
        access_paths: str = '@r',
        traversal_paths: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        :param metric: the distance metric used in `scores`
        :param renking: The ranking function that the executor uses. There are multiple
            options:
            - min: Select minimum score/distance and sort by minimum
            - max: Select maximum score/distance and sort by maximum
            - mean_min: Calculate mean score/distance and sort by minimum mean
            - mean_max: Calculate mean score/distance and sort by maximum mean
        :param access_paths: traverse path on docs, e.g. ['r'], ['c']
        :param traversal_paths: please use access_paths
        """
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger('ranker')
        self.metric = metric
        assert ranking in ['min', 'max', 'mean_min', 'mean_max']
        self.ranking = ranking
        self.logger.warning(f'ranking = {self.ranking}')
        if traversal_paths is not None:
            self.access_paths = traversal_paths
            warnings.warn("'traversal_paths' will be deprecated in the future, please use 'access_paths'.",
                          DeprecationWarning,
                          stacklevel=2)
        else:
            self.access_paths = access_paths

    @requests(on='/search')
    def rank(self, docs: DocumentArray, parameters: Dict, *args, **kwargs):
        access_paths = parameters.get('access_paths', self.access_paths)

        for doc in docs[access_paths]:
            matches_of_chunks = []
            matches_of_chunks.extend(doc.matches)
            doc.matches.clear()
            for chunk in doc.chunks:
                matches_of_chunks.extend(chunk.matches)
            groups = groupby(
                sorted(matches_of_chunks, key=lambda d: d.parent_id or d.id),
                lambda d: d.parent_id or d.id,
            )
            for key, group in groups:
                chunk_match_list = list(group)
                if self.ranking == 'min':
                    chunk_match_list = DocumentArray(
                        sorted(
                            chunk_match_list, key=lambda m: m.scores[self.metric].value
                        )
                    )
                elif self.ranking == 'max':
                    chunk_match_list = DocumentArray(
                        sorted(
                            chunk_match_list, key=lambda m: -m.scores[self.metric].value
                        )
                    )
                match = chunk_match_list[0]
                match.id = chunk_match_list[0].parent_id
                if self.ranking in ['mean_min', 'mean_max']:
                    scores = [el.scores[self.metric].value for el in chunk_match_list]
                    match.scores[self.metric] = NamedScore(
                        value=sum(scores) / len(scores), op_name=self.ranking
                    )
                doc.matches.append(match)
            if self.ranking in ['min', 'mean_min']:
                doc.matches = DocumentArray(
                    sorted(doc.matches, key=lambda d: d.scores[self.metric].value)
                )
            elif self.ranking in ['max', 'mean_max']:
                doc.matches = DocumentArray(
                    sorted(doc.matches, key=lambda d: -d.scores[self.metric].value)
                )
