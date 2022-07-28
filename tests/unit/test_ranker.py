__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import pytest
from simpleranker import SimpleRanker


@pytest.mark.parametrize('access_paths', ['@r', '@c'])
@pytest.mark.parametrize('ranking', ['min', 'max'])
def test_ranking(documents_chunk, documents_chunk_chunk, access_paths, ranking):
    ranker = SimpleRanker(
        metric='cosine',
        ranking=ranking,
        access_paths=access_paths,
    )
    if access_paths == '@r':
        ranking_docs = documents_chunk
    else:
        ranking_docs = documents_chunk_chunk

    ranker.rank(ranking_docs, parameters={})
    assert ranking_docs

    for doc in ranking_docs[access_paths]:
        assert doc.matches
        for i in range(len(doc.matches) - 1):
            match = doc.matches[i]
            assert match.tags
            if ranking == 'min':
                assert (
                    match.scores['cosine'].value
                    <= doc.matches[i + 1].scores['cosine'].value
                )
            else:
                assert (
                    match.scores['cosine'].value
                    >= doc.matches[i + 1].scores['cosine'].value
                )


@pytest.mark.parametrize('ranking', ['mean_min', 'mean_max'])
def test_mean_ranking(documents_chunk, ranking):
    access_paths = '@r'
    ranker = SimpleRanker(
        metric='cosine',
        ranking=ranking,
        access_paths=access_paths,
    )
    ranking_docs = documents_chunk

    mean_scores = []
    for doc in ranking_docs[0].chunks:
        scores = []
        for match in doc.matches:
            scores.append(match.scores['cosine'])
        mean_scores.append(sum([s.value for s in scores]) / 10)
    mean_scores.sort(reverse=ranking == 'mean_max')
    ranker.rank(ranking_docs, parameters={})
    assert ranking_docs

    for doc in ranking_docs[access_paths]:
        assert doc.matches
        for i in range(len(doc.matches) - 1):
            match = doc.matches[i]
            assert match.tags
            assert match.scores['cosine'].value == pytest.approx(mean_scores[i], 1e-5)


@pytest.mark.parametrize('access_paths', ['@r'])
@pytest.mark.parametrize('ranking', ['min', 'max'])
def test_ranking_no_chunks(documents_no_chunk, access_paths, ranking):
    ranker = SimpleRanker(
        metric='cosine',
        ranking=ranking,
        access_paths=access_paths,
    )
    ranking_docs = documents_no_chunk

    ranker.rank(ranking_docs, parameters={})
    assert ranking_docs

    for doc in ranking_docs[access_paths]:
        assert doc.matches
        for i in range(len(doc.matches) - 1):
            match = doc.matches[i]
            assert match.tags
            if ranking == 'min':
                assert (
                    match.scores['cosine'].value
                    <= doc.matches[i + 1].scores['cosine'].value
                )
            else:
                assert (
                    match.scores['cosine'].value
                    >= doc.matches[i + 1].scores['cosine'].value
                )

